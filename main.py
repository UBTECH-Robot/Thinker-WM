"""ThinkerWM inference entry point (ThinkerWMPipeline + CLI).

Python API::

    from main import ThinkerWMPipeline

    pipe = ThinkerWMPipeline("configs/inference_example.yaml",
                             "/path/to/checkpoint_step_xxx")
    out = pipe.infer(
        mode="wm",
        prompt="Pick up the red block.",
        first_frame="frame0.png",       # Image path / PIL.Image / [3,H,W] tensor
        robot_name="agibot_beta",       # Supports configured aliases
        depth_first="depth0.npy",       # Optional: provided -> also produces depth
    )
    out["video"]    # [3, T, H, W] float [0,1] (T = num_video_frames)
    out["depth"]    # [3, T, H, W] float [0,1] codec domain (only when depth_first provided)
    out["actions"]  # [action_chunk_size, action_dim] float (varies by mode)

CLI (--config / --checkpoint default to configs/inference_example.yaml and
model_weights/checkpoint)::

    python main.py infer --mode wm \
        --prompt "Pick up the red block." --first-frame frame0.png \
        --output outputs/wm_demo

Six modes (mode):
  wam    prompt + first_frame + robot (+state optional) -> video + actions (+depth)
  wm     prompt + first_frame (+robot/depth_first optional) -> video (+depth)
  vla    prompt + first_frame + robot (+state optional) -> actions
  a2v    first_frame + actions + robot (+state optional) -> video
  v2a    video (exactly 1+num_video_frames frames) + robot (+state optional) -> actions
  uncond first_frame + robot (+state optional) -> video + actions (empty-text condition)

state is optional everywhere: without it, action output is relative EEF
(delta pos + relative 6D rotation); with it, actions are restored to absolute
quantities using the state as anchor.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
import yaml

from thinkerwm import ThinkerWM, ThinkerWMConfig, MODES
from thinkerwm.thinker_wm import action_stats
from thinkerwm.thinker_wm import depth_codec
from thinkerwm.thinker_wm import io as twio
from thinkerwm.thinker_wm import infer as twm_infer

logger = logging.getLogger("thinkerwm")

# Repo root (main.py directory): relative paths in config are resolved relative to this,
# ensuring correct operation from any cwd.
_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))


def _resolve_repo_path(path: str) -> str:
    """Resolve relative paths against repo root; absolute paths/empty values returned as-is."""
    if not path or os.path.isabs(path):
        return path
    return os.path.join(_REPO_ROOT, path)

# ---------------------------------------------------------------------------
# Inference hyperparameter defaults (can also be overridden via pipe.infer(...) kwargs)
# ---------------------------------------------------------------------------
VIDEO_NUM_INFERENCE_STEPS = 50
GUIDANCE_SCALE = 5.0            # video CFG; 1.0 = disabled
ACTION_GUIDANCE_SCALE = 1.0     # action default: no CFG
ACTION_NUM_INFERENCE_STEPS = 10  # action denoising steps (fewer than video)
ACTION_SMOOTH_ALPHA = 0.0       # action temporal EMA smoothing; 0 = disabled
DEPTH_SMOOTH_ALPHA = 0.0        # depth temporal EMA smoothing; 0 = disabled
OUTPUT_FPS = 5                  # mp4 output frame rate


# ---------------------------------------------------------------------------
# Config: YAML -> ThinkerWMConfig
# ---------------------------------------------------------------------------

def _robot_configs_and_layout(
    robot_entries: Dict[str, Any],
    default_action_dim: int,
    unified_action_dim: int,
    unified_bands: Optional[Dict[str, Any]],
) -> tuple:
    """dataset.robot_names -> (robot_configs, robot_layout, alias_map);
    -1 in dim_map = dropped dim."""
    bands = None
    if isinstance(unified_bands, dict) and unified_bands:
        bands = {
            str(band): (int(span[0]), int(span[1]))
            for band, span in unified_bands.items()
        }
    robot_configs: Dict[str, int] = {}
    layout: Dict[str, Dict[str, Any]] = {}
    alias_map: Dict[str, str] = {}
    used_indices = set()
    U = int(unified_action_dim)
    for order, (name, value) in enumerate(robot_entries.items()):
        name = str(name)
        if isinstance(value, dict):
            dim = int(value.get("action_dim", default_action_dim))
            offset = int(value.get("unified_offset", 0))
            token_index = int(value.get("robot_token_index", order))
            raw_map = value.get("unified_dim_map", None)
            raw_groups = value.get("unified_rot_groups", None)
            raw_gates = value.get("unified_validity_gates", None)
            eef_relative = bool(value.get("eef_relative", True))
            aliases = value.get("aliases", []) or []
        else:
            dim = default_action_dim if value is None else int(value)
            offset, token_index = 0, order
            raw_map = raw_groups = raw_gates = None
            eef_relative = True
            aliases = []
        robot_configs[name] = dim
        for alias in aliases:
            alias_map[str(alias)] = name

        dim_map = None
        active_cols: list = []
        if raw_map is not None:
            dim_map = [int(v) for v in raw_map]
            if "unified_offset" in value:
                raise ValueError(
                    f"robot {name}: unified_dim_map and unified_offset are mutually exclusive")
            if len(dim_map) != dim:
                raise ValueError(
                    f"robot {name}: unified_dim_map length {len(dim_map)}"
                    f" != action_dim {dim}")
            active_cols = [c for c in dim_map if c >= 0]
            if len(set(active_cols)) != len(active_cols):
                raise ValueError(f"robot {name}: unified_dim_map has duplicate column indices")
            if any(c < -1 for c in dim_map) or (active_cols and max(active_cols) >= U):
                raise ValueError(
                    f"robot {name}: unified_dim_map index out of range (allowed -1 or [0, {U}))")
            if bands is not None:
                for col in active_cols:
                    if not any(s <= col < e for s, e in bands.values()):
                        raise ValueError(
                            f"robot {name}: unified_dim_map column {col} is not within any "
                            "registered unified_bands slot")
        elif offset + dim > U:
            raise ValueError(
                f"robot {name}: unified_offset {offset} + action_dim {dim}"
                f" exceeds unified_action_dim {U}")

        rot_groups = None
        if raw_groups is not None:
            if dim_map is None:
                raise ValueError(
                    f"robot {name}: unified_rot_groups requires unified_dim_map")
            rot_groups = []
            for gi, group in enumerate(raw_groups):
                local = [int(d) for d in group["local"]]
                cols = [int(c) for c in group["cols"]]
                repr_name = str(group.get("repr", "rotvec"))
                quat_order = str(group.get("quat_order", "xyzw"))
                for d in local:
                    if d < 0 or d >= dim:
                        raise ValueError(
                            f"robot {name}: rot_groups[{gi}] local dim {d}"
                            f" exceeds action_dim {dim}")
                    if dim_map[d] != -1:
                        raise ValueError(
                            f"robot {name}: rot_groups[{gi}] local dim {d} in"
                            f" unified_dim_map must be -1 (taken over by rot group)")
                if len(cols) != 6 or len(set(cols)) != 6:
                    raise ValueError(
                        f"robot {name}: rot_groups[{gi}] cols must be exactly 6 unique columns")
                if any(c < 0 or c >= U for c in cols):
                    raise ValueError(
                        f"robot {name}: rot_groups[{gi}] cols out of range [0, {U})")
                if set(cols) & set(active_cols):
                    raise ValueError(
                        f"robot {name}: rot_groups[{gi}] cols conflict with dim_map active columns")
                if bands is not None:
                    hit = [
                        bn for bn, (s, e) in bands.items()
                        if all(s <= c < e for c in cols) and bn.endswith("_rot6")
                    ]
                    if len(hit) != 1:
                        raise ValueError(
                            f"robot {name}: rot_groups[{gi}] cols {cols} do not fully "
                            "fall within the same registered *_rot6 band")
                rot_groups.append({
                    "local": local, "repr": repr_name,
                    "cols": cols, "quat_order": quat_order,
                })

        validity_gates = None
        if raw_gates is not None:
            if dim_map is None:
                raise ValueError(
                    f"robot {name}: unified_validity_gates requires unified_dim_map")
            validity_gates = []
            used_gate_cols: set = set()
            for gi, gate in enumerate(raw_gates):
                local = int(gate["local"])
                cols = [int(c) for c in gate["cols"]]
                if local < 0 or local >= dim:
                    raise ValueError(
                        f"robot {name}: validity_gates[{gi}] local dim {local}"
                        f" exceeds action_dim {dim}")
                if dim_map[local] != -1:
                    raise ValueError(
                        f"robot {name}: validity_gates[{gi}] local dim {local} in"
                        " unified_dim_map must be -1")
                if len(cols) != 2 or not (0 <= cols[0] < cols[1] <= U):
                    raise ValueError(
                        f"robot {name}: validity_gates[{gi}] cols must be"
                        f" [start, end) with 0 <= start < end <= {U}")
                gate_cols = set(range(cols[0], cols[1]))
                if used_gate_cols & gate_cols:
                    raise ValueError(
                        f"robot {name}: validity_gates[{gi}] cols overlap with another gate")
                if bands is not None:
                    for col in gate_cols:
                        if not any(s <= col < e for s, e in bands.values()):
                            raise ValueError(
                                f"robot {name}: validity_gates[{gi}] column {col}"
                                " is not within any registered unified_bands slot")
                used_gate_cols |= gate_cols
                validity_gates.append({"local": local, "cols": (cols[0], cols[1])})

        if token_index in used_indices:
            raise ValueError(
                f"robot {name}: robot_token_index {token_index} already occupied")
        used_indices.add(token_index)
        layout[name] = {
            "offset": offset,
            "token_index": token_index,
            "dim_map": dim_map,
            "rot_groups": rot_groups,
            "eef_relative": eef_relative,
            "validity_gates": validity_gates,
        }
    return robot_configs, layout, alias_map


def load_config(path: str) -> ThinkerWMConfig:
    """Read inference YAML (common / model.wan / model.vlm / model.wm /
    dataset.robot_names / dataset.unified_bands) into ThinkerWMConfig;
    relative paths in config are resolved against repo root."""
    if not os.path.exists(path):
        path = _resolve_repo_path(path)  # fall back to repo root if not found under cwd
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"Config is not a valid YAML mapping: {path}")

    common = cfg.get("common", {}) or {}
    model = cfg.get("model", {}) or {}
    wan = model.get("wan", {}) or {}
    vlm = model.get("vlm", {}) or {}
    wm = model.get("wm", {}) or {}
    dataset = cfg.get("dataset", {}) or {}
    robot_entries = dataset.get("robot_names", {}) or {}

    default_action_dim = int(common.get("action_dim", 26))
    unified_action_dim = int(common.get("unified_action_dim", 64))
    robot_configs, robot_layout, alias_map = _robot_configs_and_layout(
        robot_entries, default_action_dim, unified_action_dim,
        dataset.get("unified_bands"),
    )

    config = ThinkerWMConfig(
        # Skeleton paths (structure config.json / VAE weight file / VLM directory); relative paths resolved against repo root
        vae_path=_resolve_repo_path(str(wan.get("vae_path", ""))),
        wan_config_path=_resolve_repo_path(str(wan.get("config_path", ""))),
        video_precision=str(wan.get("precision", "bfloat16")),
        vlm_checkpoint_path=_resolve_repo_path(str(vlm.get("checkpoint_path", ""))),
        # Data dimensions
        action_state_dim=int(common.get("state_dim", default_action_dim)),
        action_dim=default_action_dim,
        video_action_freq_ratio=int(common.get("video_action_freq_ratio", 6)),
        num_video_frames=int(common.get("num_video_frames", 32)),
        video_height=int(common.get("video_height", 480)),
        video_width=int(common.get("video_width", 640)),
        unified_action_dim=unified_action_dim,
        # Network structure
        video_encoder_layer_num=int(wm.get("video_encoder_layer_num", 5)),
        video_decoder_layer_num=int(wm.get("video_decoder_layer_num", 5)),
        action_encoder_layer_num=int(wm.get("action_encoder_layer_num", 5)),
        action_decoder_layer_num=int(wm.get("action_decoder_layer_num", 5)),
        und_layer_num=int(wm.get("und_layer_num", 6)),
        action_t_downsample=int(wm.get("action_t_downsample", 24)),
        flow_shift=float(wm.get("flow_shift", 5.0)),
        action_flow_shift=(
            float(wm["action_flow_shift"])
            if wm.get("action_flow_shift", None) is not None else None
        ),
        max_und_tokens=int(wm.get("max_und_tokens", 1000)),
        # Unified action space
        robot_configs=robot_configs,
        robot_layout=robot_layout,
        unified_bands=dataset.get("unified_bands"),
        action_norm_policy=str(wm.get("action_norm_policy", "zscore")),
        action_target_repr=str(wm.get("action_target_repr", "absolute_native")),
        # Depth third generation stream
        enable_depth=bool(wm.get("enable_depth", False)),
        depth_z_near=float(wm.get("depth_z_near", 0.15)),
        depth_z_far=float(wm.get("depth_z_far", 4.0)),
        depth_mode=str(wm.get("depth_mode", "log")),
        depth_detach_video_kv=bool(wm.get("depth_detach_video_kv", False)),
        depth_uni_clone_layers=int(wm.get("depth_uni_clone_layers", 0) or 0),
    )
    config._robot_alias_map = alias_map  # ThinkerWMConfig attachment (non-dataclass field)
    # Top-level optional key: per-robot action normalization statistics JSON asset path (empty = identity normalization).
    config._action_stats_path = _resolve_repo_path(str(cfg.get("action_stats_path", "") or ""))
    return config


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

Tensor3 = torch.Tensor  # [3, H, W]


def _as_first_frame_tensor(first_frame: Any, size: tuple) -> torch.Tensor:
    """Normalize first_frame input to [3, H, W] float32 [0,1]."""
    if isinstance(first_frame, str):
        return twio.load_image(first_frame, size=size)
    if isinstance(first_frame, torch.Tensor):
        t = first_frame.detach().cpu().float()
        if t.ndim == 4 and t.shape[0] == 1:
            t = t[0]
        if t.ndim != 3 or t.shape[0] != 3:
            raise ValueError(f"first_frame tensor must be [3,H,W], got {tuple(t.shape)}")
        if t.max() > 1.5:  # fallback for 0-255 input
            t = t / 255.0
        h, w = size
        if t.shape[-2:] != (h, w):
            t = torch.nn.functional.interpolate(
                t.unsqueeze(0), size=(h, w), mode="bilinear",
                align_corners=False).squeeze(0)
        return t.clamp(0, 1)
    # PIL.Image
    arr = np.asarray(first_frame.convert("RGB"), dtype=np.float32) / 255.0
    t = torch.from_numpy(arr).permute(2, 0, 1)
    h, w = size
    if t.shape[-2:] != (h, w):
        t = torch.nn.functional.interpolate(
            t.unsqueeze(0), size=(h, w), mode="bilinear",
            align_corners=False).squeeze(0)
    return t.contiguous()


class ThinkerWMPipeline:
    """ThinkerWM single-sample inference pipeline (single-GPU, single-process); see module docstring for usage."""

    def __init__(
        self,
        config: Union[str, ThinkerWMConfig],
        checkpoint: str,
        device: str = "cuda",
    ):
        if not str(device).startswith("cuda"):
            raise ValueError(
                "ThinkerWM inference requires a CUDA GPU (internals run under cuda autocast)")
        if isinstance(config, (str, os.PathLike)):
            config = load_config(str(config))
        if not isinstance(config, ThinkerWMConfig):
            raise TypeError("config must be a YAML path or ThinkerWMConfig")
        self.config = config
        self.device = torch.device(device)

        # VAE weights are not in the checkpoint; vae_path file must exist at construction time.
        if not config.vae_path or not os.path.exists(config.vae_path):
            raise FileNotFoundError(
                f"VAE weight file not found (VAE is not in checkpoint, must be provided separately): "
                f"{config.vae_path!r}")

        # Action normalization stats must be loaded before ThinkerWM construction (snapshotted into tensors during construction).
        stats_path = str(getattr(config, "_action_stats_path", "") or "").strip()
        if stats_path:
            action_stats.load_action_stats(stats_path)
        else:
            logger.info("action_stats_path not configured: applying identity normalization for unregistered robots")

        logger.info("Building ThinkerWM (random init) ...")
        self.model = ThinkerWM(config)
        logger.info("Loading checkpoint: %s", checkpoint)
        twm_infer.load_checkpoint(self.model, checkpoint, strict=True)
        # Move everything to target device (VAE wrapper is not nn.Module, unaffected by .to()).
        self.model.to(self.device)
        self.model.eval()
        self._alias_map = getattr(config, "_robot_alias_map", {}) or {}

    # ---- Input parsing helpers ----

    def _resolve_robot_name(self, robot_name: str) -> str:
        name = str(robot_name)
        name = self._alias_map.get(name, name)
        layout = self.config.robot_layout or {}
        if layout and name not in layout:
            raise ValueError(
                f"unknown robot_name: {robot_name!r} (available: {sorted(layout)}, "
                f"aliases: {sorted(self._alias_map)})")
        return name

    def _robot_action_dim(self, robot: Optional[str]) -> int:
        """Native state/action dimension for this robot."""
        if robot is not None:
            dim = (self.config.robot_configs or {}).get(robot)
            if dim is not None:
                return int(dim)
        return int(self.config.action_state_dim)

    def _state_tensor(self, state: Any, mode: str,
                      robot: Optional[str] = None) -> torch.Tensor:
        if state is None:
            raise ValueError(f"mode={mode} requires state (current frame robot state)")
        if isinstance(state, str):
            if state.endswith(".npy"):
                t = twio.load_npy(state).float().flatten()
            else:
                t = torch.tensor(
                    [float(x) for x in state.split(",")], dtype=torch.float32)
        else:
            t = torch.as_tensor(np.asarray(state), dtype=torch.float32).flatten()
        expected = self._robot_action_dim(robot)
        if t.numel() != expected:
            raise ValueError(
                f"state dimension {t.numel()} != robot {robot or 'default'} dimension"
                f" {expected}")
        return t.unsqueeze(0)  # [1, state_dim]

    def _robot_and_state(self, mode: str, robot_name, state):
        """Modes requiring action input/output: resolve robot_name (required)
        and state (optional). Without state, action output is relative EEF
        (delta pos + relative 6D rotation); with state it is restored to
        absolute quantities via the anchor."""
        name = self._resolve_robot_name(robot_name) if robot_name else None
        if name is None:
            raise ValueError(f"mode={mode} requires robot_name")
        state_t = None
        if state is not None:
            state_t = self._state_tensor(state, mode, robot=name)
        return name, state_t

    # ---- Main entry point ----

    @torch.no_grad()
    def infer(
        self,
        mode: str,
        *,
        prompt: Optional[str] = None,
        first_frame: Any = None,
        state: Any = None,
        actions: Any = None,
        video: Any = None,
        robot_name: Optional[str] = None,
        depth_first: Any = None,
        num_inference_steps: int = VIDEO_NUM_INFERENCE_STEPS,
        guidance_scale: float = GUIDANCE_SCALE,
        action_guidance_scale: float = ACTION_GUIDANCE_SCALE,
        action_num_inference_steps: Optional[int] = ACTION_NUM_INFERENCE_STEPS,
        action_smooth_alpha: float = ACTION_SMOOTH_ALPHA,
        depth_smooth_alpha: float = DEPTH_SMOOTH_ALPHA,
    ) -> Dict[str, torch.Tensor]:
        """Single-sample six-mode inference. Returns dict (batch dim already squeezed):

        - "video": [3, T, H, W] float [0,1] (wam/wm/a2v/uncond)
        - "depth": [3, T, H, W] float [0,1] codec domain (when depth_first provided and
          mode generates video; convert to meters via depth_codec.decode_to_depth_m)
        - "actions": [action_chunk_size, action_dim] float, robot's original dimension,
          reader convention space (wam/vla/v2a/uncond). Relative EEF quantities
          (delta pos + relative 6D rotation) when state is absent; absolute
          quantities when state is given.
        """
        mode = str(mode).lower()
        if mode not in MODES:
            raise ValueError(f"unknown mode: {mode!r} (available: {list(MODES)})")
        cfg = self.config
        size = (int(cfg.video_height), int(cfg.video_width))
        dev = self.device

        # -- first_frame --
        ff_t = None
        if first_frame is not None:
            ff_t = _as_first_frame_tensor(first_frame, size).unsqueeze(0).to(dev)
        elif mode != "v2a":
            raise ValueError(f"mode={mode} requires first_frame")

        # -- depth_first (optional; provided -> also produces depth) --
        df_t = None
        if depth_first is not None:
            if not cfg.enable_depth:
                raise ValueError(
                    "depth_first provided but config enable_depth=false (no depth branch)")
            if mode in ("vla", "v2a"):
                logger.warning("mode=%s does not generate video, depth_first ignored", mode)
            else:
                if isinstance(depth_first, str):
                    df_t = twio.load_depth_first(
                        depth_first, size,
                        z_near=cfg.depth_z_near, z_far=cfg.depth_z_far,
                        mode=cfg.depth_mode)
                else:
                    df_t = torch.as_tensor(
                        np.asarray(depth_first), dtype=torch.float32)
                    if df_t.ndim == 2:
                        df_t = df_t.unsqueeze(0).expand(3, -1, -1).contiguous()
                df_t = df_t.unsqueeze(0).to(dev)  # [1, 3, H, W] in [0,1]

        # -- Mode dispatch --
        if mode in ("wam", "wm", "vla", "uncond"):
            robot, state_t = None, None
            if mode == "wm":
                if robot_name is not None:
                    robot = self._resolve_robot_name(robot_name)
            else:
                robot, state_t = self._robot_and_state(mode, robot_name, state)
            if mode == "uncond":
                # uncond mode has empty-text condition; CFG is meaningless, force guidance_scale=1.0
                # since there is no text prompt to scale away from the unconditional direction.
                messages = None
                guidance_scale = 1.0
            else:
                if prompt is None or not str(prompt).strip():
                    raise ValueError(f"mode={mode} requires non-empty prompt")
                messages = [str(prompt)]
            result = twm_infer.inference_step(
                self.model,
                ff_t,
                state=state_t.to(dev) if state_t is not None else None,
                num_inference_steps=int(num_inference_steps),
                vlm_input_messages=messages,
                robot_names=[robot] if robot else None,
                guidance_scale=float(guidance_scale),
                action_guidance_scale=float(action_guidance_scale),
                action_num_inference_steps=action_num_inference_steps,
                action_smooth_alpha=float(action_smooth_alpha),
                depth_smooth_alpha=float(depth_smooth_alpha),
                predict_action=(mode in ("wam", "vla", "uncond")),
                predict_video=(mode != "vla"),
                depth_first=df_t,
                return_depth=df_t is not None,
                uncond_prompt=(mode == "uncond"),
            )
            if df_t is not None:
                frames, pred_actions, pred_depth = result
            else:
                frames, pred_actions = result
                pred_depth = None
            out: Dict[str, torch.Tensor] = {}
            if frames is not None:
                out["video"] = frames[0].float().cpu()
            if pred_depth is not None:
                out["depth"] = pred_depth[0].float().cpu()
            if pred_actions is not None:
                out["actions"] = pred_actions[0].float().cpu()
            return out

        if mode == "a2v":
            robot, state_t = self._robot_and_state(mode, robot_name, state)
            if actions is None:
                raise ValueError("mode=a2v requires actions ([chunk, action_dim] "
                                 "original dim, reader space)")
            act_t = self._actions_tensor(actions, robot)
            frames, _echo = twm_infer.inference_action_step(
                self.model,
                ff_t,
                state=state_t.to(dev) if state_t is not None else None,
                conditioning_actions=act_t.to(dev),
                num_inference_steps=int(num_inference_steps),
                vlm_input_messages=[""],  # a2v prompt is empty-text constant
                robot_names=[robot],
            )
            return {"video": frames[0].float().cpu()}

        # mode == "v2a"
        robot, state_t = self._robot_and_state(mode, robot_name, state)
        if video is None:
            raise ValueError("mode=v2a requires video (exactly 1+num_video_frames frames)")
        vid_t = self._video_tensor(video, size)  # [1, 3, 1+T, H, W]
        pred_actions = twm_infer.inference_v2a_step(
            self.model,
            vid_t.to(dev),
            state=state_t.to(dev) if state_t is not None else None,
            num_inference_steps=int(num_inference_steps),
            vlm_input_messages=[""],  # v2a prompt is empty-text constant
            robot_names=[robot],
        )
        return {"actions": pred_actions[0].float().cpu()}

    def _actions_tensor(self, actions: Any, robot: Optional[str] = None) -> torch.Tensor:
        if isinstance(actions, str):
            t = twio.load_npy(actions).float()
        else:
            t = torch.as_tensor(np.asarray(actions), dtype=torch.float32)
        if t.ndim != 2:
            raise ValueError(f"actions must be [chunk, action_dim], got {tuple(t.shape)}")
        chunk = int(self.config.action_chunk_size)
        if t.shape[0] != chunk:
            raise ValueError(
                f"actions steps {t.shape[0]} != action_chunk_size {chunk}"
                f" (= num_video_frames x video_action_freq_ratio)")
        expected = self._robot_action_dim(robot)
        if t.shape[1] != expected:
            raise ValueError(
                f"actions dimension {t.shape[1]} != robot {robot or 'default'} dimension"
                f" {expected}")
        return t.unsqueeze(0)  # [1, chunk, action_dim]

    def _video_tensor(self, video: Any, size: tuple) -> torch.Tensor:
        """[1, 3, 1+num_video_frames, H, W] float [0,1] (first frame = conditioning frame)."""
        if isinstance(video, str):
            t = twio.load_video(video, size=size)  # [T, 3, H, W]
        else:
            t = torch.as_tensor(np.asarray(video), dtype=torch.float32)
            if t.ndim != 4 or t.shape[1] != 3:
                raise ValueError(
                    f"video tensor must be [T,3,H,W], got {tuple(t.shape)}")
        expect = 1 + int(self.config.num_video_frames)
        if t.shape[0] != expect:
            raise ValueError(
                f"mode=v2a video must be exactly {expect} frames (first frame + "
                f"num_video_frames={self.config.num_video_frames}), "
                f"got {t.shape[0]} frames")
        return t.permute(1, 0, 2, 3).unsqueeze(0)  # [1, 3, T, H, W]

    # ---- Output saving ----

    def save_outputs(self, out: Dict[str, torch.Tensor], output_dir: str) -> List[str]:
        """Save infer output dict to disk: video.mp4 / depth.mp4 + depth_m.npy /
        actions.npy. Returns list of written file paths."""
        os.makedirs(output_dir, exist_ok=True)
        written = []
        if out.get("video") is not None:
            p = os.path.join(output_dir, "video.mp4")
            twio.save_video(out["video"], p, fps=OUTPUT_FPS)
            written.append(p)
        if out.get("depth") is not None:
            p = os.path.join(output_dir, "depth.mp4")
            twio.save_video(out["depth"], p, fps=OUTPUT_FPS)
            written.append(p)
            # [3,T,H,W] -> [T,3,H,W]; infer output depth is already in [0,1] encoding domain
            depth_m, _valid = depth_codec.decode_to_depth_m(
                out["depth"].permute(1, 0, 2, 3),
                z_near=self.config.depth_z_near, z_far=self.config.depth_z_far,
                mode=self.config.depth_mode, in_range="01")
            p = os.path.join(output_dir, "depth_m.npy")
            twio.save_npy(depth_m, p)
            written.append(p)
        if out.get("actions") is not None:
            p = os.path.join(output_dir, "actions.npy")
            twio.save_npy(out["actions"], p)
            written.append(p)
        return written


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="thinkerwm",
        description="ThinkerWM inference (single-GPU single-sample)",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("infer", help="Single-sample six-mode inference")
    p.add_argument("--config", default="configs/inference_example.yaml",
                   help="Inference config YAML (relative paths resolved against cwd, fallback to repo root)")
    p.add_argument("--checkpoint", default="model_weights/checkpoint",
                   help="Checkpoint directory or weight file (.pt/.bin)")
    p.add_argument("--mode", required=True, choices=list(MODES))
    p.add_argument("--prompt", default=None, help="Task text (required for wam/wm/vla)")
    p.add_argument("--first-frame", default=None,
                   help="First-frame image (required except for v2a)")
    p.add_argument("--state", default=None,
                   help="Current robot state: comma-separated float string or .npy path"
                        " (optional; without it actions are relative EEF, with it"
                        " they are restored to absolute quantities)")
    p.add_argument("--robot-name", default=None,
                   help="Robot name (supports configured aliases; required for wam/vla/a2v/v2a/uncond)")
    p.add_argument("--actions", default=None,
                   help="a2v: conditioning actions .npy [chunk, action_dim] (reader space)")
    p.add_argument("--video", default=None,
                   help="v2a: conditioning video file (exactly 1+num_video_frames frames)")
    p.add_argument("--depth-first", default=None,
                   help="First-frame depth condition (.npy [0,1] codec domain / uint16 mm / grayscale);"
                        " provided -> also produces depth (requires enable_depth=true in config)")
    p.add_argument("--device", default="cuda")
    p.add_argument("--output", required=True, help="Output directory")
    # Inference hyperparameter overrides
    p.add_argument("--num-inference-steps", type=int, default=VIDEO_NUM_INFERENCE_STEPS,
                   help=f"Video denoising steps (default: {VIDEO_NUM_INFERENCE_STEPS})")
    p.add_argument("--guidance-scale", type=float, default=None,
                   help=f"Video CFG scale (default: {GUIDANCE_SCALE}; forced to 1.0 in uncond mode)")
    p.add_argument("--action-guidance-scale", type=float, default=None,
                   help=f"Action CFG scale (default: {ACTION_GUIDANCE_SCALE})")
    p.add_argument("--action-num-inference-steps", type=int, default=None,
                   help=f"Action denoising steps (default: {ACTION_NUM_INFERENCE_STEPS})")
    p.add_argument("--action-smooth-alpha", type=float, default=None,
                   help=f"Action temporal EMA smoothing (default: {ACTION_SMOOTH_ALPHA})")
    p.add_argument("--depth-smooth-alpha", type=float, default=None,
                   help=f"Depth temporal EMA smoothing (default: {DEPTH_SMOOTH_ALPHA})")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = _build_parser().parse_args(argv)
    if args.command == "infer":
        # checkpoint relative path falls back to repo root when not found under cwd (same as config resolution).
        checkpoint = args.checkpoint
        if not os.path.isabs(checkpoint) and not os.path.exists(checkpoint):
            checkpoint = _resolve_repo_path(checkpoint)
        pipe = ThinkerWMPipeline(args.config, checkpoint, device=args.device)
        # Resolve hyperparameter overrides: use CLI value if explicitly provided, otherwise module default.
        gs = args.guidance_scale if args.guidance_scale is not None else GUIDANCE_SCALE
        ags = args.action_guidance_scale if args.action_guidance_scale is not None else ACTION_GUIDANCE_SCALE
        ans = args.action_num_inference_steps if args.action_num_inference_steps is not None else ACTION_NUM_INFERENCE_STEPS
        asa = args.action_smooth_alpha if args.action_smooth_alpha is not None else ACTION_SMOOTH_ALPHA
        dsa = args.depth_smooth_alpha if args.depth_smooth_alpha is not None else DEPTH_SMOOTH_ALPHA
        out = pipe.infer(
            args.mode,
            prompt=args.prompt,
            first_frame=args.first_frame,
            state=args.state,
            actions=args.actions,
            video=args.video,
            robot_name=args.robot_name,
            depth_first=args.depth_first,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=gs,
            action_guidance_scale=ags,
            action_num_inference_steps=ans,
            action_smooth_alpha=asa,
            depth_smooth_alpha=dsa,
        )
        written = pipe.save_outputs(out, args.output)
        for p in written:
            logger.info("written: %s", p)
        print("outputs:")
        for p in written:
            print(f"  {p}")
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
