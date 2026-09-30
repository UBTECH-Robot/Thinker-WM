"""ThinkerWM network structure: ThinkerWMConfig / ThinkerWM / UniWorldModel (inference logic in infer.py)."""

import json
import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import PIL.Image as Image
from transformers import AutoConfig, AutoProcessor, Qwen3VLForConditionalGeneration

from wan.modules.model import WanModel
from wan.modules.vae2_2 import Wan2_2_VAE

from ..decoder.action import ActionDecoder
from ..decoder.video import VideoDecoder
from ..encoder.action import ActionEncoder
from ..encoder.robot_tokens import RobotTokenEmbedding
from ..encoder.und import UndBranch
from ..encoder.video import VideoEncoder
from ..uni.branch import UniBranch, _compact_und_kv
from . import action_repr
from .action_stats import build_gripper_hand_policy

logger = logging.getLogger(__name__)


@dataclass
class ThinkerWMConfig:
    """ThinkerWM inference configuration."""

    vae_path: str = ""
    wan_config_path: str = ""      # Directory containing config.json (used only for structure)
    video_precision: str = "bfloat16"
    vlm_checkpoint_path: str = ""
    # und token count hard cap (uni branch rope freqs 1024 rows; tail truncated on overflow)
    max_und_tokens: int = 1000

    video_height: int = 480
    video_width: int = 640

    action_state_dim: int = 26
    action_dim: int = 26
    video_action_freq_ratio: int = 6
    action_t_downsample: int = 24
    num_video_frames: int = 32

    flow_shift: float = 5.0        # Video-side flow-matching sigma shift
    action_flow_shift: Optional[float] = None  # Action-side sigma shift; None = follows flow_shift

    video_encoder_layer_num: int = 5
    video_decoder_layer_num: int = 5
    action_encoder_layer_num: int = 5
    action_decoder_layer_num: int = 5
    und_layer_num: int = 6

    # Depth third generation stream (depth-as-video): follows video sigma/t/noise
    enable_depth: bool = False
    depth_z_near: float = 0.15
    depth_z_far: float = 4.0
    depth_mode: str = "log"
    depth_detach_video_kv: bool = False
    depth_uni_clone_layers: int = 0

    robot_configs: Optional[Dict[str, int]] = None
    unified_action_dim: int = 64
    # {name: {offset, token_index, dim_map(-1 = dropped dim), rot_groups,
    #         eef_relative, validity_gates}}
    robot_layout: Optional[Dict[str, Dict[str, Any]]] = None
    unified_bands: Optional[Dict[str, Any]] = None  # {band: [start, end)}
    action_norm_policy: str = "zscore"
    action_target_repr: str = "absolute_native"

    def __post_init__(self):
        self.action_chunk_size = self.num_video_frames * self.video_action_freq_ratio


class UniWorldModel(nn.Module):
    """encoder/uni/decoder branch assembly: video + action (+ depth) + und joint forward."""

    def __init__(self,
                 vlm_preprocess, vlm_model, wan_model: WanModel,
                 video_encoder_layer_num: int, video_decoder_layer_num: int,
                 action_encoder_layer_num: int, action_decoder_layer_num: int,
                 und_layer_num: int,
                 action_t_downsample: int = 24,
                 video_action_freq_ratio: int = 6,
                 unified_action_dim: int = 64,
                 robot_token_indices: Optional[Dict[str, int]] = None,
                 gradient_checkpointing: bool = False,
                 checkpoint_video_branch: bool = True,
                 checkpoint_uni_branch: bool = True,
                 max_und_tokens: int = 1000,
                 enable_depth: bool = False,
                 depth_uni_clone_layers: int = 0,
    ):
        super().__init__()
        assert (video_encoder_layer_num + video_decoder_layer_num
                < len(wan_model.blocks))

        wan_dim = wan_model.dim
        wan_ffn_dim = wan_model.ffn_dim
        wan_num_heads = wan_model.num_heads

        # wan_model.text_embedding only serves the T5 path, unused in this model, frozen as-is.
        for p in wan_model.text_embedding.parameters():
            p.requires_grad = False

        # Robot token table is owned by this module, passed as parameter to branches during encode/decode.
        self.robot_token_embedding = RobotTokenEmbedding(
            robot_token_indices or {}, max_robots=64, num_tokens=8, dim=wan_dim,
        )

        self.enable_depth = bool(enable_depth)
        video_gc = bool(gradient_checkpointing and checkpoint_video_branch)
        self.video_encoder = VideoEncoder(
            wan_model,
            video_encoder_layer_num,
            gradient_checkpointing=video_gc,
            enable_depth=self.enable_depth,
        )
        self.video_decoder = VideoDecoder(
            wan_model,
            video_decoder_layer_num,
            gradient_checkpointing=video_gc,
            enable_depth=self.enable_depth,
        )

        self.action_encoder = ActionEncoder(
            layer_num=action_encoder_layer_num,
            wan_dim=wan_dim, wan_ffn_dim=wan_ffn_dim, wan_num_heads=wan_num_heads,
            t_downsample=action_t_downsample,
            video_action_freq_ratio=video_action_freq_ratio,
            unified_action_dim=unified_action_dim,
        )
        self.action_decoder = ActionDecoder(
            layer_num=action_decoder_layer_num,
            wan_dim=wan_dim, wan_ffn_dim=wan_ffn_dim, wan_num_heads=wan_num_heads,
            t_downsample=action_t_downsample,
            video_action_freq_ratio=video_action_freq_ratio,
            unified_action_dim=unified_action_dim,
        )

        self.und_branch = UndBranch(
            vlm_preprocess, vlm_model,
            wan_dim, wan_ffn_dim,
            und_layer_num,
            max_und_tokens=max_und_tokens,
        )

        # und placeholder when prompt is missing: learnable null token.
        self.null_und_token = nn.Parameter(torch.zeros(1, 1, wan_dim).normal_(std=0.02))

        self.uni_branch = UniBranch(
            wan_model=wan_model,
            encode_layer_num=video_encoder_layer_num,
            decode_layer_num=video_decoder_layer_num,
            gradient_checkpointing=bool(gradient_checkpointing and checkpoint_uni_branch),
            # Action token i's RoPE time position corresponds to latent time i * t_downsample /
            # (ratio * 4) (4 = Wan2.2 VAE temporal stride).
            action_rope_time_scale=action_t_downsample / (video_action_freq_ratio * 4.0),
            depth_uni_clone_layers=int(depth_uni_clone_layers or 0) if self.enable_depth else 0,
        )

    def null_und_cache(self, B: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """und triplet when prompt is missing: (token, grid_sizes, seq_lens) -- 1 null token per sample."""
        token = self.null_und_token.expand(B, 1, -1)
        grid_sizes = torch.ones(B, 3, dtype=torch.long, device=token.device)
        seq_lens = torch.ones(B, dtype=torch.long, device=token.device)
        return token, grid_sizes, seq_lens

    def forward(self, t: torch.Tensor,
                vlm_input_message: list[str], vlm_input_first_frame: list[Image.Image],
                video: Optional[torch.Tensor], action: Optional[torch.Tensor],
                t_video: torch.Tensor = None, t_action: torch.Tensor = None,
                robot_names=None,
                return_list=False,
                und_cache=None,
                drop_prompt: bool = False,
                depth: Optional[torch.Tensor] = None, t_depth: torch.Tensor = None,
                depth_detach_video_kv: bool = False,
    ):
        """Joint forward; video/action/depth can be None (missing modality skipped, returns None).
        Depth is unidirectionally visible: video/action/und do not read depth."""

        if t_video is None:
            t_video = t
        if t_action is None:
            t_action = t
        if t_depth is None:
            t_depth = t_video

        # und (VLM understanding) is a global condition, computed once and cached; und_seq_lens are
        # per-sample real lengths, used by cross-attn to mask VLM padding positions.
        if und_cache is not None:
            und_token, und_grid_sizes, und_seq_lens = und_cache
        elif drop_prompt:
            if video is not None:
                b = video.shape[0]
            elif action is not None:
                b = action.shape[0]
            else:
                b = len(vlm_input_message)
            und_token, und_grid_sizes, und_seq_lens = self.null_und_cache(b)
        else:
            und_token, und_grid_sizes, und_seq_lens = self.und_branch.encode(
                vlm_input_message, vlm_input_first_frame)

        # und_token_init: raw und token before uni, used for downstream K/V concatenation.
        und_token_init = und_token
        video_token = video_grid_sizes = video_seq_lens = video_e = video_e0 = None
        if video is not None:
            video_token, video_grid_sizes, video_seq_lens, video_e, video_e0 = self.video_encoder.encode(
                video, t_video, und_token, context_lens=und_seq_lens,
                robot_names=robot_names,
                robot_token_embedding=self.robot_token_embedding)

        action_token = action_grid_sizes = action_seq_lens = action_e0 = None
        if action is not None:
            action_token, action_grid_sizes, action_seq_lens, action_e0 = self.action_encoder(
                action, t_action, und_token, robot_names=robot_names,
                context_lens=und_seq_lens,
                robot_token_embedding=self.robot_token_embedding)

        depth_token = depth_grid_sizes = depth_seq_lens = depth_e = depth_e0 = None
        if depth is not None:
            assert self.enable_depth, "forward(depth=...) requires enable_depth=True"
            depth_token, depth_grid_sizes, depth_seq_lens, depth_e, depth_e0 = self.video_encoder.encode_depth(
                depth, t_depth, und_token, context_lens=und_seq_lens,
                robot_names=robot_names,
                robot_token_embedding=self.robot_token_embedding)

        video_token, action_token, depth_token, und_token = self.uni_branch.uni_process(
            video_token, video_grid_sizes,
            action_token, action_grid_sizes,
            und_token, und_grid_sizes,
            t_video, t_action, video_seq_lens, action_seq_lens, und_seq_lens,
            depth_token=depth_token, depth_grid_sizes=depth_grid_sizes,
            depth_seq_lens=depth_seq_lens, t_depth=t_depth,
            depth_detach_video_kv=depth_detach_video_kv,
            und_token_init=und_token_init)

        # Decoder cross-attn K/V = compact concatenation [und token 0, uni output und token t].
        und_ctx, und_ctx_lens = _compact_und_kv(
            und_token_init, und_token, und_seq_lens)
        video_out = None
        if video_token is not None:
            video_out = self.video_decoder.decode(
                video_token, und_ctx, video_grid_sizes, video_seq_lens, video_e, video_e0,
                context_lens=und_ctx_lens,
                robot_names=robot_names,
                robot_token_embedding=self.robot_token_embedding)
        action_out = None
        if action_token is not None:
            action_out = self.action_decoder(
                action_token, und_ctx, action_grid_sizes, action_seq_lens, action_e0,
                robot_names=robot_names, context_lens=und_ctx_lens,
                robot_token_embedding=self.robot_token_embedding)

        depth_out = None
        if depth_token is not None:
            depth_out = self.video_decoder.decode_depth(
                depth_token, und_ctx, depth_grid_sizes, depth_seq_lens, depth_e, depth_e0,
                context_lens=und_ctx_lens,
                robot_names=robot_names,
                robot_token_embedding=self.robot_token_embedding)

        if not return_list:
            video_out = torch.stack(video_out, dim=0) if video_out is not None else None
            action_out = torch.stack(action_out, dim=0) if action_out is not None else None
            depth_out = torch.stack(depth_out, dim=0) if depth_out is not None else None

        return video_out, action_out, depth_out


class ThinkerWM(nn.Module):
    """ThinkerWM unified video-action(-depth) world model (structure + assembly + VAE encode/decode).
    See infer.py for the six inference modes."""

    @property
    def device(self) -> torch.device:
        """Current compute device (taken from world_model first parameter, automatically correct after transfer/load)."""
        return next(self.world_model.parameters()).device

    def __init__(self, config: ThinkerWMConfig):
        super().__init__()
        self.config = config
        self.dtype = torch.bfloat16

        logger.info("Loading VLM Preprocessor...")
        vlm_preprocess = AutoProcessor.from_pretrained(config.vlm_checkpoint_path)

        logger.info("Building Wan architecture from config.json (no pretrained weights)...")
        config_json_path = os.path.join(config.wan_config_path, "config.json")
        if not os.path.exists(config_json_path):
            raise FileNotFoundError(f"Wan config.json not found at {config_json_path}")
        with open(config_json_path, "r") as f:
            model_config = json.load(f)
        precision = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }[config.video_precision]
        wan_model = WanModel(**model_config).to(device="cuda", dtype=precision)

        # VAE is not in checkpoint, loaded separately from vae_path.
        self.vae = Wan2_2_VAE(vae_pth=config.vae_path, device=torch.device("cuda"))

        logger.info("Building VLM from config (no pretrained weights)...")
        vlm_config = AutoConfig.from_pretrained(
            config.vlm_checkpoint_path, trust_remote_code=True
        )
        # Random init on GPU (fallback to CPU if no GPU); weights fully overwritten by checkpoint.
        with torch.device("cuda" if torch.cuda.is_available() else "cpu"):
            try:
                vlm_model = Qwen3VLForConditionalGeneration._from_config(
                    vlm_config, dtype=self.dtype
                )
            except TypeError:  # Fallback for transformers version without dtype kwarg
                vlm_model = Qwen3VLForConditionalGeneration(vlm_config).to(dtype=self.dtype)
        for param in vlm_model.parameters():
            param.requires_grad = False

        self.world_model = UniWorldModel(
            vlm_preprocess, vlm_model, wan_model,
            video_encoder_layer_num=config.video_encoder_layer_num,
            video_decoder_layer_num=config.video_decoder_layer_num,
            action_encoder_layer_num=config.action_encoder_layer_num,
            action_decoder_layer_num=config.action_decoder_layer_num,
            und_layer_num=config.und_layer_num,
            action_t_downsample=config.action_t_downsample,
            video_action_freq_ratio=config.video_action_freq_ratio,
            unified_action_dim=config.unified_action_dim,
            robot_token_indices={k: v["token_index"] for k, v in (config.robot_layout or {}).items()},
            max_und_tokens=config.max_und_tokens,
            enable_depth=bool(getattr(config, "enable_depth", False)),
            depth_uni_clone_layers=int(getattr(config, "depth_uni_clone_layers", 0) or 0),
        ).to(dtype=self.dtype)

        if isinstance(self.vae, nn.Module):
            self._freeze_module(self.vae)

        action_flow_shift = getattr(config, "action_flow_shift", None)
        self._action_flow_shift = (
            float(action_flow_shift)
            if action_flow_shift is not None
            else float(config.flow_shift)
        )

        self._norm_policy = None
        if str(getattr(config, "action_norm_policy", "zscore")) == "gripper_hand_minmax":
            self._norm_policy = build_gripper_hand_policy(
                config.robot_layout or {}, config.unified_bands or {})

        self._repr_specs = None
        if str(getattr(config, "action_target_repr", "absolute_native")) == "relative_eef_rot6":
            self._repr_specs = self._build_repr_specs()

        self.log_parameter_counts()

    @staticmethod
    def _freeze_module(module: nn.Module) -> Tuple[int, int]:
        """Freeze all parameters of the module, returns (tensor_count, param_count)."""
        frozen_params = 0
        frozen_tensors = 0
        for param in module.parameters():
            if param.requires_grad:
                frozen_params += param.numel()
                frozen_tensors += 1
            param.requires_grad = False
        return frozen_tensors, frozen_params

    def log_parameter_counts(self):
        """Log total parameter count and requires_grad parameter count."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        logger.info("ThinkerWM parameter breakdown:")
        logger.info(f"  Total parameters: {total_params / 1e9:.2f}B")
        logger.info(f"  Trainable parameters: {trainable_params / 1e9:.2f}B")

    def encode_video(self, video_pixels: torch.Tensor) -> torch.Tensor:
        """VAE encode: pixels [B, C, T, H, W] [-1,1] -> latent [B, C', T', H', W']."""
        with torch.no_grad():
            return self.vae.encode(video_pixels)

    def decode_video(self, video_latents: torch.Tensor) -> torch.Tensor:
        """VAE decode: latent [B, C, T, H, W] -> pixels [B, C', T', H', W'] [-1,1]."""
        with torch.no_grad():
            video_pixels = []
            for i in range(video_latents.shape[0]):
                pixels = self.vae.decode([video_latents[i]])[0]
                video_pixels.append(pixels)
            return torch.stack(video_pixels, dim=0)

    _ROT6_BAND_SUFFIX = "_rot6"

    def _build_repr_specs(self) -> Dict[str, Dict[str, Any]]:
        """Build per-robot indices for relative_eef_rot6: eef pos local dims + rotation groups."""
        bands = {
            str(band): (int(span[0]), int(span[1]))
            for band, span in (self.config.unified_bands or {}).items()
        }
        specs: Dict[str, Dict[str, Any]] = {}
        for name, entry in (self.config.robot_layout or {}).items():
            dim_map = entry.get("dim_map")
            relative = bool(entry.get("eef_relative", True))
            pos_dims: List[int] = []
            if relative and dim_map and bands:
                for k, c in enumerate(dim_map):
                    c = int(c)
                    if c < 0:
                        continue
                    for band_name, (start, end) in bands.items():
                        if (
                            start <= c < end
                            and band_name.startswith("eef_")
                            and band_name.endswith("_pos")
                        ):
                            pos_dims.append(k)
                            break
            groups: List[Dict[str, Any]] = []
            for group in entry.get("rot_groups") or []:
                cols = [int(c) for c in group["cols"]]
                local = [int(d) for d in group["local"]]
                repr_name = str(group.get("repr", "rotvec"))
                if len(cols) != 6:
                    raise ValueError(
                        f"robot {name}: rot_groups cols must be exactly 6 columns (6D rotation), "
                        f"got {cols}"
                    )
                if len(local) != action_repr.repr_width(repr_name):
                    raise ValueError(
                        f"robot {name}: rot_groups local dimension count {len(local)} does not match repr "
                        f"{repr_name!r} width {action_repr.repr_width(repr_name)}"
                    )
                if bands:
                    hit = [
                        band_name
                        for band_name, (start, end) in bands.items()
                        if all(start <= c < end for c in cols)
                        and band_name.endswith(self._ROT6_BAND_SUFFIX)
                    ]
                    if len(hit) != 1:
                        raise ValueError(
                            f"robot {name}: rot_groups cols {cols} do not fully fall within the same "
                            f"*{self._ROT6_BAND_SUFFIX} band (matched {hit})"
                        )
                groups.append({
                    "local": torch.tensor(local, dtype=torch.long),
                    "cols": torch.tensor(cols, dtype=torch.long),
                    "repr": repr_name,
                    "quat_order": str(group.get("quat_order", "xyzw")),
                    "relative": relative,
                })
            specs[str(name)] = {
                "pos_dims": torch.tensor(pos_dims, dtype=torch.long),
                "groups": groups,
            }
        return specs
