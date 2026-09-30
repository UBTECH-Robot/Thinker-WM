"""ThinkerWM inference and checkpoint loading: six-mode denoising, normalization/scatter, checkpoint loading."""

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import torch
import PIL.Image

from . import action_repr
from .action_stats import denormalize_batch, normalize_batch

if TYPE_CHECKING:
    from .model import ThinkerWMConfig

logger = logging.getLogger(__name__)

FUNCTION_MODES = ("wam", "wm", "vla", "a2v", "v2a", "uncond")

# Fixed empty-text condition used for unconditional branch (CFG uncond / uncond mode).
UNCONDITIONAL_VLM_MESSAGE = ""


def load_checkpoint(model, path: str, strict: bool = True) -> Dict:
    """Load checkpoint (directory auto-resolves weight file, or direct file path),
    returns additional state beyond model weights from checkpoint."""
    checkpoint_path = Path(path)
    if checkpoint_path.is_dir():
        candidates = [
            checkpoint_path / "pytorch_model_0.bin",
            checkpoint_path / "pytorch_model.bin",
            checkpoint_path / "mp_rank_00_model_states.pt",
            checkpoint_path / "pytorch_model" / "mp_rank_00_model_states.pt",
        ]
        ckpt_file = next((p for p in candidates if p.exists()), None)
        if ckpt_file is None:
            tried = "\n".join(f"  - {p}" for p in candidates)
            raise FileNotFoundError(
                f"No supported checkpoint weight file found. Tried:\n{tried}"
            )
        path = str(ckpt_file)

    try:
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location='cpu')
    if isinstance(checkpoint, dict):
        state_dict = checkpoint.get(
            'module', checkpoint.get('state_dict', checkpoint))
    else:
        raise TypeError(
            f"Unsupported checkpoint payload type: {type(checkpoint)!r}")
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=strict)
    logger.info(f"Checkpoint loaded from {path}: missing={len(missing_keys)}, unexpected={len(unexpected_keys)}")

    return {k: v for k, v in checkpoint.items()
            if k not in ['module', 'state_dict', 'config']}


def _robot_action_columns(
    config: "ThinkerWMConfig",
    robot_names: Optional[List[str]],
) -> Tuple[Optional[List[int]], int, int]:
    """Unified space -> target columns for per-robot original dimension (dim_map column indices or offset slice)."""
    action_dim = int(config.action_dim)
    robot_offset = 0
    cols: Optional[List[int]] = None
    if robot_names:
        name = robot_names[0] if isinstance(robot_names, (list, tuple)) else robot_names
        layout = getattr(config, "robot_layout", None) or {}
        entry = layout.get(str(name))
        if entry is not None:
            dim_map = entry.get("dim_map")
            if dim_map is not None:
                cols = [int(c) for c in dim_map]
            else:
                robot_offset = int(entry["offset"])
        robot_configs = config.robot_configs or {}
        action_dim = int(robot_configs.get(str(name), action_dim))
    return cols, robot_offset, action_dim


def _slice_unified_actions(
    actions: torch.Tensor,
    cols: Optional[List[int]],
    offset: int,
    dim: int,
) -> torch.Tensor:
    """Unified space [B, T, U] -> per-robot original dim [B, T, dim] (dim_map -1 = dropped dim,
    gather column 0 placeholder and set to 0)."""
    if cols is not None:
        idx = torch.tensor(
            [max(int(c), 0) for c in cols], dtype=torch.long, device=actions.device
        )
        out = actions.index_select(-1, idx)
        if any(int(c) < 0 for c in cols):
            dropped = torch.tensor(
                [int(c) < 0 for c in cols], dtype=torch.bool, device=actions.device
            )
            out = out.masked_fill(dropped.view(*([1] * (out.ndim - 1)), -1), 0.0)
        return out
    return actions[..., offset:offset + dim]


def _dropped_local_dims(cols: Optional[List[int]]) -> List[int]:
    """Local dimension indices dropped (-1) in dim_map."""
    if cols is None:
        return []
    return [i for i, c in enumerate(cols) if int(c) < 0]


def _temporal_ema(x: torch.Tensor, alpha: float, dim: int = 1) -> torch.Tensor:
    """Temporal exponential moving average y[t]=alpha*x[t]+(1-alpha)*y[t-1]; dim default 1, alpha>=1 returns as-is."""
    alpha = float(alpha)
    if dim != 1:
        return _temporal_ema(x.transpose(1, dim), alpha, dim=1).transpose(1, dim)
    if alpha >= 1.0 or x.shape[1] <= 1:
        return x
    ys = [x[:, 0]]
    prev = ys[0]
    for t in range(1, x.shape[1]):
        prev = alpha * x[:, t] + (1.0 - alpha) * prev
        ys.append(prev)
    return torch.stack(ys, dim=1)


def _inference_sigmas(model, num_inference_steps: int, shift: Optional[float] = None) -> torch.Tensor:
    """Inference sigma grid: sigma = shift*s / (1 + (shift-1)*s), s uniform grid [1,0]."""
    shift = float(model.config.flow_shift) if shift is None else float(shift)
    s = torch.linspace(1.0, 0.0, num_inference_steps + 1,
                       device=model.device, dtype=torch.float32)
    return shift * s / (1.0 + (shift - 1.0) * s)


def _normalize_state_actions(
    model,
    state: Optional[torch.Tensor],
    actions: Optional[torch.Tensor],
    robot_names: Optional[List[str]],
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Model boundary normalization entry point (dispatched by action_norm_policy)."""
    if model._norm_policy is not None:
        return model._norm_policy.normalize(state, actions, robot_names)
    return normalize_batch(state, actions, robot_names)


def _denormalize_actions(
    model,
    actions: Optional[torch.Tensor],
    robot_names: Optional[List[str]],
) -> Optional[torch.Tensor]:
    """Model boundary denormalization entry point (symmetric with _normalize_state_actions)."""
    if model._norm_policy is not None:
        return model._norm_policy.denormalize(actions, robot_names)
    return denormalize_batch(actions, robot_names)


def _repr_rot_local_dims(model, robot_name: Any) -> set:
    """Local dimensions taken over by rot group for this robot (excluded from output zeroing)."""
    if not model._repr_specs:
        return set()
    spec = model._repr_specs.get(str(robot_name))
    if spec is None:
        return set()
    dims = set()
    for group in spec["groups"]:
        dims.update(int(d) for d in group["local"].tolist())
    return dims


def _gather_native_actions(
    model,
    actions: torch.Tensor,
    state_local: Optional[torch.Tensor],
    robot_names: Optional[List[str]],
    cols: Optional[List[int]],
    offset: int,
    dim: int,
) -> torch.Tensor:
    """Unified space [B, T, U] -> per-robot original dim + relative_eef_rot6 relativization restore
    (eef pos add back state, rot6 -> absolute pose local representation); absolute_native equivalent to pure slice."""
    out = _slice_unified_actions(actions, cols, offset, dim)
    if model._repr_specs is None:
        return out
    B = out.shape[0]
    names = robot_names if robot_names is not None else [None] * B
    if isinstance(names, str):
        names = [names] * B
    for i in range(B):
        spec = model._repr_specs.get(str(names[i])) if names[i] is not None else None
        if spec is None:
            continue
        pos_dims = spec["pos_dims"].to(out.device)
        if pos_dims.numel() and state_local is not None:
            out[i, :, pos_dims] += state_local[i, pos_dims].to(out.dtype)
        for group in spec["groups"]:
            local = group["local"].to(out.device)
            cols_t = group["cols"].to(out.device)
            if int(local.max()) >= out.shape[-1]:
                continue
            rel_6d = actions[i, :, cols_t]
            if not group.get("relative", True):
                # eef_relative=false: delta rotation directly converted to local representation
                out[i, :, local] = action_repr.matrix_to_repr(
                    action_repr.d6_to_matrix(rel_6d),
                    group["repr"], group["quat_order"],
                ).to(out.dtype)
                continue
            if state_local is not None:
                s_rot = state_local[i, local]
            else:
                s_rot = torch.zeros(
                    int(local.numel()), dtype=actions.dtype, device=actions.device)
            abs_rot = action_repr.absolute_rotation_from_6d(
                s_rot, rel_6d, group["repr"], group["quat_order"])
            out[i, :, local] = abs_rot.to(out.dtype)
    return out


def _scatter_to_unified(
    model,
    state: Optional[torch.Tensor],
    actions: Optional[torch.Tensor],
    robot_names: Optional[List[str]],
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], torch.Tensor]:
    """Per-robot action/state scatter to unified dimension [B,..,U] via dim_map, returns
    (state, actions, valid-dim mask); relative_eef_rot6 does EEF relativization first
    (pos subtract state, rotation group to 6D)."""
    U = int(model.config.unified_action_dim)
    B = state.shape[0] if state is not None else actions.shape[0]
    device = (state if state is not None else actions).device
    dtype = (state if state is not None else actions).dtype
    layout = model.config.robot_layout or {}
    robot_configs = model.config.robot_configs or {}
    names = robot_names if robot_names is not None else [None] * B
    if isinstance(names, str):
        names = [names] * B
    actual_dim = (state if state is not None else actions).shape[-1]
    # Per-sample (local dim, unified column) pairs; -1 dropped dims skip scatter/mask.
    pair_indices: List[Tuple[torch.Tensor, torch.Tensor]] = []
    for name in names:
        key = str(name) if name is not None else ""
        entry = layout.get(key)
        if entry is None:
            if name is not None:
                logger.warning(
                    "robot %s not in robot_layout, falling back to offset=0, dim=%d",
                    name, model.config.action_dim,
                )
            dim = int(robot_configs.get(key, model.config.action_dim))
            cols = list(range(0, dim))
        else:
            dim_map = entry.get("dim_map")
            if dim_map is not None:
                cols = [int(c) for c in dim_map]
            else:
                logger.warning(
                    "robot %s has no unified_dim_map registered, falling back to offset=%d front-fill",
                    name, int(entry["offset"]),
                )
                dim = int(robot_configs.get(key, model.config.action_dim))
                start = int(entry["offset"])
                cols = list(range(start, start + dim))
        pairs = [
            (k, c) for k, c in enumerate(cols[:actual_dim]) if 0 <= c < U
        ]
        pair_indices.append(
            (
                torch.tensor([k for k, _ in pairs], dtype=torch.long, device=device),
                torch.tensor([c for _, c in pairs], dtype=torch.long, device=device),
            )
        )
    # relative_eef_rot6: local-space EEF relativization, 6D values overwritten by cols after scatter.
    group_writes = None
    if model._repr_specs is not None:
        group_writes = [[] for _ in range(B)]
        if actions is not None:
            actions = actions.clone()
        for i in range(B):
            spec = (
                model._repr_specs.get(str(names[i]))
                if names[i] is not None else None
            )
            if spec is None:
                continue
            pos_dims = spec["pos_dims"].to(device)
            if (
                pos_dims.numel()
                and state is not None
                and actions is not None
            ):
                actions[i, ..., pos_dims] -= state[i, pos_dims]
            for group in spec["groups"]:
                local = group["local"].to(device)
                if int(local.max()) >= actual_dim:
                    continue
                cols_t = group["cols"].to(device)
                s_rot = state[i, local] if state is not None else None
                s6 = None
                if s_rot is not None:
                    s6 = action_repr.matrix_to_6d(action_repr.repr_to_matrix(
                        s_rot, group["repr"], group["quat_order"]))
                a6 = None
                if actions is not None:
                    a_rot = actions[i, ..., local]
                    if s_rot is not None and group.get("relative", True):
                        _, a6 = action_repr.relative_rotation_6d(
                            s_rot, a_rot, group["repr"], group["quat_order"])
                    else:
                        a6 = action_repr.matrix_to_6d(
                            action_repr.repr_to_matrix(
                                a_rot, group["repr"], group["quat_order"]))
                group_writes[i].append((cols_t, s6, a6))
    mask = torch.zeros(B, U, dtype=torch.bool, device=device)
    for i in range(B):
        _, col_idx = pair_indices[i]
        if col_idx.numel():
            mask[i, col_idx] = True
    if state is not None:
        scattered_state = torch.zeros(B, U, dtype=dtype, device=device)
        for i in range(B):
            local_idx, col_idx = pair_indices[i]
            if col_idx.numel():
                scattered_state[i, col_idx] = state[i, local_idx]
    else:
        scattered_state = None
    if actions is not None:
        scattered_actions = torch.zeros(*actions.shape[:-1], U, dtype=dtype, device=device)
        for i in range(B):
            local_idx, col_idx = pair_indices[i]
            if col_idx.numel():
                scattered_actions[i, ..., col_idx] = actions[i, ..., local_idx]
    else:
        scattered_actions = None
    if group_writes is not None:
        for i in range(B):
            for cols_t, s6, a6 in group_writes[i]:
                mask[i, cols_t] = True
                if s6 is not None and scattered_state is not None:
                    scattered_state[i, cols_t] = s6.to(dtype)
                if a6 is not None and scattered_actions is not None:
                    scattered_actions[i, ..., cols_t] = a6.to(dtype)
    return scattered_state, scattered_actions, mask


def _modality_missing(x: Any) -> bool:
    """Modality missing check: None or all-zero tensor treated as missing; non-tensor always considered present."""
    if x is None:
        return True
    if not torch.is_tensor(x):
        return False
    return bool(x.numel() == 0 or x.abs().amax() == 0)


def inference_step(
    model,
    first_frame: torch.Tensor,
    state: torch.Tensor = None,
    num_inference_steps: int = 50,
    language_embeddings: Optional[List[torch.Tensor]] = None,
    vlm_inputs: Optional[List] = None,
    first_frame_pils: Optional[List[PIL.Image.Image]] = None,
    t5_input_messages: Optional[List[str]] = None,
    vlm_input_messages: Optional[List[str]] = None,
    robot_names: Optional[List[str]] = None,
    guidance_scale: float = 1.0,
    action_guidance_scale: float = 1.0,
    action_num_inference_steps: Optional[int] = None,
    action_smooth_alpha: float = 0.0,
    depth_smooth_alpha: float = 0.0,
    predict_action: Optional[bool] = None,
    predict_video: bool = True,
    depth_first: Optional[torch.Tensor] = None,   # [B, 3, H, W] in [0,1]
    return_depth: bool = False,
    condition_video_frames: Optional[torch.Tensor] = None,  # [B, T, C, H, W] [0,1]
    condition_actions: Optional[torch.Tensor] = None,  # [B, chunk, dim]
    condition_depth_frames: Optional[torch.Tensor] = None,  # [B, T, C, H, W] [0,1]
    uncond_prompt: bool = False,  # True = uncond mode semantics (empty-text constant condition)
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Joint video/action(/depth) denoising inference, returns (predicted_frames,
    predicted_actions[, predicted_depth]); missing modality returns None.
    Missing = input is None/all-zero; batch-internal missing state must be homogeneous."""
    if first_frame is not None and torch.is_tensor(first_frame):
        B = first_frame.shape[0]
    elif state is not None and torch.is_tensor(state):
        B = state.shape[0]
    elif vlm_input_messages is not None and isinstance(vlm_input_messages, (list, tuple)):
        B = len(vlm_input_messages)
    else:
        raise ValueError(
            "inference_step: first_frame/state/vlm_input_messages all missing, "
            "cannot determine batch size"
        )
    video_missing = _modality_missing(first_frame)
    state_missing = _modality_missing(state)
    if predict_action is not None:
        # Explicit modality declaration overrides the zero-based heuristic.
        # predict_action=True also predicts with state=None (no anchor): actions
        # come out as relative EEF (delta pos + relative 6D rotation).
        state_missing = not bool(predict_action)
    if first_frame is not None and torch.is_tensor(first_frame) and not video_missing:
        per_sample = (first_frame.abs().amax(dim=(1, 2, 3)) == 0)
        if bool(per_sample.any()) and not bool(per_sample.all()):
            raise ValueError(
                "inference_step: first_frame missing state is inconsistent within batch"
                " (some samples all zero), please batch by homogeneous modality"
            )
    if state is not None and torch.is_tensor(state) and not state_missing:
        per_sample = (state.abs().amax(dim=1) == 0)
        if bool(per_sample.any()) and not bool(per_sample.all()):
            raise ValueError(
                "inference_step: state missing state is inconsistent within batch"
                " (some samples all zero), please batch by homogeneous modality"
            )
    prompt_missing = (
        vlm_input_messages is None
        or (isinstance(vlm_input_messages, (list, tuple))
            and all(str(m).strip() == "" for m in vlm_input_messages))
    )
    if video_missing and state_missing and prompt_missing:
        raise ValueError(
            "inference_step: all three modalities video/action/prompt missing, no usable input"
        )
    if not predict_video and state_missing:
        raise ValueError(
            "inference_step: predict_video=False (VLA mode) and action modality"
            " missing (state None/all-zero or predict_action=False), no output"
        )

    action_cols, robot_offset, action_dim = _robot_action_columns(
        model.config, robot_names)

    # und is entirely text-based: do not supply placeholder PIL when video is missing
    # (black image would incorrectly route to image encoding path).
    und_pils = first_frame_pils

    with torch.autocast(device_type="cuda", dtype=model.dtype):
        condition_frame_latent = None
        if not video_missing:
            first_frame_norm = (first_frame * 2.0 - 1.0).unsqueeze(2)   # [B, C, 1, H, W]
            with torch.no_grad():
                condition_frame_latent = model.encode_video(first_frame_norm.to(model.dtype))

        # und token computed once outside the denoising loop and cached.
        with torch.no_grad():
            if uncond_prompt:
                if guidance_scale != 1.0:
                    raise ValueError(
                        "inference_step: with uncond_prompt=True the condition is already empty-text "
                        "constant, guidance_scale != 1.0 CFG is meaningless, use guidance_scale=1.0"
                    )
                und_cache = model.world_model.und_branch.encode(
                    [UNCONDITIONAL_VLM_MESSAGE] * B, und_pils)
                und_cache_uncond = None
            elif prompt_missing:
                if guidance_scale != 1.0:
                    raise ValueError(
                        "inference_step: prompt modality missing (None/all-empty) "
                        "guidance_scale != 1.0 CFG is meaningless, use guidance_scale=1.0"
                    )
                und_cache = model.world_model.null_und_cache(B)
                und_cache_uncond = None
            else:
                und_cache = model.world_model.und_branch.encode(
                    vlm_input_messages, und_pils)
                und_cache_uncond = None
                if guidance_scale != 1.0:
                    und_cache_uncond = model.world_model.und_branch.encode(
                        [UNCONDITIONAL_VLM_MESSAGE] * B, und_pils)

        # Depth third stream: enabled only for return_depth or when pinning GT depth; disabled without depth_first.
        tf_depth = (
            condition_depth_frames is not None
            and torch.is_tensor(condition_depth_frames)
        )
        predict_depth = bool(
            (return_depth or tf_depth)
            and getattr(model.config, "enable_depth", False)
            and depth_first is not None
            and torch.is_tensor(depth_first)
            and not video_missing
            and predict_video
        )
        tf_depth = bool(tf_depth and predict_depth)
        depth_first_norm = None
        depth_condition_latent = None
        if predict_depth:
            depth_first_norm = (depth_first * 2.0 - 1.0).unsqueeze(2).to(model.dtype)
            with torch.no_grad():
                depth_condition_latent = model.encode_video(depth_first_norm)

        video_latent = None
        if not video_missing:
            if predict_video:
                B, C_latent, f_latent, H_latent, W_latent = condition_frame_latent.shape
                num_total_latent_frames = 1 + model.config.num_video_frames // 4
                video_latent = torch.randn((B, C_latent, num_total_latent_frames, H_latent, W_latent), device=model.device, dtype=torch.float32).to(model.dtype)
                video_latent[:, :, 0:1] = condition_frame_latent
            else:
                # VLA mode: video condition uses only clean first-frame latent, no denoising or generation.
                video_latent = condition_frame_latent
        depth_latent = None
        if predict_depth:
            # Depth reuses the same noise as video, then independently overwrites first-frame condition.
            depth_latent = video_latent.clone()
            depth_latent[:, :, 0:1] = depth_condition_latent
        # teacher-force: pin the corresponding modality to GT latent (t=0, no advance).
        tf_video = (
            condition_video_frames is not None
            and torch.is_tensor(condition_video_frames)
            and not video_missing
            and predict_video
            and condition_frame_latent is not None
        )
        if tf_video:
            vf = condition_video_frames.to(device=model.device, dtype=model.dtype)
            video_normalized = (vf * 2.0 - 1.0).permute(0, 2, 1, 3, 4)
            full_video = torch.cat([first_frame_norm, video_normalized], dim=2)
            with torch.no_grad():
                video_latent = model.encode_video(full_video.to(model.dtype))
            video_latent[:, :, 0:1] = condition_frame_latent
        if tf_depth:
            df = condition_depth_frames.to(device=model.device, dtype=model.dtype)
            depth_normalized = (df * 2.0 - 1.0).permute(0, 2, 1, 3, 4)
            full_depth = torch.cat([depth_first_norm, depth_normalized], dim=2)
            with torch.no_grad():
                depth_latent = model.encode_video(full_depth.to(model.dtype))
            depth_latent[:, :, 0:1] = depth_condition_latent
        action_latent = None
        if not state_missing:
            action_latent = torch.randn(
                (B, model.config.action_chunk_size, model.config.unified_action_dim),
                device=model.device, dtype=torch.float32).to(model.dtype)

        # state normalization; state_local keeps local-space copy for output-side relativization restore.
        state_local = None
        if not state_missing:
            state = _normalize_state_actions(model, state, None, robot_names)[0]
            state_local = state

        tf_action = (
            condition_actions is not None
            and torch.is_tensor(condition_actions)
            and not state_missing
            and state_local is not None
        )
        if tf_action:
            cond_act = condition_actions.to(device=model.device, dtype=model.dtype)
            cond_act = _normalize_state_actions(model, None, cond_act, robot_names)[1]
            _, cond_act, _ = _scatter_to_unified(model, state_local, cond_act, robot_names)
            action_latent = cond_act

        # Denoising loop: per-modality sigma grids (action can have independent steps),
        # loop advances on the finer grid, coarser modality stays still (dt=0) on some steps.
        sigmas = _inference_sigmas(model, num_inference_steps)
        n_action_steps = num_inference_steps
        if action_num_inference_steps is not None and int(action_num_inference_steps) >= 1:
            n_action_steps = int(action_num_inference_steps)
        sigmas_action = (
            _inference_sigmas(model, n_action_steps, shift=model._action_flow_shift)
            if not state_missing else None
        )
        n_loop = max(num_inference_steps, n_action_steps)
        for i in range(n_loop):
            i_v = min(i * num_inference_steps // n_loop, num_inference_steps - 1)
            i_a = min(i * n_action_steps // n_loop, n_action_steps - 1)
            sigma = sigmas[i_v]
            sigma_action = sigmas_action[i_a] if sigmas_action is not None else sigma
            adv_video = (i + 1) * num_inference_steps // n_loop > i_v
            adv_action = (i + 1) * n_action_steps // n_loop > i_a
            dt_video = float(sigmas[i_v + 1] - sigma) if (adv_video and predict_video) else 0.0
            dt_action = (
                float(sigmas_action[i_a + 1] - sigma_action)
                if (adv_action and sigmas_action is not None) else 0.0
            )
            t_video = (sigma * 1000).expand(B).to(model.dtype).unsqueeze(1)
            t_action = (sigma_action * 1000).expand(B).to(model.dtype).unsqueeze(1)
            # teacher-force modalities get t set to 0; depth not pinned must stay at current sigma.
            t_depth = t_video.clone()
            if tf_video:
                t_video = torch.zeros_like(t_video)
            if tf_action:
                t_action = torch.zeros_like(t_action)
            if tf_depth:
                t_depth = torch.zeros_like(t_depth)

            video_velocity, action_velocity, depth_velocity = model.world_model(
                t_video, vlm_input_messages, und_pils,
                video_latent, action_latent,
                t_video=t_video, t_action=t_action,
                robot_names=robot_names,
                und_cache=und_cache,
                depth=depth_latent, t_depth=t_depth,
                depth_detach_video_kv=bool(getattr(model.config, "depth_detach_video_kv", False)),
            )
            if und_cache_uncond is not None:
                # CFG: v = v_uncond + scale*(v_cond - v_uncond); action defaults to no CFG.
                video_velocity_uncond, action_velocity_uncond, _ = model.world_model(
                    t_video, [UNCONDITIONAL_VLM_MESSAGE] * B, und_pils,
                    video_latent, action_latent,
                    t_video=t_video, t_action=t_action,
                    robot_names=robot_names,
                    und_cache=und_cache_uncond,
                )
                if video_velocity is not None and video_velocity_uncond is not None:
                    video_velocity = video_velocity_uncond + guidance_scale * (
                        video_velocity - video_velocity_uncond)
                if (
                    float(action_guidance_scale) != 1.0
                    and action_velocity is not None and action_velocity_uncond is not None
                ):
                    action_velocity = action_velocity_uncond + float(action_guidance_scale) * (
                        action_velocity - action_velocity_uncond)

            # Euler integration (skip for missing modality / non-advancing steps / teacher-force)
            if (not tf_video) and video_velocity is not None and dt_video != 0.0:
                video_latent = video_latent + video_velocity * dt_video
            if (not tf_action) and action_velocity is not None and dt_action != 0.0:
                action_latent = action_latent + action_velocity * dt_action
            if (
                (not tf_depth)
                and depth_latent is not None
                and depth_velocity is not None
                and dt_video != 0.0
            ):
                depth_latent = depth_latent + depth_velocity * dt_video

            if video_latent is not None:
                video_latent[:, :, 0:1] = condition_frame_latent
            if depth_latent is not None:
                depth_latent[:, :, 0:1] = depth_condition_latent

        predicted_frames = None
        if video_latent is not None and predict_video:
            with torch.no_grad():
                decoded_frames = model.decode_video(video_latent)
                predicted_frames = decoded_frames[:, :, 1:]  # remove conditioning first frame
                predicted_frames = (predicted_frames + 1.0) / 2.0  # [-1,1] to [0,1]
                predicted_frames = torch.clamp(predicted_frames, 0, 1).float()

        predicted_actions = None
        if action_latent is not None:
            if float(action_smooth_alpha) > 0.0:
                action_latent = _temporal_ema(action_latent, float(action_smooth_alpha))
            predicted_actions = action_latent.float()  # [B, chunk, U]
            predicted_actions = _gather_native_actions(
                model, predicted_actions, state_local, robot_names,
                action_cols, robot_offset, action_dim)
            predicted_actions = _denormalize_actions(model, predicted_actions, robot_names)
            # Reconstruct dropped dims as raw 0; rot-group-controlled dims already have real values, exclude.
            dropped_dims = _dropped_local_dims(action_cols)
            if model._repr_specs is not None and robot_names:
                rot_dims = _repr_rot_local_dims(model, robot_names[0])
                if rot_dims:
                    dropped_dims = [d for d in dropped_dims if d not in rot_dims]
            if dropped_dims:
                predicted_actions[..., dropped_dims] = 0.0

        predicted_depth = None
        if depth_latent is not None:
            with torch.no_grad():
                decoded_depth = model.decode_video(depth_latent)
                # Normalize first then EMA (includes first-frame condition, pins y[0]), then remove condition frame.
                decoded_depth = (decoded_depth + 1.0) / 2.0
                decoded_depth = torch.clamp(decoded_depth, 0, 1).float()
                if float(depth_smooth_alpha) > 0.0:
                    decoded_depth = _temporal_ema(
                        decoded_depth, float(depth_smooth_alpha), dim=2,
                    )
                predicted_depth = decoded_depth[:, :, 1:]

    if return_depth:
        return predicted_frames, predicted_actions, predicted_depth
    return predicted_frames, predicted_actions


def inference_action_step(
    model,
    first_frame: torch.Tensor,
    state: torch.Tensor = None,
    conditioning_actions: torch.Tensor = None,
    num_inference_steps: int = 50,
    language_embeddings: Optional[List[torch.Tensor]] = None,
    vlm_inputs: Optional[List] = None,
    first_frame_pils: Optional[List[PIL.Image.Image]] = None,
    t5_input_messages: Optional[List[str]] = None,
    vlm_input_messages: Optional[List[str]] = None,
    robot_names: Optional[List[str]] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """a2v inference: conditioning actions (t_action=0 pinned) + first frame -> video, also outputs action echo."""
    B = first_frame.shape[0]

    assert language_embeddings is None
    assert vlm_inputs is None
    action_cols, robot_offset, action_dim = _robot_action_columns(
        model.config, robot_names)
    if conditioning_actions is not None:
        action_dim = int(conditioning_actions.shape[-1])

    with torch.autocast(device_type="cuda", dtype=model.dtype):
        first_frame_norm = (first_frame * 2.0 - 1.0).unsqueeze(2)   # [B, C, 1, H, W]
        with torch.no_grad():
            condition_frame_latent = model.encode_video(first_frame_norm.to(model.dtype))

        with torch.no_grad():
            und_cache = model.world_model.und_branch.encode(
                vlm_input_messages, first_frame_pils)

        B, C_latent, f_latent, H_latent, W_latent = condition_frame_latent.shape
        num_total_latent_frames = 1 + model.config.num_video_frames // 4
        video_latent = torch.randn((B, C_latent, num_total_latent_frames, H_latent, W_latent), device=model.device, dtype=torch.float32).to(model.dtype)
        video_latent[:, :, 0:1] = condition_frame_latent
        action_latent = torch.randn(
            (B, model.config.action_chunk_size, model.config.unified_action_dim),
            device=model.device, dtype=torch.float32).to(model.dtype)

        state = _normalize_state_actions(model, state, None, robot_names)[0]
        state_local = state
        if conditioning_actions is not None:
            conditioning_actions = _normalize_state_actions(
                model, None, conditioning_actions, robot_names
            )[1]
            _, conditioning_actions, _ = _scatter_to_unified(
                model, state_local, conditioning_actions, robot_names)

        sigmas = _inference_sigmas(model, num_inference_steps)
        for i in range(num_inference_steps):
            sigma = sigmas[i]
            sigma_next = sigmas[i + 1]
            dt = sigma_next - sigma
            t_scaled = (sigma * 1000).expand(B).to(model.dtype).unsqueeze(1)

            t_action = torch.zeros_like(t_scaled)  # conditioning actions t=0
            if conditioning_actions is not None:
                input_actions = conditioning_actions
            else:
                input_actions = action_latent

            video_velocity, action_velocity, _ = model.world_model(
                t_scaled, vlm_input_messages, first_frame_pils,
                video_latent, input_actions,
                t_video=t_scaled, t_action=t_action,
                robot_names=robot_names,
                und_cache=und_cache,
            )

            video_latent = video_latent + video_velocity * dt
            action_latent = action_latent + action_velocity * dt

            video_latent[:, :, 0:1] = condition_frame_latent

        with torch.no_grad():
            decoded_frames = model.decode_video(video_latent)
            predicted_frames = decoded_frames[:, :, 1:]  # remove conditioning first frame
            predicted_frames = (predicted_frames + 1.0) / 2.0  # [-1,1] to [0,1]
            predicted_frames = torch.clamp(predicted_frames, 0, 1).float()

        predicted_actions = (
            conditioning_actions.float() if conditioning_actions is not None else action_latent.float()
        )
        predicted_actions = _gather_native_actions(
            model, predicted_actions, state_local, robot_names,
            action_cols, robot_offset, action_dim)
        predicted_actions = _denormalize_actions(model, predicted_actions, robot_names)
        dropped_dims = _dropped_local_dims(action_cols)
        if model._repr_specs is not None and robot_names:
            rot_dims = _repr_rot_local_dims(model, robot_names[0])
            if rot_dims:
                dropped_dims = [d for d in dropped_dims if d not in rot_dims]
        if dropped_dims:
            predicted_actions[..., dropped_dims] = 0.0

        return predicted_frames, predicted_actions


def inference_v2a_step(
    model,
    video_frames: torch.Tensor,
    state: torch.Tensor = None,
    num_inference_steps: int = 50,
    first_frame_pils: Optional[List[PIL.Image.Image]] = None,
    vlm_input_messages: Optional[List[str]] = None,
    robot_names: Optional[List[str]] = None,
) -> torch.Tensor:
    """v2a inference: full GT video sequence as condition (t=0), denoise action trajectory from noise."""
    B = video_frames.shape[0]

    action_cols, robot_offset, action_dim = _robot_action_columns(
        model.config, robot_names)

    with torch.autocast(device_type="cuda", dtype=model.dtype):
        video_norm = video_frames * 2.0 - 1.0   # [0,1] -> [-1,1]
        with torch.no_grad():
            video_condition_latent = model.encode_video(
                video_norm.to(model.dtype))      # [B, C', 1+F/4, H', W']

        # prompt is always dropped (all-empty text -> uncond constant).
        with torch.no_grad():
            und_cache = model.world_model.und_branch.encode(
                vlm_input_messages, first_frame_pils)

        action_latent = torch.randn(
            (B, model.config.action_chunk_size, model.config.unified_action_dim),
            device=model.device, dtype=torch.float32).to(model.dtype)

        # state only serves as output-side relativization anchor (no state token, not fed into network).
        state_local = None
        if state is not None and torch.is_tensor(state):
            state_local = _normalize_state_actions(
                model, state, None, robot_names)[0]

        sigmas_action = _inference_sigmas(
            model, num_inference_steps, shift=model._action_flow_shift)
        t_video = torch.zeros((B, 1), device=model.device, dtype=model.dtype)
        for i in range(num_inference_steps):
            sigma = sigmas_action[i]
            dt = float(sigmas_action[i + 1] - sigma)
            t_action = (sigma * 1000).expand(B).to(model.dtype).unsqueeze(1)

            _video_velocity, action_velocity, _ = model.world_model(
                t_video, vlm_input_messages, first_frame_pils,
                video_condition_latent, action_latent,
                t_video=t_video, t_action=t_action,
                robot_names=robot_names,
                und_cache=und_cache,
            )

            action_latent = action_latent + action_velocity * dt

        predicted_actions = action_latent.float()
        predicted_actions = _gather_native_actions(
            model, predicted_actions, state_local, robot_names,
            action_cols, robot_offset, action_dim)
        predicted_actions = _denormalize_actions(model, predicted_actions, robot_names)
        dropped_dims = _dropped_local_dims(action_cols)
        if model._repr_specs is not None and robot_names:
            rot_dims = _repr_rot_local_dims(model, robot_names[0])
            if rot_dims:
                dropped_dims = [d for d in dropped_dims if d not in rot_dims]
        if dropped_dims:
            predicted_actions[..., dropped_dims] = 0.0

        return predicted_actions
