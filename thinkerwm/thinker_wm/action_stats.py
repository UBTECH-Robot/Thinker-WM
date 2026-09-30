"""Per-robot action/state normalization statistics (JSON asset loading) and normalization strategies."""

from __future__ import annotations

import json
import logging
from typing import Dict, List, Optional, Sequence, Tuple

import torch

logger = logging.getLogger(__name__)

# Canonical robot name -> per-dimension stats (load_action_stats fills from JSON asset;
# empty registry = identity normalization for all robots).
_ACTION_STATS: Dict[str, Dict[str, list]] = {}


def load_action_stats(path: str) -> int:
    """Load per-robot normalization statistics from JSON asset into module-level registry, returns entry count."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"action stats JSON top level must be an object: {path}")
    for name, entry in data.items():
        if not isinstance(entry, dict):
            raise ValueError(f"action stats[{name!r}] must be an object: {path}")
        for key, vec in entry.items():
            if not isinstance(vec, list) or not all(
                    v is None or isinstance(v, (int, float)) for v in vec):
                raise ValueError(
                    f"action stats[{name!r}][{key!r}] must be a list of number|null: {path}")
    _ACTION_STATS.update(data)
    logger.info("Loaded action stats for %d robots from %s", len(data), path)
    return len(data)


def get_stats(robot_name: str) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    """Return (mean, std) tensors for the given robot; None if unregistered."""
    stats = _ACTION_STATS.get(str(robot_name))
    if not stats:
        return None
    mean = stats.get("mean")
    std = stats.get("std")
    if not mean or not std or len(mean) != len(std):
        return None
    return torch.tensor(mean, dtype=torch.float32), torch.tensor(
        std, dtype=torch.float32
    )


def _stats_like(
    stats: Tuple[torch.Tensor, torch.Tensor], ref: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    mean, std = stats
    return (
        mean.to(device=ref.device, dtype=ref.dtype),
        std.to(device=ref.device, dtype=ref.dtype),
    )


def normalize(value: torch.Tensor, robot_name: str) -> torch.Tensor:
    """Single-robot normalization: (x - mean) / std (last dim is robot dimension)."""
    stats = get_stats(robot_name)
    if stats is None:
        return value
    mean, std = _stats_like(stats, value)
    return (value - mean) * (1.0 / std)


def denormalize(value: torch.Tensor, robot_name: str) -> torch.Tensor:
    """Single-robot denormalization: x * std + mean."""
    stats = get_stats(robot_name)
    if stats is None:
        return value
    mean, std = _stats_like(stats, value)
    return value * std + mean


def _names_list(
    robot_names: Optional[Sequence[str]], batch: int
) -> Optional[list[str]]:
    if robot_names is None:
        return None
    if isinstance(robot_names, str):
        return [robot_names] * batch
    names = [str(name) for name in robot_names]
    if len(names) != batch:
        raise ValueError(
            f"robot_names length {len(names)} != batch {batch}"
        )
    return names


def _apply_per_name(
    value: torch.Tensor,
    names: Optional[list[str]],
    fn,
) -> torch.Tensor:
    """Apply fn per-sample robot_name within batch (value: [B, ..., D])."""
    if value is None:
        return None
    if names is None:
        return value
    out = value.clone()
    for name in set(names):
        stats = get_stats(name)
        if stats is None:
            continue
        mask = [n == name for n in names]
        out[mask] = fn(value[mask], name)
    return out


def normalize_batch(
    state: Optional[torch.Tensor],
    actions: Optional[torch.Tensor],
    robot_names: Optional[Sequence[str]],
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Batch normalize state/actions (per-sample robot_name).

    state: [B, D] or None; actions: [B, T, D] or None.
    robot_names: None (identity) / str (same robot for whole batch) / list[str] (per-sample).
    """
    if robot_names is None:
        return state, actions
    batch = None
    if state is not None:
        batch = int(state.shape[0])
    elif actions is not None:
        batch = int(actions.shape[0])
    if batch is None:
        return state, actions
    names = _names_list(robot_names, batch)
    if names is None:
        return state, actions
    state = _apply_per_name(state, names, normalize)
    actions = _apply_per_name(actions, names, normalize)
    return state, actions


def denormalize_batch(
    actions: torch.Tensor,
    robot_names: Optional[Sequence[str]],
) -> torch.Tensor:
    """Batch denormalize actions (per-sample robot_name); unregistered/None pass through."""
    if robot_names is None or actions is None:
        return actions
    names = _names_list(robot_names, int(actions.shape[0]))
    if names is None:
        return actions
    return _apply_per_name(actions, names, denormalize)


# gripper_hand_minmax strategy: only gripper/dexterous-hand dimensions normalize to [-1,1]
# (min-max boundary missing falls back to physical-space z-score, then identity),
# other dimensions keep physical units; input sources that are already in z-score space
# (entries with reader_z_*) are inverse-transformed back to physical units first.


def _norm01_local_dims(
    dim_map: Optional[Sequence[int]],
    bands: Optional[Dict[str, Tuple[int, int]]],
) -> List[int]:
    """Local dimension indices (sorted) that fall into gripper/hand-type bands in dim_map."""
    if not dim_map or not bands:
        return []
    dims = []
    for k, c in enumerate(dim_map):
        c = int(c)
        if c < 0:
            continue
        for band_name, (start, end) in bands.items():
            if start <= c < end and (
                band_name.endswith("_gripper") or band_name.startswith("hand_")
            ):
                dims.append(k)
                break
    return dims


class GripperHandNormPolicy:
    """gripper_hand_minmax normalization policy (applied per-sample robot_name,
    same calling convention as normalize_batch)."""

    def __init__(self, specs: Dict[str, Dict[str, Optional[torch.Tensor]]]):
        self._specs = specs

    def normalize(
        self,
        state: Optional[torch.Tensor],
        actions: Optional[torch.Tensor],
        robot_names: Optional[Sequence[str]],
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        return self._apply(state, actions, robot_names, inverse=False)

    def denormalize(
        self,
        actions: Optional[torch.Tensor],
        robot_names: Optional[Sequence[str]],
    ) -> Optional[torch.Tensor]:
        _, actions = self._apply(None, actions, robot_names, inverse=True)
        return actions

    # ---- internal ----

    def _apply(self, state, actions, robot_names, inverse: bool):
        if robot_names is None:
            return state, actions
        batch = None
        if state is not None:
            batch = int(state.shape[0])
        elif actions is not None:
            batch = int(actions.shape[0])
        if batch is None:
            return state, actions
        names = _names_list(robot_names, batch)
        if names is None:
            return state, actions
        for name in set(names):
            spec = self._specs.get(str(name))
            if spec is None:
                continue
            mask = [n == name for n in names]
            if state is not None:
                state = state.clone()
                state[mask] = self._map_value(state[mask], spec, inverse)
            if actions is not None:
                actions = actions.clone()
                actions[mask] = self._map_value(actions[mask], spec, inverse)
        return state, actions

    @staticmethod
    def _to_ref(spec_tensor: Optional[torch.Tensor], ref: torch.Tensor):
        if spec_tensor is None:
            return None
        return spec_tensor.to(device=ref.device, dtype=ref.dtype)

    def _map_value(self, value, spec, inverse: bool):
        x = value
        dims = spec["norm01_dims"]
        rz_mean = self._to_ref(spec["reader_z_mean"], x)
        rz_std = self._to_ref(spec["reader_z_std"], x)
        if inverse:
            x = self._map_norm01(x, spec, dims, inverse=True)
            if rz_mean is not None:
                x = (x - rz_mean) * (1.0 / rz_std)
        else:
            if rz_mean is not None:
                x = x * rz_std + rz_mean
            x = self._map_norm01(x, spec, dims, inverse=False)
        return x

    def _map_norm01(self, x, spec, dims, inverse: bool):
        if dims is None or not dims.numel():
            return x
        dims = dims.to(device=x.device)
        dims = dims[dims < x.shape[-1]]
        if not dims.numel():
            return x
        q_low = self._to_ref(spec["q_low"], x)[dims]
        q_high = self._to_ref(spec["q_high"], x)[dims]
        p_mean = self._to_ref(spec["phys_mean"], x)[dims]
        p_std = self._to_ref(spec["phys_std"], x)[dims]
        use_q = (
            torch.isfinite(q_low)
            & torch.isfinite(q_high)
            & ((q_high - q_low).abs() > 1e-12)
        )
        use_z = (
            ~use_q
            & torch.isfinite(p_mean)
            & torch.isfinite(p_std)
            & (p_std.abs() > 1e-12)
        )
        x = x.clone()
        cur = x[..., dims]
        if not inverse:
            if use_q.any():
                cur[..., use_q] = (
                    2.0 * (cur[..., use_q] - q_low[use_q]) / (q_high[use_q] - q_low[use_q]) - 1.0
                )
            if use_z.any():
                cur[..., use_z] = (cur[..., use_z] - p_mean[use_z]) * (1.0 / p_std[use_z])
        else:
            if use_q.any():
                cur[..., use_q] = (
                    (cur[..., use_q] + 1.0) / 2.0 * (q_high[use_q] - q_low[use_q]) + q_low[use_q]
                )
            if use_z.any():
                cur[..., use_z] = cur[..., use_z] * p_std[use_z] + p_mean[use_z]
        x[..., dims] = cur
        return x


def build_gripper_hand_policy(
    robot_layout: Optional[Dict[str, Dict]],
    unified_bands: Optional[Dict[str, Sequence]],
) -> GripperHandNormPolicy:
    """Build per-robot gripper/hand dimension normalization spec from robot_layout + unified_bands."""
    bands = None
    if unified_bands:
        bands = {
            str(band): (int(span[0]), int(span[1]))
            for band, span in unified_bands.items()
        }
    specs: Dict[str, Dict[str, Optional[torch.Tensor]]] = {}
    for name, entry in (robot_layout or {}).items():
        dim_map = entry.get("dim_map")
        dims = _norm01_local_dims(dim_map, bands)
        stats = _ACTION_STATS.get(str(name)) or {}
        dim = len(stats.get("mean") or []) or (len(dim_map) if dim_map else 0)

        def _vec(key: str) -> torch.Tensor:
            raw = stats.get(key)
            if not raw or len(raw) != dim:
                return torch.full((dim,), float("nan"), dtype=torch.float32)
            return torch.tensor(
                [float(v) if v is not None else float("nan") for v in raw],
                dtype=torch.float32,
            )

        def _z_vec(key: str) -> Optional[torch.Tensor]:
            raw = stats.get(key)
            if not raw or len(raw) != dim:
                return None
            return torch.tensor([float(v) for v in raw], dtype=torch.float32)

        q_low, q_high = _vec("q_low"), _vec("q_high")
        phys_mean, phys_std = _vec("phys_mean"), _vec("phys_std")
        if dims:
            d = torch.tensor(dims, dtype=torch.long)
            has_q = (
                torch.isfinite(q_low[d]) & torch.isfinite(q_high[d])
            )
            has_z = torch.isfinite(phys_mean[d]) & torch.isfinite(phys_std[d])
            if not bool((has_q | has_z).all()):
                logger.warning(
                    "gripper_hand_minmax: robot %s gripper/hand dimensions %s have "
                    "neither q_low/q_high nor phys_mean/phys_std in action stats "
                    "registry; these dimensions will remain as physical unit identity "
                    "(same for unregistered robots)",
                    name, dims,
                )
        specs[str(name)] = {
            "norm01_dims": torch.tensor(dims, dtype=torch.long),
            "q_low": q_low,
            "q_high": q_high,
            "phys_mean": phys_mean,
            "phys_std": phys_std,
            "reader_z_mean": _z_vec("reader_z_mean"),
            "reader_z_std": _z_vec("reader_z_std"),
        }
    return GripperHandNormPolicy(specs)
