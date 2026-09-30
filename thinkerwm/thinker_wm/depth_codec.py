"""Depth codec: millimeter depth <-> model-space 3-channel grayscale (near=1 bright / far=0 dark, analytically invertible)."""
from __future__ import annotations

import math
from typing import Literal

import torch

Mode = Literal["inverse", "linear", "log"]
InvalidTo = Literal["far", "near"]
OutRange = Literal["pm1", "01"]


def _to_tensor(depth) -> torch.Tensor:
    if not torch.is_tensor(depth):
        depth = torch.as_tensor(depth)
    return depth


def encode_depth_mm(
    depth_mm,
    z_near: float = 0.1,
    z_far: float = 10.0,
    mode: Mode = "log",
    invalid_to: InvalidTo = "far",
    out_range: OutRange = "pm1",
) -> torch.Tensor:
    """Millimeter depth -> 3-channel model tensor [..., 3, H, W] (0 = invalid/pad)."""
    depth = _to_tensor(depth_mm).float()
    z = depth / 1000.0
    valid = depth > 0

    if mode == "inverse":
        inv_near = 1.0 / z_near
        inv_far = 1.0 / z_far
        z_c = z.clamp(z_near, z_far)
        v = (1.0 / z_c - inv_far) / (inv_near - inv_far)
    elif mode == "log":
        log_near, log_far = math.log(z_near), math.log(z_far)
        z_c = z.clamp(z_near, z_far)
        v = 1.0 - (torch.log(z_c) - log_near) / (log_far - log_near)
    elif mode == "linear":
        v = 1.0 - z.clamp(0.0, z_far) / z_far
    else:
        raise ValueError(f"unknown mode: {mode!r}")

    invalid_v = 0.0 if invalid_to == "far" else 1.0
    v = torch.where(valid, v, torch.full_like(v, invalid_v))
    v = v.clamp(0.0, 1.0)

    x = v if out_range == "01" else v * 2.0 - 1.0
    x = x.unsqueeze(-3).expand(*x.shape[:-2], 3, *x.shape[-2:])
    return x.contiguous()


def decode_to_depth_m(
    x3,
    z_near: float = 0.1,
    z_far: float = 10.0,
    mode: Mode = "log",
    invalid_eps: float = 0.0,
    in_range: OutRange = "pm1",
) -> tuple[torch.Tensor, torch.Tensor]:
    """3-channel model tensor -> metric depth, returns (depth_m, valid_mask);
    v <= invalid_eps is invalid (indistinguishable from z_far endpoint encoding)."""
    x = _to_tensor(x3).float()
    if x.shape[-3] in (1, 3) and x.dim() >= 3:
        # During encoding all 3 channels are identical; generated depth 3-channels differ slightly, take mean.
        x = x.mean(dim=-3)

    v = (x if in_range == "01" else (x + 1.0) / 2.0).clamp(0.0, 1.0)
    valid = v > invalid_eps

    if mode == "inverse":
        inv_near = 1.0 / z_near
        inv_far = 1.0 / z_far
        inv = v * (inv_near - inv_far) + inv_far
        z = 1.0 / inv.clamp_min(1e-8)
    elif mode == "log":
        log_near, log_far = math.log(z_near), math.log(z_far)
        z = torch.exp((1.0 - v) * (log_far - log_near) + log_near)
    elif mode == "linear":
        z = (1.0 - v) * z_far
    else:
        raise ValueError(f"unknown mode: {mode!r}")

    return z, valid
