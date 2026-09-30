"""Rotation representation conversion utilities: rotvec/quat/rot3 <-> rotation matrix <-> 6D, with EEF relativization."""

from __future__ import annotations

from typing import Tuple

import torch

REPR_WIDTHS = {"rotvec": 3, "quat": 4, "rot3": 3, "rot6": 6}

_EPS = 1e-8


def repr_width(repr_name: str) -> int:
    if repr_name not in REPR_WIDTHS:
        raise ValueError(
            f"Unknown rotation representation {repr_name!r} (registered: {sorted(REPR_WIDTHS)})"
        )
    return REPR_WIDTHS[repr_name]


def _eye(ref: torch.Tensor) -> torch.Tensor:
    return torch.eye(3, dtype=ref.dtype, device=ref.device).expand(
        *ref.shape[:-2], 3, 3
    )


def rotvec_to_matrix(rotvec: torch.Tensor) -> torch.Tensor:
    """rotvec [..., 3] -> rotation matrix (Rodrigues, stable for small angles)."""
    dtype_in = rotvec.dtype
    rv = rotvec.float()
    theta = rv.norm(dim=-1, keepdim=True)  # [..., 1]
    small = theta < 1e-4
    theta_safe = theta.clamp(min=1e-8)
    a = torch.where(small, 1.0 - theta**2 / 6.0, torch.sin(theta_safe) / theta_safe)
    b = torch.where(
        small,
        0.5 - theta**2 / 24.0,
        (1.0 - torch.cos(theta_safe)) / theta_safe**2,
    )
    x, y, z = rv.unbind(-1)
    zero = torch.zeros_like(x)
    k = torch.stack(
        [zero, -z, y, z, zero, -x, -y, x, zero], dim=-1
    ).reshape(*rv.shape[:-1], 3, 3)
    k2 = k @ k
    eye = torch.eye(3, dtype=k.dtype, device=k.device).expand_as(k)
    r = eye + a.unsqueeze(-1) * k + b.unsqueeze(-1) * k2
    return r.to(dtype_in)


def matrix_to_rotvec(matrix: torch.Tensor) -> torch.Tensor:
    """Rotation matrix -> rotvec (via quaternion, well-behaved at theta->pi)."""
    dtype_in = matrix.dtype
    q = matrix_to_quat(matrix, "wxyz").float()
    w, xyz = q[..., :1], q[..., 1:4]
    s = (1.0 - w * w).clamp(min=0.0).sqrt()  # sin(theta/2) >= 0 (w normalized to >=0 hemisphere)
    theta = 2.0 * torch.arccos(w.clamp(-1.0, 1.0))
    small = s < 1e-4
    rv = torch.where(small, 2.0 * xyz, theta * xyz / s.clamp(min=1e-8))
    return rv.to(dtype_in)


def quat_to_matrix(quat: torch.Tensor, order: str = "xyzw") -> torch.Tensor:
    """quat -> rotation matrix (zero-norm falls back to identity)."""
    dtype_in = quat.dtype
    q = quat.float()
    if order == "xyzw":
        x, y, z, w = q.unbind(-1)
    elif order == "wxyz":
        w, x, y, z = q.unbind(-1)
    else:
        raise ValueError(f"Unknown quat component order {order!r} (supported: xyzw/wxyz)")
    norm = (x**2 + y**2 + z**2 + w**2).sqrt().clamp(min=1e-8)
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    two = 2.0
    r = torch.stack(
        [
            1 - two * (y * y + z * z), two * (x * y - z * w), two * (x * z + y * w),
            two * (x * y + z * w), 1 - two * (x * x + z * z), two * (y * z - x * w),
            two * (x * z - y * w), two * (y * z + x * w), 1 - two * (x * x + y * y),
        ],
        dim=-1,
    ).reshape(*q.shape[:-1], 3, 3)
    zero_input = (quat.float().norm(dim=-1) < 1e-8)[..., None, None]
    eye = torch.eye(3, dtype=r.dtype, device=r.device).expand_as(r)
    r = torch.where(zero_input, eye, r)
    return r.to(dtype_in)


def matrix_to_quat(matrix: torch.Tensor, order: str = "xyzw") -> torch.Tensor:
    """Rotation matrix -> quat (Shepperd method, normalized to w >= 0 hemisphere to disambiguate q/-q)."""
    dtype_in = matrix.dtype
    r = matrix.float()
    m00, m11, m22 = r[..., 0, 0], r[..., 1, 1], r[..., 2, 2]
    w = torch.sqrt((1.0 + m00 + m11 + m22).clamp(min=0.0)) / 2.0
    x = torch.sqrt((1.0 + m00 - m11 - m22).clamp(min=0.0)) / 2.0
    y = torch.sqrt((1.0 - m00 + m11 - m22).clamp(min=0.0)) / 2.0
    z = torch.sqrt((1.0 - m00 - m11 + m22).clamp(min=0.0)) / 2.0
    x = torch.copysign(x, r[..., 2, 1] - r[..., 1, 2])
    y = torch.copysign(y, r[..., 0, 2] - r[..., 2, 0])
    z = torch.copysign(z, r[..., 1, 0] - r[..., 0, 1])
    q = torch.stack([x, y, z, w], dim=-1)
    q = q / q.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    neg = q[..., 3:4] < 0.0
    q = torch.where(neg.expand_as(q), -q, q)
    if order == "xyzw":
        out = q
    elif order == "wxyz":
        out = torch.cat([q[..., 3:4], q[..., :3]], dim=-1)
    else:
        raise ValueError(f"Unknown quat component order {order!r} (supported: xyzw/wxyz)")
    return out.to(dtype_in)


def matrix_to_6d(matrix: torch.Tensor) -> torch.Tensor:
    """Rotation matrix [..., 3, 3] -> 6D [..., 6] (concatenate first two columns: x col ++ y col)."""
    return torch.cat([matrix[..., :, 0], matrix[..., :, 1]], dim=-1)


def d6_to_matrix(d6: torch.Tensor) -> torch.Tensor:
    """6D -> rotation matrix (Gram-Schmidt orthogonalization; degenerate input falls back to identity)."""
    dtype_in = d6.dtype
    x = d6.float()
    a1, a2 = x[..., :3], x[..., 3:6]
    n1 = a1.norm(dim=-1, keepdim=True)
    b1 = a1 / n1.clamp(min=1e-8)
    e2 = a2 - (b1 * a2).sum(dim=-1, keepdim=True) * b1
    n2 = e2.norm(dim=-1, keepdim=True)
    b2 = e2 / n2.clamp(min=1e-8)
    b3 = torch.cross(b1, b2, dim=-1)
    r = torch.stack([b1, b2, b3], dim=-1)  # columns as basis
    degenerate = ((n1 < 1e-6) | (n2 < 1e-6))[..., None]
    eye = torch.eye(3, dtype=r.dtype, device=r.device).expand_as(r)
    r = torch.where(degenerate, eye, r)
    return r.to(dtype_in)


def repr_to_matrix(x: torch.Tensor, repr_name: str, quat_order: str = "xyzw") -> torch.Tensor:
    """Any registered representation [..., W] -> rotation matrix [..., 3, 3]."""
    if repr_name in ("rotvec", "rot3"):
        return rotvec_to_matrix(x)
    if repr_name == "quat":
        return quat_to_matrix(x, order=quat_order)
    if repr_name == "rot6":
        return d6_to_matrix(x)
    raise ValueError(f"Unknown rotation representation {repr_name!r}")


def matrix_to_repr(matrix: torch.Tensor, repr_name: str, quat_order: str = "xyzw") -> torch.Tensor:
    """Rotation matrix [..., 3, 3] -> any registered representation [..., W]."""
    if repr_name in ("rotvec", "rot3"):
        return matrix_to_rotvec(matrix)
    if repr_name == "quat":
        return matrix_to_quat(matrix, order=quat_order)
    if repr_name == "rot6":
        return matrix_to_6d(matrix)
    raise ValueError(f"Unknown rotation representation {repr_name!r}")


def relative_rotation_6d(
    state_rot: torch.Tensor,
    action_rot: torch.Tensor,
    repr_name: str,
    quat_order: str = "xyzw",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """B1/B2 target transform: state absolute 6D + action relative 6D (R_rel = R_s^T * R_i)."""
    r_s = repr_to_matrix(state_rot, repr_name, quat_order)  # [..., 3, 3]
    r_i = repr_to_matrix(action_rot, repr_name, quat_order)  # [..., T, 3, 3]
    r_rel = r_s.transpose(-1, -2).unsqueeze(-3) @ r_i
    return matrix_to_6d(r_s), matrix_to_6d(r_rel)


def absolute_rotation_from_6d(
    state_rot: torch.Tensor,
    rel_6d: torch.Tensor,
    repr_name: str,
    quat_order: str = "xyzw",
) -> torch.Tensor:
    """B1/B2 inference restore: R_abs = R_s * R_rel, convert back to native local representation."""
    r_s = repr_to_matrix(state_rot, repr_name, quat_order)  # [..., 3, 3]
    r_rel = d6_to_matrix(rel_6d)  # [..., T, 3, 3]
    r_abs = r_s.unsqueeze(-3) @ r_rel
    return matrix_to_repr(r_abs, repr_name, quat_order)
