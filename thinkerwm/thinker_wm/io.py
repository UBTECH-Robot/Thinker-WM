"""Inference input/output file I/O (images / video / npy / first-frame depth)."""

from __future__ import annotations

import os
from typing import Optional, Tuple

import numpy as np
import torch
import PIL.Image

from . import depth_codec


def load_image(path: str, size: Optional[Tuple[int, int]] = None) -> torch.Tensor:
    """Load image as [3, H, W] float32 [0,1] (RGB); resizes if size=(height, width) given."""
    img = PIL.Image.open(path).convert("RGB")
    if size is not None:
        h, w = int(size[0]), int(size[1])
        if img.size != (w, h):  # PIL size is (W, H)
            img = img.resize((w, h), PIL.Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0  # [H, W, 3]
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


def load_video(path: str, size: Optional[Tuple[int, int]] = None) -> torch.Tensor:
    """Load video as [T, 3, H, W] float32 [0,1] (RGB). decord preferred, cv2 fallback."""
    frames = None
    try:
        import decord

        vr = decord.VideoReader(path)
        frames = vr.get_batch(range(len(vr))).asnumpy()  # [T, H, W, 3] uint8
    except Exception:
        frames = None
    if frames is None:
        import cv2

        cap = cv2.VideoCapture(path)
        chunks = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            chunks.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()
        if not chunks:
            raise RuntimeError(f"Could not decode video: {path}")
        frames = np.stack(chunks, axis=0)
    video = torch.from_numpy(frames.astype(np.float32) / 255.0).permute(0, 3, 1, 2)
    if size is not None:
        h, w = int(size[0]), int(size[1])
        if video.shape[-2:] != (h, w):
            video = torch.nn.functional.interpolate(
                video, size=(h, w), mode="bilinear", align_corners=False
            )
    return video.contiguous()


def save_video(video: torch.Tensor, path: str, fps: int = 5) -> None:
    """Write [3, T, H, W] or [T, 3, H, W] float [0,1] (or uint8) as mp4."""
    if not torch.is_tensor(video):
        video = torch.as_tensor(video)
    video = video.detach().cpu()
    if video.ndim != 4:
        raise ValueError(f"save_video expects 4D tensor, got {tuple(video.shape)}")
    if video.shape[0] == 3 and video.shape[1] != 3:
        video = video.permute(1, 0, 2, 3)  # [3,T,H,W] -> [T,3,H,W]
    elif video.shape[-3] == 3 and video.shape[0] != 3:
        pass  # already [T,3,H,W]
    elif video.shape[0] == 3:
        video = video.permute(1, 0, 2, 3)
    if video.dtype != torch.uint8:
        video = (video.clamp(0, 1) * 255.0).round().to(torch.uint8)
    frames = video.permute(0, 2, 3, 1).numpy()  # [T, H, W, 3]
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    try:
        import imageio.v2 as imageio

        imageio.mimsave(path, frames, fps=int(fps), macro_block_size=1)
        return
    except Exception:
        pass
    import cv2

    t, h, w, _ = frames.shape
    writer = cv2.VideoWriter(
        path, cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (w, h)
    )
    if not writer.isOpened():
        raise RuntimeError(f"cv2 could not write video: {path}")
    for frame in frames:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()


def load_npy(path: str) -> torch.Tensor:
    """Load .npy as torch.Tensor (float32; integer types like uint16 returned as-is)."""
    arr = np.load(path)
    if arr.dtype.kind == "f":
        arr = arr.astype(np.float32)
    return torch.from_numpy(arr)


def save_npy(tensor: torch.Tensor, path: str) -> None:
    """Write tensor as .npy (float32)."""
    if not torch.is_tensor(tensor):
        tensor = torch.as_tensor(tensor)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    np.save(path, tensor.detach().cpu().float().numpy())


def load_depth_first(
    path: str,
    size: Tuple[int, int],
    z_near: float,
    z_far: float,
    mode: str = "log",
) -> torch.Tensor:
    """Load first-frame depth condition as [3, H, W] float32 [0,1] (codec-domain grayscale):
    float .npy treated as already encoded, uint16 .npy as millimeter depth, image loaded as grayscale."""
    h, w = int(size[0]), int(size[1])
    if path.endswith(".npy"):
        arr = np.load(path)
        if arr.dtype == np.uint16:
            depth = torch.from_numpy(arr.astype(np.float32))  # mm
            out = depth_codec.encode_depth_mm(
                depth, z_near=float(z_near), z_far=float(z_far),
                mode=mode, out_range="01",
            )  # [3, H, W] in [0,1]
        else:
            out = torch.from_numpy(arr.astype(np.float32))
            if out.ndim == 2:
                out = out.unsqueeze(0).expand(3, -1, -1).contiguous()
            elif out.ndim == 3 and out.shape[0] == 1:
                out = out.expand(3, -1, -1).contiguous()
            elif out.ndim == 3 and out.shape[0] == 3:
                pass
            else:
                raise ValueError(
                    f"depth_first .npy shape not supported: {tuple(out.shape)}"
                    " (expected [H,W]/[1,H,W]/[3,H,W])"
                )
    else:
        img = PIL.Image.open(path).convert("L")
        out = torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0)
        out = out.unsqueeze(0).expand(3, -1, -1).contiguous()
    if out.shape[-2:] != (h, w):
        out = torch.nn.functional.interpolate(
            out.unsqueeze(0), size=(h, w), mode="bilinear", align_corners=False
        ).squeeze(0)
    return out.clamp(0.0, 1.0).contiguous()
