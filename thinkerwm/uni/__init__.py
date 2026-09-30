"""Uni subpackage: four-modality joint processing layer for video/action/depth/und."""

from .branch import UniBranch, _compact_und_kv

__all__ = [
    "UniBranch",
    "_compact_und_kv",
]
