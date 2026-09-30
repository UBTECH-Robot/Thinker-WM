"""Per-robot 8 learnable token table (append-only slots, prepended to cross-attn context)."""
from typing import Dict, List, Optional, Union

import torch
import torch.nn as nn


class RobotTokenEmbedding(nn.Module):
    """Per-robot 8 learnable tokens; slots are explicitly assigned by YAML robot_token_index
    (no reuse/reorder allowed, ensuring ckpt row stability)."""

    def __init__(
        self,
        robot_token_indices: Dict[str, int],
        max_robots: int = 64,
        num_tokens: int = 8,
        dim: int = 0,
    ):
        super().__init__()
        if dim <= 0:
            raise ValueError(f"RobotTokenEmbedding dim must be > 0, got {dim}")
        if max_robots <= 0:
            raise ValueError(f"RobotTokenEmbedding max_robots must be > 0, got {max_robots}")
        if num_tokens <= 0:
            raise ValueError(f"RobotTokenEmbedding num_tokens must be > 0, got {num_tokens}")
        self.max_robots = max_robots
        self.num_tokens = num_tokens
        self.dim = dim
        # Append-only slot table: robot -> row index.
        self._indices: Dict[str, int] = {}
        for name, index in (robot_token_indices or {}).items():
            key = str(name)
            index = int(index)
            if not (0 <= index < max_robots):
                raise ValueError(
                    f"robot {key}: robot_token_index {index} exceeds capacity [0, {max_robots})"
                )
            if index in self._indices.values():
                raise ValueError(
                    f"robot {key}: robot_token_index {index} already occupied by another robot (no reuse)"
                )
            self._indices[key] = index
        # 0.02 std initialization (matching Wan DiT embedding scale).
        self.table = nn.Parameter(
            torch.empty(max_robots, num_tokens, dim).normal_(std=0.02)
        )

    def forward(
        self, robot_names: Optional[Union[str, List[str]]]
    ) -> Optional[torch.Tensor]:
        """Returns [B, num_tokens, dim]; returns None if robot_names is None/empty (no tokens prepended)."""
        if robot_names is None or len(robot_names) == 0:
            return None
        if isinstance(robot_names, str):
            robot_names = [robot_names]
        slots: List[int] = []
        for name in robot_names:
            key = str(name)
            if key not in self._indices:
                raise KeyError(
                    f"Unknown robot name {key!r}: robot_token_index not registered in "
                    f"YAML robot_names (registered: {sorted(self._indices)})"
                )
            slots.append(self._indices[key])
        idx = torch.tensor(slots, dtype=torch.long, device=self.table.device)
        return self.table[idx]  # [B, num_tokens, dim]
