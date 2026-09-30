"""Action decoder: stacked WanAttentionBlock + action_unembedding directly outputting unified dimension U."""
import torch
import torch.nn as nn

from wan.modules.model import (
    WanAttentionBlock,
    rope_params,
)


class ActionDecoder(nn.Module):
    def __init__(self,
                 layer_num,
                 wan_dim, wan_ffn_dim, wan_num_heads,
                 freq_dim=256,
                 t_downsample=24, video_action_freq_ratio=6,
                 unified_action_dim=64):
        super().__init__()
        self.freq_dim = freq_dim
        self.dim = wan_dim
        self.t_downsample = t_downsample
        self.video_action_freq_ratio = video_action_freq_ratio

        # Unified action space: unembedding directly outputs unified dimension U.
        self.unified_action_dim = unified_action_dim

        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, wan_dim),
            nn.SiLU(),
            nn.Linear(wan_dim, wan_dim)
        )

        self.time_projection = nn.Sequential(
            nn.SiLU(),
            nn.Linear(wan_dim, wan_dim * 6)
        )

        self.blocks = nn.ModuleList([
            WanAttentionBlock(wan_dim, wan_ffn_dim, wan_num_heads)
            for _ in range(layer_num)
        ])

        # buffers (don't use register_buffer otherwise dtype will be changed in to())
        assert (wan_dim % wan_num_heads) == 0 and (wan_dim // wan_num_heads) % 2 == 0
        d = wan_dim // wan_num_heads
        self.freqs = torch.cat([
            rope_params(1024, d - 4 * (d // 6)),
            rope_params(1024, 2 * (d // 6)),
            rope_params(1024, 2 * (d // 6))
        ], dim=1)

        self.action_unembedding = nn.Linear(wan_dim, t_downsample * unified_action_dim)

        # initialize weights
        self.init_weights()

    def forward(
        self,
        x,
        context,
        grid_sizes,
        seq_lens,
        e0,
        robot_names=None,
        context_lens=None,
        robot_token_embedding=None,
    ):
        """action token through decode blocks + unembedding, outputs [B, T, U] list."""
        device = self.time_embedding[0].weight.device
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)

        # robot tokens prepended to und context (matching encoder side).
        rt = None
        if robot_names is not None and robot_token_embedding is not None:
            rt = robot_token_embedding(robot_names)

        if rt is not None:
            context = torch.cat([rt.to(context.dtype), context], dim=1)
            if context_lens is not None:
                context_lens = context_lens + 8
        # arguments
        kwargs = dict(
            e=e0,
            seq_lens=seq_lens,
            grid_sizes=grid_sizes,
            freqs=self.freqs,
            context=context,
            context_lens=context_lens)
        for i, block in enumerate(self.blocks):
            x = block(x, **kwargs)

        # (B, L, D): L = T // t_downsample (no state prefix token).
        x_action = x  # (B, T // t_downsample, D)
        bs = x_action.size(0)

        # action unemb
        x_action = self.action_unembedding(x_action)
        ts = x_action.size(1)
        x = x_action.view(bs, ts * self.t_downsample, -1)  # (B, T, U)

        # Slicing back to per-robot original dimension is handled by model boundary (infer.py).
        return [xb for xb in x]   ## List[(T, unified_action_dim)]

    def init_weights(self):
        """Xavier initialization."""

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # init embeddings
        for m in self.time_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)
