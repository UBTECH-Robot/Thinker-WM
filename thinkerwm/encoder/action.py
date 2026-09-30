"""
action encoder: unified space [B, T, U] sampled down via Conv3d with stride t_downsample into tokens
(one token per d steps), processed through WanAttentionBlock, outputs (x, grid_sizes, seq_lens, e0).
"""
import torch
import torch.nn as nn

from wan.modules.model import (
    WanAttentionBlock,
    sinusoidal_embedding_1d,
    rope_params,
)


class ActionEncoder(nn.Module):
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

        # Unified action space: unified dimension U directly serves as model-internal action dim (no projection).
        self.unified_action_dim = unified_action_dim

        self.action_embedding = nn.Conv3d(
            unified_action_dim, wan_dim,
            kernel_size=(self.t_downsample, 1, 1),
            stride=(self.t_downsample, 1, 1)
        )

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

        # initialize weights
        self.init_weights()

    def _check_action_dims(self, action):
        """Check that last dim matches unified dimension U."""
        assert action.size(-1) == self.unified_action_dim, (
            f"action dim mismatch: expected unified_action_dim={self.unified_action_dim}, "
            f"got {action.size(-1)}"
        )

    def forward(
        self,
        action,
        t,
        context,
        robot_names=None,
        context_lens=None,
        robot_token_embedding=None,
    ):
        """
        action [B,T,U] -> downsampled token sequence (action token i occupies RoPE time position i);
        robot tokens prepended to cross-attn context.
        """
        self._check_action_dims(action)

        # Temporal downsampling: all T steps grouped by stride=t_downsample (one token per d steps).
        T = action.size(1)
        assert T % self.t_downsample == 0, (
            f"action steps T={T} not divisible by t_downsample={self.t_downsample}"
        )

        # (B, T, D) -> (B, D, T, 1, 1)
        x = action.permute(0, 2, 1).unsqueeze(-1).unsqueeze(-1)

        device = self.action_embedding.weight.device
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)

        # embeddings
        x = self.action_embedding(x)
        x = x.flatten(2).transpose(1, 2)    # (B, T'', wan_dim)

        # grid built on downsampled action token sequence (no state prefix token).
        seq_len = x.size(1)
        grid_sizes = torch.stack(
            [torch.tensor([seq_len, 1, 1], dtype=torch.long) for _ in range(x.size(0))])

        seq_lens = torch.full((x.size(0),), seq_len, dtype=torch.long)

        # time embeddings
        if t.dim() == 1:
            t = t.expand(x.size(0), seq_len)
        elif t.dim() == 2 and t.size(1) == 1:
            t = t.expand(t.size(0), seq_len)

        with torch.amp.autocast('cuda', dtype=torch.float32):
            bt = t.size(0)
            t = t.flatten()
            e = self.time_embedding(
                sinusoidal_embedding_1d(self.freq_dim,
                                        t).unflatten(0, (bt, seq_len)).float())
            e0 = self.time_projection(e).unflatten(2, (6, self.dim))
            assert e.dtype == torch.float32 and e0.dtype == torch.float32

        if not torch.is_tensor(context):
            raise TypeError(
                "ActionEncoder context must be a projected und token tensor "
                "([B, L_und, dim])."
            )

        # robot tokens prepended to und context (context_lens incremented by +8).
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

        return x, grid_sizes, seq_lens, e0

    def init_weights(self):
        """Xavier initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # init embeddings
        nn.init.xavier_uniform_(self.action_embedding.weight.flatten(1))
        for m in self.time_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)
