import copy
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from wan.modules.model import WanModel, sinusoidal_embedding_1d


class VideoEncoder(nn.Module):
    """Video encoder: patch embedding + time embedding + first encode_layer_num
    Wan blocks (with depth input stream: dedicated patch embedding + stream_embed)."""

    def __init__(self, wan_model: WanModel,
                 encode_layer_num,
                 gradient_checkpointing: bool = False,
                 enable_depth: bool = False):
        super().__init__()

        self.patch_size = wan_model.patch_size
        self.in_dim = wan_model.in_dim
        self.dim = wan_model.dim
        self.ffn_dim = wan_model.ffn_dim
        self.freq_dim = wan_model.freq_dim
        self.out_dim = wan_model.out_dim
        self.num_heads = wan_model.num_heads
        self.num_layers = wan_model.num_layers
        self.window_size = wan_model.window_size
        self.qk_norm = wan_model.qk_norm
        self.cross_attn_norm = wan_model.cross_attn_norm
        self.eps = wan_model.eps

        self.patch_embedding = wan_model.patch_embedding
        # Do not reference wan_model.text_embedding (T5 path, unused in this model).
        self.time_embedding = wan_model.time_embedding
        self.time_projection = wan_model.time_projection

        self.encode_blocks = wan_model.blocks[:encode_layer_num]

        self.freqs = wan_model.freqs
        self.gradient_checkpointing = gradient_checkpointing

        # Depth input stream: dedicated patch embedding + zero-init stream_embed (differentiates depth/RGB).
        self.enable_depth = bool(enable_depth)
        if self.enable_depth:
            self.depth_patch_embedding = copy.deepcopy(wan_model.patch_embedding)
            self.depth_stream_embed = nn.Parameter(torch.zeros(1, 1, self.dim))

    def _run_block(self, block, x, e0, context, kwargs):
        if not (self.gradient_checkpointing and self.training):
            return block(x, **kwargs)

        def block_forward(x, e0, context):
            block_kwargs = dict(kwargs)
            block_kwargs["e"] = e0
            block_kwargs["context"] = context
            return block(x, **block_kwargs)

        return checkpoint(
            block_forward,
            x,
            e0,
            context,
            use_reentrant=False,
            preserve_rng_state=False,
        )

    def encode(self, x, t, context, context_lens=None, robot_names=None,
               robot_token_embedding=None):
        """video patchify + time embedding (first-frame token timestep set to 0) + encode blocks;
        context is projected und token, robot tokens prepended. Returns
        (x, grid_sizes, seq_lens, e, e0)."""
        # params
        device = self.patch_embedding.weight.device
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)

        # embeddings
        x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
        grid_sizes = torch.stack(
            [torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
        x = [u.flatten(2).transpose(1, 2) for u in x]
        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
        seq_len = seq_lens[0]
        assert seq_lens.max() <= seq_len
        x = torch.cat([
            torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))],
                      dim=1) for u in x
        ])

        # time embeddings
        if t.dim() == 1:
            t = t.expand(x.size(0), seq_len)
        elif t.dim() == 2 and t.size(1) == 1:
            t = t.expand(t.size(0),seq_len)

        # First frame is clean conditioning frame: set all h*w token timesteps to 0.
        t = t.clone()  # avoid in-place modification
        tokens_per_frame = int(grid_sizes[0, 1] * grid_sizes[0, 2])
        t[:, :tokens_per_frame] = 0.0

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
                "VideoEncoder.encode context must be a projected und token "
                "tensor ([B, L_und, dim])."
            )

        # robot tokens prepended to und context (context_lens incremented by +8).
        rt = None
        if robot_names is not None and robot_token_embedding is not None:
            rt = robot_token_embedding(robot_names)

        if rt is not None:
            context = torch.cat([rt.to(context.dtype), context], dim=1)
            if context_lens is not None:
                context_lens = context_lens + 8
        kwargs = dict(
            e=e0,
            seq_lens=seq_lens,
            grid_sizes=grid_sizes,
            freqs=self.freqs,
            context=context,
            context_lens=context_lens)
        for i, block in enumerate(self.encode_blocks):
            x = self._run_block(block, x, e0, context, kwargs)
        return x, grid_sizes, seq_lens, e, e0

    def encode_depth(self, x, t, context, context_lens=None, robot_names=None,
                     robot_token_embedding=None):
        """Depth stream encoder: same as encode but with depth-specific patch embedding +
        stream_embed, sharing encode blocks."""
        assert self.enable_depth
        device = self.depth_patch_embedding.weight.device
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)

        x = [self.depth_patch_embedding(u.unsqueeze(0)) for u in x]
        grid_sizes = torch.stack(
            [torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
        x = [u.flatten(2).transpose(1, 2) for u in x]
        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
        seq_len = seq_lens[0]
        assert seq_lens.max() <= seq_len
        x = torch.cat([
            torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))],
                      dim=1) for u in x
        ])
        # stream_embed (zero-init) added to each depth token.
        x = x + self.depth_stream_embed.to(x.dtype)

        if t.dim() == 1:
            t = t.expand(x.size(0), seq_len)
        elif t.dim() == 2 and t.size(1) == 1:
            t = t.expand(t.size(0), seq_len)
        # First-frame token timestep set to 0 (consistent with video).
        t = t.clone()
        tokens_per_frame = int(grid_sizes[0, 1] * grid_sizes[0, 2])
        t[:, :tokens_per_frame] = 0.0

        with torch.amp.autocast('cuda', dtype=torch.float32):
            bt = t.size(0)
            t = t.flatten()
            e = self.time_embedding(
                sinusoidal_embedding_1d(self.freq_dim,
                                        t).unflatten(0, (bt, seq_len)).float())
            e0 = self.time_projection(e).unflatten(2, (6, self.dim))
            assert e.dtype == torch.float32 and e0.dtype == torch.float32

        rt = None
        if robot_names is not None and robot_token_embedding is not None:
            rt = robot_token_embedding(robot_names)
        if rt is not None:
            context = torch.cat([rt.to(context.dtype), context], dim=1)
            if context_lens is not None:
                context_lens = context_lens + 8
        kwargs = dict(
            e=e0,
            seq_lens=seq_lens,
            grid_sizes=grid_sizes,
            freqs=self.freqs,
            context=context,
            context_lens=context_lens)
        for i, block in enumerate(self.encode_blocks):
            x = self._run_block(block, x, e0, context, kwargs)
        return x, grid_sizes, seq_lens, e, e0
