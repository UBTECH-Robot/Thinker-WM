import copy
import math
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from wan.modules.model import WanModel


class VideoDecoder(nn.Module):
    """Video decoder: last decode_layer_num Wan blocks + output head
    (with depth output stream: dedicated head, shared decode blocks)."""

    def __init__(self, wan_model: WanModel,
                 decode_layer_num,
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

        self.decode_block = wan_model.blocks[-decode_layer_num:]

        self.head = wan_model.head

        self.freqs = wan_model.freqs
        self.gradient_checkpointing = gradient_checkpointing

        # Depth output stream: dedicated output head. Not created when enable_depth=False.
        self.enable_depth = bool(enable_depth)
        if self.enable_depth:
            self.depth_head = copy.deepcopy(wan_model.head)

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

    def decode(self, x, context, grid_sizes, seq_lens, e, e0, context_lens=None,
               robot_names=None, robot_token_embedding=None):
        """video token through decode blocks + head + unpatchify,
        returns [C_out, F, H, W] list."""
        device = self.head.head.weight.device
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
            context_lens=context_lens
        )
        for i, block in enumerate(self.decode_block):
            x = self._run_block(block, x, e0, context, kwargs)

        x = self.head(x, e)

        x = self.unpatchify(x, grid_sizes)
        return [u.float() for u in x]

    def decode_depth(self, x, context, grid_sizes, seq_lens, e, e0,
                     context_lens=None, robot_names=None,
                     robot_token_embedding=None):
        """Same as decode but uses depth-specific output head."""
        assert self.enable_depth
        device = self.head.head.weight.device
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)

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
            context_lens=context_lens
        )
        for i, block in enumerate(self.decode_block):
            x = self._run_block(block, x, e0, context, kwargs)

        x = self.depth_head(x, e)
        x = self.unpatchify(x, grid_sizes)
        return [u.float() for u in x]

    def unpatchify(self, x, grid_sizes):
        """Restore patch sequence to video tensor [C_out, F, H, W] list."""

        c = self.out_dim
        out = []
        for u, v in zip(x, grid_sizes.tolist()):
            u = u[:math.prod(v)].view(*v, *self.patch_size, c)
            u = torch.einsum('fhwpqrc->cfphqwr', u)
            u = u.reshape(c, *[i * j for i, j in zip(v, self.patch_size)])
            out.append(u)
        return out
