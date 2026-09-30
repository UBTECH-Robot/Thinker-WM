import copy

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from wan.modules.model import (
    WanModel,
    WanSelfAttention,
    WanCrossAttention,
    WanAttentionBlock,
    rope_apply,
    flash_attention,
    sinusoidal_embedding_1d
)


def _compact_und_kv(und_a, und_b, seq_lens):
    """Compact-concatenate two right-padded und token sequences by per-sample real length, returns
    (ctx [B, 2L, D], 2*seq_lens) (used for und as cross-attn K/V)."""
    b, l, d = und_a.shape
    device = und_a.device
    lens = seq_lens.to(device=device, dtype=torch.long).clamp(0, l).unsqueeze(1)
    pos = torch.arange(2 * l, device=device).unsqueeze(0)       # [1, 2L]
    in_a = pos < lens                                           # [B, 2L]
    in_b = (pos >= lens) & (pos < 2 * lens)                     # [B, 2L]
    # a section takes pos, b section takes pos-l (compact offset); padding indices clamped
    # then zeroed by trailing mask.
    idx = torch.where(in_a, pos, pos - lens).clamp(0, l - 1)    # [B, 2L]
    idx = idx.unsqueeze(-1).expand(-1, -1, d)
    gathered_a = torch.gather(und_a, 1, idx)
    gathered_b = torch.gather(und_b, 1, idx)
    ctx = torch.where(in_a.unsqueeze(-1), gathered_a, gathered_b)
    ctx = ctx * (in_a | in_b).unsqueeze(-1).to(ctx.dtype)
    return ctx, lens.squeeze(1) * 2


class UniBranch(nn.Module):
    def __init__(self, wan_model:WanModel,
                 encode_layer_num, decode_layer_num,
                 gradient_checkpointing: bool = False,
                 action_rope_time_scale: float = 0.25,
                 depth_uni_clone_layers: int = 0):
        super().__init__()
        assert encode_layer_num + decode_layer_num < len(wan_model.blocks)

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

        self.uni_blocks = wan_model.blocks[encode_layer_num:-decode_layer_num]

        # Depth-specific uni layers: independent copies of the last N uni layers,
        # video/action/und use original layers.
        n_clone = int(depth_uni_clone_layers or 0)
        assert 0 <= n_clone <= len(self.uni_blocks), (
            f"depth_uni_clone_layers={n_clone} exceeds uni layer count {len(self.uni_blocks)}")
        self.depth_uni_clone_layers = n_clone
        self.depth_uni_first_layer = len(self.uni_blocks) - n_clone
        self.depth_uni_blocks = nn.ModuleList(
            copy.deepcopy(b)
            for b in self.uni_blocks[self.depth_uni_first_layer:]
        ) if n_clone > 0 else None

        self.head = wan_model.head

        self.freqs = wan_model.freqs
        # Action token RoPE temporal frequency scaling (action token i corresponds to latent time
        # i * scale, h/w position always 0).
        ang = torch.angle(wan_model.freqs)
        self.freqs_action = torch.polar(
            torch.ones_like(ang), ang * action_rope_time_scale)
        self.gradient_checkpointing = gradient_checkpointing
    
    @staticmethod
    def self_attn_proc(
            model: WanSelfAttention,
            video_token, video_grid_sizes,
            action_token, action_grid_sizes,
            und_token, und_grid_sizes,
            freqs, freqs_action,
            video_seq_lens, action_seq_lens, und_seq_lens,
            depth_token=None, depth_grid_sizes=None, depth_seq_lens=None,
            depth_model=None,
            depth_detach_video_kv=False):
        r"""Directed self-attention: video/action read {video, action} (do not read depth),
        depth reads {video, action, depth} (video K/V detached when depth_detach_video_kv=True),
        und reads itself. Missing modalities skipped; und_seq_lens used only for K/V side
        mask padding positions (Q side uses full length)."""
        b, n, d = und_token.shape[0], model.num_heads, model.head_dim
        L_und = und_token.size(1)

        uq = model.norm_q(model.q(und_token)).view(b, L_und, n, d)
        uk = model.norm_k(model.k(und_token)).view(b, L_und, n, d)
        uv = model.v(und_token).view(b, L_und, n, d)
        # und: 1D text-like rope (grid [L,1,1])
        uq = rope_apply(uq, und_grid_sizes, freqs)
        uk = rope_apply(uk, und_grid_sizes, freqs)

        # Present A-group modalities each have their own q/k/v + RoPE (video/depth: 3D; action: freqs_action).
        qkv = {}
        a_L = {}
        slens = {}
        for name, tok, gs, rfreqs, sl in (
            ("video", video_token, video_grid_sizes, freqs, video_seq_lens),
            ("action", action_token, action_grid_sizes, freqs_action, action_seq_lens),
            ("depth", depth_token, depth_grid_sizes, freqs, depth_seq_lens),
        ):
            if tok is None:
                continue
            # In cloned layers, depth uses its copy's projections, other modalities use original layer.
            proj = depth_model if (name == "depth" and depth_model is not None) else model
            Ln = tok.size(1)
            q = proj.norm_q(proj.q(tok)).view(b, Ln, n, d)
            k = proj.norm_k(proj.k(tok)).view(b, Ln, n, d)
            v = proj.v(tok).view(b, Ln, n, d)
            q = rope_apply(q, gs, rfreqs)
            k = rope_apply(k, gs, rfreqs)
            qkv[name] = (q, k, v)
            a_L[name] = Ln
            slens[name] = sl

        def kv_sources(reader):
            if reader == "video":
                srcs = ["video", "action"]                 # never read depth
            elif reader == "depth":
                srcs = ["video", "action", "depth"]        # unidirectionally reads RGB/action + self
            else:  # action same as video: never reads depth
                srcs = ["video", "action"]
            return [s for s in srcs if s in qkv]

        # Each reader calls flash once with its own K/V set (directed visibility).
        a_out = {}
        for reader in ("video", "action", "depth"):
            if reader not in qkv:
                continue
            srcs = kv_sources(reader)
            ks, vs = [], []
            for s in srcs:
                k_s, v_s = qkv[s][1], qkv[s][2]
                # depth detaches video K/V when reading.
                if depth_detach_video_kv and reader == "depth" and s == "video":
                    k_s, v_s = k_s.detach(), v_s.detach()
                ks.append(k_s)
                vs.append(v_s)
            k = torch.cat(ks, dim=1)
            v = torch.cat(vs, dim=1)
            k_lens = None
            for s in srcs:
                k_lens = slens[s] if k_lens is None else k_lens + slens[s]
            a_out[reader] = flash_attention(
                qkv[reader][0], k, v,
                q_lens=slens[reader], k_lens=k_lens,
                window_size=model.window_size,
            )

        # und <-> und: Q side full length, K side uses real length to mask padding positions.
        und_token = flash_attention(
            uq, uk, uv,
            q_lens=None, k_lens=und_seq_lens,
            window_size=model.window_size,
        )

        order = [m for m in ("video", "action", "depth") if m in qkv]
        # Output projection: in cloned layers, depth section uses its copy's o, other sections computed once.
        split_depth_o = depth_model is not None and "depth" in a_out
        main = [m for m in order if not (split_depth_o and m == "depth")]
        x = torch.cat([a_out[m] for m in main] + [und_token], dim=1).flatten(2)
        x = model.o(x)
        parts = dict(zip(main + ["und"], x.split([a_L[m] for m in main] + [L_und], dim=1)))
        if split_depth_o:
            parts["depth"] = depth_model.o(a_out["depth"].flatten(2))
        return (
            parts.get("video"),
            parts.get("action"),
            parts.get("depth"),
            parts["und"],
        )
    
    @staticmethod
    def cross_attn_proc(
            model: WanCrossAttention,
            video_token, action_token, und_token,
            video_seq_lens, action_seq_lens, und_seq_lens,
            und_token_init=None,
            depth_token=None, depth_seq_lens=None, depth_model=None):
        r"""Directed cross-attention: A(video/action/depth) query und (K/V = compact
        concatenation [und token 0, und token t], single segment when und_token_init=None);
        und only queries video+action (not depth). q_lens/k_lens convention same as self_attn_proc;
        when depth_model is a clone layer, depth section q/o uses copy, und K/V detached."""
        b, n, d = und_token.shape[0], model.num_heads, model.head_dim
        L_und = und_token.size(1)

        uq = model.norm_q(model.q(und_token)).view(b, L_und, n, d)
        if und_token_init is not None:
            # A->B K/V = compact concatenation [und token 0, und token t].
            und_kv, ab_k_lens = _compact_und_kv(
                und_token_init, und_token, und_seq_lens)
        else:
            und_kv, ab_k_lens = und_token, und_seq_lens
        uk = model.norm_k(model.k(und_kv)).view(b, und_kv.size(1), n, d)
        uv = model.v(und_kv).view(b, und_kv.size(1), n, d)

        qkv = {}
        a_L = {}
        slens = {}
        for name, tok, sl in (
            ("video", video_token, video_seq_lens),
            ("action", action_token, action_seq_lens),
            ("depth", depth_token, depth_seq_lens),
        ):
            if tok is None:
                continue
            # In cloned layers, depth uses its copy's projections, other modalities use original layer.
            proj = depth_model if (name == "depth" and depth_model is not None) else model
            Ln = tok.size(1)
            q = proj.norm_q(proj.q(tok)).view(b, Ln, n, d)
            k = proj.norm_k(proj.k(tok)).view(b, Ln, n, d)
            v = proj.v(tok).view(b, Ln, n, d)
            qkv[name] = (q, k, v)
            a_L[name] = Ln
            slens[name] = sl

        # A -> B: video/action/depth query und (k_lens uses und real length).
        order = [m for m in ("video", "action", "depth") if m in qkv]
        split_depth = depth_model is not None and "depth" in qkv
        a_out = {}
        main = [m for m in order if not (split_depth and m == "depth")]
        if main:
            a_q = torch.cat([qkv[m][0] for m in main], dim=1)
            group_seq_lens = None
            for m in main:
                group_seq_lens = slens[m] if group_seq_lens is None else group_seq_lens + slens[m]
            out_ab = flash_attention(
                a_q, uk, uv,
                q_lens=group_seq_lens,
                k_lens=ab_k_lens,
                window_size=model.window_size,
            )
            a_out = dict(zip(main, out_ab.split([a_L[m] for m in main], dim=1)))
        if split_depth:
            # In cloned layer, depth does separate flash (und K/V detached).
            a_out["depth"] = flash_attention(
                qkv["depth"][0], uk.detach(), uv.detach(),
                q_lens=slens["depth"],
                k_lens=ab_k_lens,
                window_size=depth_model.window_size,
            )

        # B -> A: und only queries video+action (Q side full length).
        b_order = [m for m in ("video", "action") if m in qkv]
        if b_order:
            b_k = torch.cat([qkv[m][1] for m in b_order], dim=1)
            b_v = torch.cat([qkv[m][2] for m in b_order], dim=1)
            b_seq_lens = None
            for m in b_order:
                b_seq_lens = slens[m] if b_seq_lens is None else b_seq_lens + slens[m]
            und_from_ab = flash_attention(
                uq, b_k, b_v,
                q_lens=None,
                k_lens=b_seq_lens,
                window_size=model.window_size,
            )
        else:
            # video/action both missing (should not happen): und identity.
            und_from_ab = uq

        # Output projection: in cloned layers, depth section uses copy's o.
        x = torch.cat([a_out[m] for m in main] + [und_from_ab], dim=1).flatten(2)
        x = model.o(x)
        parts = dict(zip(main + ["und"], x.split([a_L[m] for m in main] + [L_und], dim=1)))
        if split_depth:
            parts["depth"] = depth_model.o(a_out["depth"].flatten(2))
        return (
            parts.get("video"),
            parts.get("action"),
            parts.get("depth"),
            parts["und"],
        )

    @staticmethod
    def attn_block_proc(
            model: WanAttentionBlock,
            video_token, video_grid_sizes,
            action_token, action_grid_sizes,
            und_token, und_grid_sizes,
            freqs, freqs_action, e,
            video_seq_lens, action_seq_lens, und_seq_lens,
            und_token_init=None,
            depth_token=None, depth_grid_sizes=None, depth_seq_lens=None,
            depth_model=None,
            depth_detach_video_kv=False):
        r"""Single uni block: modulation -> self-attn (directed) -> cross-attn -> FFN.
        e segment order (video, action, depth, und); when depth_model is not None,
        depth segment's parameterized modules use cloned copies (norm1/norm2 are stateless,
        computed once for all segments)."""
        assert e.dtype == torch.float32

        tokens = {"video": video_token, "action": action_token,
                  "depth": depth_token, "und": und_token}
        # Segment order fixed (video, action, depth, und), matching e construction order.
        present = [m for m in ("video", "action", "depth", "und") if tokens[m] is not None]
        split_depth = depth_model is not None and "depth" in present

        def cat_present(dct):
            return torch.cat([dct[m] for m in present], dim=1)

        def split_present(x, ref):
            return dict(zip(present, x.split([ref[m].size(1) for m in present], dim=1)))

        def seg_apply(fn, depth_fn, dct):
            """Apply parameterized module per segment: depth segment uses depth_fn, rest computed once together."""
            if not split_depth:
                return split_present(fn(cat_present(dct)), dct)
            main = [m for m in present if m != "depth"]
            out = fn(torch.cat([dct[m] for m in main], dim=1))
            res = dict(zip(main, out.split([dct[m].size(1) for m in main], dim=1)))
            res["depth"] = depth_fn(dct["depth"])
            return res

        with torch.amp.autocast('cuda', dtype=torch.float32):
            if split_depth:
                # depth segment modulation uses its copy.
                dim = model.modulation.shape[-1]
                mod = torch.cat([
                    (depth_model if m == "depth" else model).modulation
                    .unsqueeze(0).expand(1, tokens[m].size(1), 6, dim)
                    for m in present
                ], dim=1)
            else:
                mod = model.modulation.unsqueeze(0)
            e = (mod + e).chunk(6, dim=2)
        assert e[0].dtype == torch.float32

        # self-attention (directed visibility handled inside self_attn_proc)
        normed = (model.norm1(cat_present(tokens)).float()
                  * (1 + e[1].squeeze(2)) + e[0].squeeze(2))
        normed = split_present(normed, tokens)

        sa = UniBranch.self_attn_proc(
            model.self_attn,
            normed.get("video"), video_grid_sizes,
            normed.get("action"), action_grid_sizes,
            normed["und"], und_grid_sizes,
            freqs, freqs_action,
            video_seq_lens, action_seq_lens, und_seq_lens,
            depth_token=normed.get("depth"), depth_grid_sizes=depth_grid_sizes,
            depth_seq_lens=depth_seq_lens,
            depth_model=depth_model.self_attn if split_depth else None,
            depth_detach_video_kv=depth_detach_video_kv,
        )
        sa = {"video": sa[0], "action": sa[1], "depth": sa[2], "und": sa[3]}

        with torch.amp.autocast('cuda', dtype=torch.float32):
            x = cat_present(tokens) + torch.cat([sa[m] for m in present], dim=1) * e[2].squeeze(2)
            cur = split_present(x, tokens)

        # cross attention: A(query und) <-> und(query video,action)
        pre_cross = cur
        cur_norm = seg_apply(model.norm3, depth_model.norm3 if split_depth else None, pre_cross)
        ca = UniBranch.cross_attn_proc(
            model.cross_attn,
            cur_norm.get("video"), cur_norm.get("action"), cur_norm["und"],
            video_seq_lens, action_seq_lens, und_seq_lens,
            und_token_init=und_token_init,
            depth_token=cur_norm.get("depth"), depth_seq_lens=depth_seq_lens,
            depth_model=depth_model.cross_attn if split_depth else None,
        )
        ca = {"video": ca[0], "action": ca[1], "depth": ca[2], "und": ca[3]}
        for m in present:
            cur[m] = pre_cross[m] + ca[m]

        # FFN (norm2 stateless, computed once for all; depth segment uses copy's ffn).
        ffn_in = (model.norm2(cat_present(cur)) *
                  (1 + e[4].squeeze(2)) + e[3].squeeze(2))
        if split_depth:
            ffn_y = seg_apply(model.ffn, depth_model.ffn, split_present(ffn_in, cur))
            y = torch.cat([ffn_y[m] for m in present], dim=1)
        else:
            y = model.ffn(ffn_in)
        with torch.amp.autocast('cuda', dtype=torch.float32):
            y = y * e[5].squeeze(2)
            ffn_out = split_present(y, cur)
            for m in present:
                cur[m] = cur[m] + ffn_out[m]

        # return (missing modality returns None)
        return (
            cur.get("video"),
            cur.get("action"),
            cur.get("depth"),
            cur["und"],
        )

    def uni_process(self,
                    video_token, video_grid_sizes,
                    action_token, action_grid_sizes,
                    und_token, und_grid_sizes,
                    t_video, t_action,
                    video_seq_lens, action_seq_lens, und_seq_lens,
                    depth_token=None, depth_grid_sizes=None, depth_seq_lens=None,
                    t_depth=None,
                    depth_detach_video_kv=False,
                    und_token_init=None):
        """Process all four modalities through all uni layers, returns (video, action, depth, und) tokens;
        video/action/depth can be None, und is always present."""
        video_present = video_token is not None
        action_present = action_token is not None
        depth_present = depth_token is not None
        video_seq_len = video_token.size(1) if video_present else 0
        action_seq_len = action_token.size(1) if action_present else 0
        depth_seq_len = depth_token.size(1) if depth_present else 0
        und_seq_len = und_token.size(1)
        seq_len = video_seq_len + action_seq_len + depth_seq_len + und_seq_len

        if t_video is None:
            t_video = t_action
        if t_action is None:
            t_action = t_video
        # depth defaults to same noise schedule as video (t_depth=None -> uses t_video).
        if t_depth is None:
            t_depth = t_video
        base_t = t_video if t_video is not None else t_action
        if base_t is None:
            raise ValueError("uni_process: at least one of t_video or t_action must be non-None")
        if base_t.dim() == 1:
            base_t = base_t.unsqueeze(1)
        if t_video is not None and t_video.dim() == 1:
            t_video = t_video.unsqueeze(1)
        if t_action is not None and t_action.dim() == 1:
            t_action = t_action.unsqueeze(1)
        if t_depth is not None and t_depth.dim() == 1:
            t_depth = t_depth.unsqueeze(1)

        # time embeddings (built per-segment according to present modalities, segment order video/action/depth/und).
        segs = []
        if video_present:
            # First frame is clean conditioning frame: all h*w tokens set to 0.
            tv = t_video.expand(t_video.size(0), video_seq_len).clone()
            tokens_per_frame = int(video_grid_sizes[0, 1] * video_grid_sizes[0, 2])
            tv[:, :tokens_per_frame] = 0.0
            segs.append(tv)
        if action_present:
            ta = t_action.expand(t_action.size(0), action_seq_len).clone()
            segs.append(ta)
        if depth_present:
            # Depth first frame is also clean conditioning frame.
            td = t_depth.expand(t_depth.size(0), depth_seq_len).clone()
            depth_tokens_per_frame = int(depth_grid_sizes[0, 1] * depth_grid_sizes[0, 2])
            td[:, :depth_tokens_per_frame] = 0.0
            segs.append(td)
        # und token is clean VLM condition: entire segment set to 0.
        tu = base_t.expand(base_t.size(0), und_seq_len).clone()
        tu[:] = 0.0
        segs.append(tu)
        t = torch.cat(segs, dim=1)

        with torch.amp.autocast('cuda', dtype=torch.float32):
            bt = t.size(0)
            t = t.flatten()
            e = self.time_embedding(
                sinusoidal_embedding_1d(self.freq_dim,
                                        t).unflatten(0, (bt, seq_len)).float())
            e0 = self.time_projection(e).unflatten(2, (6, self.dim))
            assert e.dtype == torch.float32 and e0.dtype == torch.float32

        self.freqs = self.freqs.to(und_token.device)
        self.freqs_action = self.freqs_action.to(und_token.device)
        for i, block in enumerate(self.uni_blocks):
            # Clone zone (last N uni layers): depth uses its own copy.
            depth_block = (
                self.depth_uni_blocks[i - self.depth_uni_first_layer]
                if (self.depth_uni_blocks is not None and i >= self.depth_uni_first_layer)
                else None
            )
            # Missing modalities still go through checkpoint (otherwise all uni layers stay active in GPU memory during dropout).
            if self.gradient_checkpointing and self.training:
                def block_forward(video_token, action_token, depth_token, und_token, und_token_init, e0,
                                  block=block, depth_block=depth_block):
                    return self.attn_block_proc(
                        model=block,
                        video_token=video_token, video_grid_sizes=video_grid_sizes,
                        action_token=action_token, action_grid_sizes=action_grid_sizes,
                        und_token=und_token, und_grid_sizes=und_grid_sizes,
                        freqs=self.freqs, freqs_action=self.freqs_action, e=e0,
                        video_seq_lens=video_seq_lens,
                        action_seq_lens=action_seq_lens,
                        und_seq_lens=und_seq_lens,
                        und_token_init=und_token_init,
                        depth_token=depth_token, depth_grid_sizes=depth_grid_sizes,
                        depth_seq_lens=depth_seq_lens,
                        depth_model=depth_block,
                        depth_detach_video_kv=depth_detach_video_kv)

                video_token, action_token, depth_token, und_token = checkpoint(
                    block_forward,
                    video_token,
                    action_token,
                    depth_token,
                    und_token,
                    und_token_init,
                    e0,
                    use_reentrant=False,
                    preserve_rng_state=False,
                )
            else:
                (video_token, action_token, depth_token, und_token) = self.attn_block_proc(
                    model=block,
                    video_token=video_token, video_grid_sizes=video_grid_sizes,
                    action_token=action_token, action_grid_sizes=action_grid_sizes,
                    und_token=und_token, und_grid_sizes=und_grid_sizes,
                    freqs=self.freqs, freqs_action=self.freqs_action, e=e0,
                    video_seq_lens=video_seq_lens,
                    action_seq_lens=action_seq_lens,
                    und_seq_lens=und_seq_lens,
                    und_token_init=und_token_init,
                    depth_token=depth_token, depth_grid_sizes=depth_grid_sizes,
                    depth_seq_lens=depth_seq_lens,
                    depth_model=depth_block,
                    depth_detach_video_kv=depth_detach_video_kv)

        return video_token, action_token, depth_token, und_token
