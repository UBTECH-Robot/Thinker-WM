from typing import Optional
from PIL.Image import Image
import logging
import torch
import torch.nn as nn

from utils.vlm_utils import _process_vlm_inputs_batch
from utils.vlm_utils import preprocess_vlm_messages
from utils.vlm_utils import preprocess_vlm_text_only

logger = logging.getLogger(__name__)

class UndEncoder(nn.Module):
    """Residual MLP refinement stack for und tokens."""

    def __init__(self, dim, ffn_dim, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, ffn_dim),
                nn.GELU(approximate='tanh'),
                nn.Linear(ffn_dim, dim)
            )
            for _ in range(num_layers)
        ])

        self.norms = nn.ModuleList([
            nn.LayerNorm(dim)
            for _ in range(num_layers)
        ])

    def forward(self, x):
        for layer, norm in zip(self.layers, self.norms):
            x = x + (layer(norm(x)))
        return x

class UndBranch(nn.Module):
    """und branch: frozen VLM last-hidden -> bridge + UndEncoder -> und token triplet."""

    def __init__(self, vlm_preprocess, vlm_model, dim, ffn_dim, num_layers, vlm_dim=2048,
                 max_und_tokens: int = 1000):
        super().__init__()
        # VLM model reference
        self.vlm_preprocess = vlm_preprocess
        self.vlm_model = vlm_model
        # Empty-text VLM feature constant (through tokenizer + embedding only), lazily cached.
        self._uncond_vlm_feature = None
        # und token count upper bound (uni rope freqs 1024 rows); tail truncation on overflow.
        self.max_und_tokens = int(max_und_tokens)
        if not 0 < self.max_und_tokens <= 1024:
            raise ValueError(
                f"max_und_tokens must satisfy 0 < max_und_tokens <= 1024 (uni branch "
                f"rope freqs row cap), got {self.max_und_tokens}"
            )
        self._last_truncated = 0
        self._truncated_warn_left = 10  # warning rate limit

        self.bridge = nn.Linear(vlm_dim, dim)

        # Understanding Expert reference
        self.und_encoder = UndEncoder(
            dim=dim,
            ffn_dim=ffn_dim,
            num_layers=num_layers
        )

    def extract_und_features(
        self,
        prompt_inputs: list[str],
        first_frame_pil: Optional[list[Image]]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """VLM last-hidden encoding: returns (und_token [B, L_padded, vlm_dim],
        real_lens [B]); first_frame_pil=None uses text-only path."""
        if first_frame_pil is None:
            vlm_inputs = [
                preprocess_vlm_text_only(v, self.vlm_preprocess)
                for v in prompt_inputs
            ]
        else:
            vlm_inputs = [preprocess_vlm_messages(v, f, self.vlm_preprocess) for v, f in zip(prompt_inputs, first_frame_pil)]
        vlm_inputs = _process_vlm_inputs_batch(vlm_inputs)

        if isinstance(vlm_inputs, list):
            B = len(vlm_inputs)
        else:
            B = vlm_inputs['input_ids'].shape[0]

        # Returns: inputs_embeds, attention_mask, visual_pos_masks, deepstack_image_embeds, position_ids
        inputs_embeds, attention_mask, visual_pos_masks, deepstack_image_embeds, position_ids = self._process_vlm_inputs_to_tokens(vlm_inputs, B)

        if attention_mask is not None:
            real_lens = attention_mask.sum(dim=1).to(torch.long)
        else:
            real_lens = torch.full(
                (B,), inputs_embeds.shape[1], dtype=torch.long,
                device=inputs_embeds.device)

        # Forward through VLM with proper attention_mask and DeepStack features
        vlm_kwargs = {
            'inputs_embeds': inputs_embeds,
            'attention_mask': attention_mask,
            'position_ids': position_ids,
            'past_key_values': None,
            'use_cache': False,
            'output_attentions': False,
            'output_hidden_states': False,
            'return_dict': True
        }

        # Add DeepStack parameters for Qwen3-VL
        if visual_pos_masks is not None:
            vlm_kwargs['visual_pos_masks'] = visual_pos_masks
        if deepstack_image_embeds is not None:
            vlm_kwargs['deepstack_visual_embeds'] = deepstack_image_embeds

        with torch.no_grad():
            vlm_output = self.vlm_model.model.language_model(**vlm_kwargs)

        # Extract last layer features directly without keeping every hidden state.
        und_token = getattr(vlm_output, 'last_hidden_state', None)
        if und_token is None:
            und_token = vlm_output[0]

        return und_token, real_lens

    def _process_vlm_inputs_to_tokens(self, vlm_inputs, B: int) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[list], torch.Tensor]:
        """VLM inputs -> (inputs_embeds, attention_mask, visual_pos_masks,
        deepstack_image_embeds, position_ids)."""
        # Handle both old format (List[Dict]) and new format (Dict[str, Tensor])
        device = self.und_encoder.layers[0][0].weight.device
        dtype = self.und_encoder.layers[0][0].weight.dtype
        if isinstance(vlm_inputs, list):
            # Old format: List[Dict] - do padding and batching
            input_ids_list = [vlm_input['input_ids'] for vlm_input in vlm_inputs]
            attention_mask_list = [vlm_input.get('attention_mask') for vlm_input in vlm_inputs]
            pixel_values_list = [vlm_input.get('pixel_values') for vlm_input in vlm_inputs]
            image_grid_thw_list = [vlm_input.get('image_grid_thw') for vlm_input in vlm_inputs]

            # Pad input_ids and attention_mask to same length
            max_seq_len = max(ids.shape[1] for ids in input_ids_list)
            padded_input_ids = []
            padded_attention_masks = []

            for ids, mask in zip(input_ids_list, attention_mask_list):
                if ids.shape[1] < max_seq_len:
                    padding_size = max_seq_len - ids.shape[1]
                    # Pad input_ids with zeros
                    id_padding = torch.zeros(ids.shape[0], padding_size, dtype=ids.dtype, device=ids.device)
                    padded_ids = torch.cat([ids, id_padding], dim=1)
                    # Pad attention_mask with zeros (padding tokens should be ignored)
                    mask_padding = torch.zeros(mask.shape[0], padding_size, dtype=mask.dtype, device=mask.device)
                    padded_mask = torch.cat([mask, mask_padding], dim=1)
                else:
                    padded_ids = ids
                    padded_mask = mask
                padded_input_ids.append(padded_ids)
                padded_attention_masks.append(padded_mask)

            # Batch process
            input_ids_batch = torch.cat(padded_input_ids, dim=0).to(device)
            attention_mask_batch = torch.cat(padded_attention_masks, dim=0).to(device)
            pixel_values_batch = torch.cat([pv.to(device) for pv in pixel_values_list], dim=0)
            image_grid_thw_batch = torch.cat([igt.to(device) for igt in image_grid_thw_list], dim=0)
        else:

            # New format: Dict[str, Tensor] - already batched and padded
            input_ids_batch = vlm_inputs['input_ids'].to(device)
            attention_mask_batch = vlm_inputs['attention_mask'].to(device)
            pixel_values = vlm_inputs.get('pixel_values')
            image_grid_thw = vlm_inputs.get('image_grid_thw')
            pixel_values_batch = (
                pixel_values.to(device) if pixel_values is not None else None
            )
            image_grid_thw_batch = (
                image_grid_thw.to(device) if image_grid_thw is not None else None
            )

        # Get input embeddings
        inputs_embeds = self.vlm_model.get_input_embeddings()(input_ids_batch)

        if pixel_values_batch is not None:
            # Process images - handle different return formats between Qwen2.5-VL and Qwen3-VL
            image_embeds, deepstack_image_embeds = self.vlm_model.get_image_features(pixel_values_batch, image_grid_thw_batch)

            image_embeds = torch.cat(image_embeds, dim=0).to(device, dtype)

            # Insert image embeddings
            image_mask, _ = self.vlm_model.model.get_placeholder_mask(
                input_ids_batch, inputs_embeds=inputs_embeds, image_features=image_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

            visual_pos_masks = image_mask[..., 0]  # [B, seq_len] - visual positions only
        else:
            # Text-only path: no image tokens, DeepStack/visual masks all empty.
            visual_pos_masks = None
            deepstack_image_embeds = None

        # Compute position_ids (position_ids remains as original: [3, B, seq_len])
        # Qwen3-VL get_rope_index has different signature: (input_ids, image_grid_thw, video_grid_thw, attention_mask)
        position_ids, _rope_deltas = self.vlm_model.model.get_rope_index(
            input_ids=input_ids_batch,
            image_grid_thw=image_grid_thw_batch,
            video_grid_thw=None,  # No video in current implementation
            attention_mask=attention_mask_batch
        )

        return inputs_embeds, attention_mask_batch, visual_pos_masks, deepstack_image_embeds, position_ids

    @torch.no_grad()
    def get_uncond_vlm_feature(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Empty-text VLM feature constant (through tokenizer + embedding layer), returns
        (hidden [1, L_e, vlm_dim], real_len [1]). Empty text convention =
        UNCONDITIONAL_VLM_MESSAGE ("" defined in thinkerwm/thinker_wm/infer.py;
        not imported here to avoid circular dependency)."""
        if self._uncond_vlm_feature is None:
            device = self.und_encoder.layers[0][0].weight.device
            inputs = preprocess_vlm_text_only("", self.vlm_preprocess)
            input_ids = inputs["input_ids"].to(device)
            hidden = self.vlm_model.get_input_embeddings()(input_ids)
            real_len = torch.tensor(
                [int(input_ids.shape[1])], dtype=torch.long, device=device)
            self._uncond_vlm_feature = (hidden, real_len)
        return self._uncond_vlm_feature

    def precompute_vlm_features(
        self,
        prompt_inputs: list[str],
        first_frame_pil: Optional[list[Image]],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """VLM section of und encoding: empty text uses constant, non-empty uses extract_und_features."""
        texts = [str(t) for t in prompt_inputs]
        B = len(texts)
        empty_idx = [i for i, t in enumerate(texts) if not t.strip()]
        if not empty_idx:
            return self.extract_und_features(texts, first_frame_pil)
        uncond_hidden, uncond_len = self.get_uncond_vlm_feature()
        if len(empty_idx) == B:
            return uncond_hidden.expand(B, -1, -1), uncond_len.repeat(B)
        empty_set = set(empty_idx)
        nonempty = [i for i in range(B) if i not in empty_set]
        sub_pils = (
            [first_frame_pil[i] for i in nonempty]
            if first_frame_pil is not None else None
        )
        sub_hidden, sub_lens = self.extract_und_features(
            [texts[i] for i in nonempty], sub_pils)
        l_e = int(uncond_hidden.shape[1])
        l_max = max(int(sub_hidden.shape[1]), l_e)
        hidden = sub_hidden.new_zeros(B, l_max, sub_hidden.shape[2])
        real_lens = torch.empty(B, dtype=torch.long, device=sub_hidden.device)
        for row, i in enumerate(nonempty):
            l_i = int(sub_lens[row])
            hidden[i, :l_i] = sub_hidden[row, :l_i]
            real_lens[i] = sub_lens[row]
        for i in empty_idx:
            hidden[i, :l_e] = uncond_hidden[0].to(hidden.dtype)
            real_lens[i] = uncond_len[0]
        return hidden, real_lens

    def encode_from_vlm_hidden(
        self,
        vlm_hidden: torch.Tensor,
        real_lens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Bridge section of und encoding: VLM last-hidden -> bridge + und_encoder ->
        (und_token, grid_sizes, seq_lens); truncates tail if exceeding max_und_tokens."""
        cap = self.max_und_tokens
        if vlm_hidden.size(1) > cap:
            self._last_truncated = int((real_lens > cap).sum().item())
            if self._truncated_warn_left > 0:
                self._truncated_warn_left -= 1
                logger.warning(
                    "und token exceeds max_und_tokens=%d (batch padded length %d), "
                    "truncating tail to prevent crash (uni rope 1024 cap); warning rate-limited",
                    cap, int(vlm_hidden.size(1)),
                )
            vlm_hidden = vlm_hidden[:, :cap]
            real_lens = real_lens.clamp(max=cap)
        else:
            self._last_truncated = 0
        und_tokens = self.bridge(vlm_hidden)
        und_tokens = self.und_encoder(und_tokens)

        bs = und_tokens.size(0)
        l_und = und_tokens.size(1)

        # seq_lens are per-sample real (non-padded) lengths: mask padding positions in K/V side;
        # Q side uses full length (flash_attention varlen requires full q length).
        seq_lens = real_lens

        # und uses text-like 1D RoPE: grid = [L_und, 1, 1].
        grid_sizes = torch.tensor([l_und, 1, 1]).repeat(bs, 1)

        return und_tokens, grid_sizes, seq_lens

    def encode(self, prompt_inputs: list[str], first_frame_pil: Optional[list[Image]]) -> torch.Tensor:
        """Full und encoding = precompute_vlm_features (VLM section) +
        encode_from_vlm_hidden (bridge section)."""
        vlm_hidden, real_lens = self.precompute_vlm_features(
            prompt_inputs, first_frame_pil)
        return self.encode_from_vlm_hidden(vlm_hidden, real_lens)
