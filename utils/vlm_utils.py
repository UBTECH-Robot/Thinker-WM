# VLM input preprocessing (text+image / text-only -> processor inputs -> batch padding).

from qwen_vl_utils import process_vision_info
from typing import Any, Dict, List

import torch


def preprocess_vlm_messages(text_instruction: str, image_pil, processor):
    """Text+image VLM preprocessing: apply chat template + process images, return processor inputs."""
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_pil},
                {"type": "text", "text": text_instruction}
            ]
        }
    ]

    # Apply chat template
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    # Process vision info
    image_inputs, video_inputs = process_vision_info(messages)

    # Get final processor inputs
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )

    return inputs

def preprocess_vlm_text_only(text_instruction: str, processor):
    """Text-only VLM preprocessing (no chat template; empty text fallback encodes single newline)."""
    text = str(text_instruction)
    inputs = processor(text=[text], padding=True, return_tensors="pt")
    if inputs["input_ids"].shape[1] == 0:
        inputs = processor(text=["\n"], padding=True, return_tensors="pt")
    return inputs


def _process_vlm_inputs_batch(vlm_inputs: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """Batch padding and assembly of VLM inputs."""
    input_ids_list = [vlm_input['input_ids'] for vlm_input in vlm_inputs]
    pixel_values_list = [vlm_input.get('pixel_values') for vlm_input in vlm_inputs]
    image_grid_thw_list = [vlm_input.get('image_grid_thw') for vlm_input in vlm_inputs]
    attention_mask_list = [vlm_input.get('attention_mask') for vlm_input in vlm_inputs]

    max_seq_len = max(ids.shape[1] for ids in input_ids_list)
    padded_input_ids = []
    padded_attention_masks = []

    for ids, mask in zip(input_ids_list, attention_mask_list):
        if ids.shape[1] < max_seq_len:
            padding_size = max_seq_len - ids.shape[1]
            padding = torch.zeros(ids.shape[0], padding_size, dtype=ids.dtype, device=ids.device)
            padded_ids = torch.cat([ids, padding], dim=1)
            if mask is not None:
                mask_padding = torch.zeros(mask.shape[0], padding_size, dtype=mask.dtype, device=mask.device)
                padded_mask = torch.cat([mask, mask_padding], dim=1)
            else:
                padded_mask = None
        else:
            padded_ids = ids
            padded_mask = mask

        padded_input_ids.append(padded_ids)
        padded_attention_masks.append(padded_mask)

    # Batch everything
    return {
        'input_ids': torch.cat(padded_input_ids, dim=0),
        'pixel_values': torch.cat([pv for pv in pixel_values_list if pv is not None], dim=0) if pixel_values_list and any(pv is not None for pv in pixel_values_list) else None,
        'image_grid_thw': torch.cat([igt for igt in image_grid_thw_list if igt is not None], dim=0) if image_grid_thw_list and any(igt is not None for igt in image_grid_thw_list) else None,
        'attention_mask': torch.cat([mask for mask in padded_attention_masks if mask is not None], dim=0) if any(mask is not None for mask in padded_attention_masks) else None,
    }
