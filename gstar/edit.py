"""Source-image conditioned editing with independent text and image guidance."""
import contextlib
import json
import math
from pathlib import Path

import numpy as np
import torch

from .vendor.generation.infer import _denorm_to_pil
from .vendor.generation.pipeline_image_head import GenerationImageHead
from .vendor.visual_tokenizer import encode_images_to_vision_tokens, get_reorder_fn


@torch.inference_mode()
def generate_edit(model, tokenizer, visual, image_path, instruction, *,
                  user_token_id, assistant_token_id, height=1024, width=1024,
                  text_cfg=1.8, image_cfg=1.0, top_k=4096, top_p=0.75,
                  temperature=0.7, seed=42, max_prompt_tokens=1024):
    """Return image, generated codes and input metadata for one local edit.

    Branch order is full, image-only, then empty when image_cfg != 1.
    Source-image preprocessing follows the checkpoint's multimodal processor.
    Output dimensions must match its processed source dimensions.
    """
    path = Path(image_path).resolve(strict=True)
    if not instruction.strip():
        raise ValueError("Editing requires a nonempty instruction")
    if not all(math.isfinite(x) and x >= 0 for x in (text_cfg, image_cfg)):
        raise ValueError("Guidance scales must be finite and nonnegative")
    if temperature <= 0 or not 0 < top_p <= 1 or top_k < 0:
        raise ValueError("Invalid sampling parameters")
    vc = model.config.visual_config
    mms = int(getattr(vc, "merger_merge_size", 1) or 1)
    merge = int(getattr(vc, "merge_size", None) or vc.spatial_merge_size)
    patch = int(vc.patch_size)
    factor = patch * merge * mms
    if min(height, width) <= 0 or height % factor or width % factor:
        raise ValueError("Output dimensions must be positive multiples of " + str(factor))
    if not model.processor.enable_image_token_size_prefix or not model.processor.enable_image_line_token:
        raise ValueError("Editing requires image-size prefixes and image newline tokens")
    code_h, code_w = height // (patch * merge), width // (patch * merge)
    prompt_length = len(tokenizer.encode(instruction, add_special_tokens=False))
    if prompt_length > max_prompt_tokens:
        raise ValueError("Editing instruction exceeds max_prompt_tokens; truncation is disabled")
    media = '<img_start>' + json.dumps({'type': 'file', 'path': str(path)},
                                     ensure_ascii=False, separators=(',', ':')) + '<img_end>'
    processed = [model.processor.process_one(media + instruction),
                 model.processor.process_one(media)]
    three_branches = image_cfg != 1.0
    if any(p.input_ids is None for p in processed):
        raise ValueError("Processor rejected the edit input")
    source_grids = list(processed[0].images_grid)
    if len(source_grids) != 1 or tuple(source_grids[0]) != (1, height // patch, width // patch):
        raise ValueError("Processed source dimensions must match output dimensions; "
                         "resize the source image first or request matching dimensions")
    prefix = tokenizer.encode('<img_token_size>' + f'{code_h // mms} {code_w // mms}'
                              + '</img_token_size>', add_special_tokens=False)
    suffix = [int(assistant_token_id)] + prefix + [int(vc.image_start_token_id)]
    rows = [[int(user_token_id)] + list(p.input_ids) + suffix for p in processed]
    if three_branches:
        rows.append([int(user_token_id)] + suffix)
    # Text CFG 1 with image CFG 1 needs no duplicated condition branch.
    if not three_branches and text_cfg == 1.0:
        processed, rows = processed[:1], rows[:1]
    device = next(model.parameters()).device
    length = max(map(len, rows))
    input_ids = torch.full((len(rows), length), int(model.config.pad_token_id),
                           dtype=torch.long, device=device)
    attention_mask = torch.zeros_like(input_ids)
    for i, row in enumerate(rows):
        input_ids[i, -len(row):] = torch.tensor(row, dtype=torch.long, device=device)
        attention_mask[i, -len(row):] = 1
    images = [torch.as_tensor(x, device=device, dtype=torch.float32)
              for p in processed for x in (p.images or [])]
    grids = [g for p in processed for g in (p.images_grid or [])]
    patch_nums = torch.tensor([n for p in processed for n in (p.patch_nums or [])],
                              dtype=torch.int32, device=device)
    reorder = get_reorder_fn(model)
    if mms > 1 and reorder is None:
        raise RuntimeError("Spatial merge requires block-major reorder support")
    vision_tokens = encode_images_to_vision_tokens(
        visual, images, grids, mms=mms, reorder_fn=reorder)
    expected_pads = int((input_ids == int(vc.image_pad_token_id)).sum())
    if expected_pads != vision_tokens.shape[0]:
        raise ValueError("Source image pad/token count mismatch")
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    previous_cache, was_training = model.config.use_cache, model.training
    model.config.use_cache = True
    model.eval()
    try:
        out = GenerationImageHead.generate(
            model, input_ids=input_ids, attention_mask=attention_mask,
            images=None, vision_tokens=vision_tokens, images_grid=grids,
            patch_nums=patch_nums, labels=None, past_key_values=None,
            max_new_tokens=code_h * code_w // (mms * mms),
            finish_until_max_new_tokens=True, num_beams=1, do_sample=top_k != 1,
            top_k=top_k, top_p=top_p, temperature=temperature,
            num_return_sequences=1, output_hidden_states=True,
            return_dict_in_generate=True, use_cache=True, cfg_scale=text_cfg,
            image_cfg_scale=image_cfg if three_branches else None)
    finally:
        model.config.use_cache = previous_cache
        model.train(was_training)
    codes = out.vision_sequences[:1].clone()
    depth = len(model.config.visual_quantizer_config.codebook_sizes)
    if tuple(codes.shape) != (1, code_h * code_w, depth):
        raise RuntimeError("Incomplete edited image token grid: " + str(tuple(codes.shape)))
    for level, size in enumerate(model.config.visual_quantizer_config.codebook_sizes):
        if bool((codes[..., level] < 0).any() or (codes[..., level] >= size).any()):
            raise RuntimeError("Out-of-range edited image code")
    indices = (codes.view(1, code_h // mms, code_w // mms, mms, mms, depth)
               .permute(0, 1, 3, 2, 4, 5).reshape(1, code_h, code_w, depth))
    autocast = (torch.autocast('cuda', dtype=torch.bfloat16)
                if getattr(visual, '_needs_bf16_autocast', False) and device.type == 'cuda'
                else contextlib.nullcontext())
    with autocast:
        pixels = visual.decode_to_pixels(indices, code_h, code_w, height, width)
    if tuple(pixels.shape) != (1, 3, height, width) or not bool(torch.isfinite(pixels).all()):
        raise RuntimeError("Invalid edited image pixels")
    image = _denorm_to_pil(pixels)
    return {'image': image, 'codes': indices.cpu(), 'metadata': {
        'branch_lengths': [len(row) for row in rows], 'branches': len(rows),
        'prompt_tokens': prompt_length, 'prompt_truncated': False,
        'source_grid': source_grids, 'input_vision_tokens': expected_pads,
        'text_cfg': text_cfg, 'image_cfg': image_cfg, 'seed': seed,
        'code_shape': list(indices.shape), 'pixels_finite': True,
        'image_stddev': np.asarray(image).std(axis=(0, 1)).tolist()}}
