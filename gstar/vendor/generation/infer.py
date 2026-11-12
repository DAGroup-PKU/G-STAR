"""c2i wrapper around the vendored GenerationImageHead pipeline.

Public entry point: generate_c2i_batch(model, class_ids, ...).

Builds the LLM prompt for each class id (cond + uncond when CFG > 1.0), runs
the KV-cache AR sampler from pipeline_image_head, decodes the resulting
RVQ-token grid back to pixels via the model's own visual_quantizer.

This follows the upstream native multimodal t2i flow but drops everything
c2i doesn't need (anyres prefix, SFT roles, img_newline,
spatial-merger MTP).
"""
from __future__ import annotations

import contextlib
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image

from .pipeline_image_head import GenerationImageHead


def _denorm_to_pil(t: torch.Tensor) -> Image.Image:
    # decode_to_pixels 的输出范围随 tokenizer 而异：我们的 RVQ vit_decoder 末端 sigmoid ->
    # [0,1]；BAR 的像素解码器输出 [-1,1]（与其 0.5 归一化训练一致）。用 min<0 自动判别并把
    # [-1,1] 平移回 [0,1]，两种 tokenizer 都对。
    t = t.detach().float()
    if t.amin() < -0.01:
        t = t * 0.5 + 0.5
    arr = t.clamp(0, 1).cpu().numpy()
    if arr.ndim == 4:
        arr = arr[0]
    arr = (arr.transpose(1, 2, 0) * 255).round().astype(np.uint8)
    return Image.fromarray(arr)


def _resolve_visual_tokenizer(model, visual_tokenizer):
    """解析出解码用的独立视觉 tokenizer（契约方法 decode_to_pixels 的持有者）。

    视觉 tokenizer 已解耦：优先用外部传入的独立 tokenizer（BAR / 我们的 RVQ 导出，都实现
    decode_to_pixels 契约）；回退到旧 bundled 模型内嵌的 model.visual_tokenizer（兼容解耦前
    的 merged ckpt）。两者都没有则报错并提示传参。
    """
    if visual_tokenizer is not None:
        return visual_tokenizer
    _inner = getattr(model, "model", model)
    _vt = getattr(_inner, "visual_tokenizer", None)
    if _vt is None:
        raise RuntimeError(
            "视觉 tokenizer 已解耦，LLM 不再内置 visual_tokenizer；"
            "请传 visual_tokenizer=<独立视觉 tokenizer>。")
    return _vt


@torch.no_grad()
def generate_c2i_batch(
    model,
    class_ids: Sequence[int],
    *,
    cls_base_id: int = 151704,
    uncond_token_id: int = 152704,
    image_size: int = 256,
    cfg_scale: float = 2.0,
    top_k: int = 0,
    top_p: float = 1.0,
    temperature: float = 1.0,
    do_sample: bool = True,
    visual_tokenizer: object = None,
    refine_gamma: float = None,
) -> List[Image.Image]:
    """Generate one PIL image per class id via KV-cache AR.

    Inputs (per sample):
      cond:   [<cls_K>, <img_start>]
      uncond: [<cls_uncond>, <img_start>]

    The pipeline's generate() expands cond+uncond into the batch dim when
    cfg_scale != 1.0; we only build the cond half here and pass cfg_scale.
    """
    cfg = model.config
    patch_size = cfg.visual_config.patch_size
    sm = cfg.visual_config.spatial_merge_size
    # Token-grid factor is merge_size (the tokenizer's patch->code downsample),
    # NOT spatial_merge_size. They coincide for 16x-nobridge (both 1) and
    # 32x tokenizers (both 2), but diverge for repar_16x (sm=2, merge_size=1):
    # the BAR ViT still groups patches 2x2 (spatial_merge_size) yet quantizes
    # before the merge (merger_after_quant=false) so the code grid stays at the
    # full patch grid. This mirrors the encode/token-count formula in
    # processor_omni.py (patch_nums = grid.prod() // merge_size**2 // mms**2).
    mgs = int(getattr(cfg.visual_config, "merge_size", None) or sm)
    if image_size % (patch_size * sm) != 0:
        image_size = (image_size // (patch_size * sm)) * (patch_size * sm)
    code_h = image_size // patch_size // mgs
    code_w = code_h
    n_codes = code_h * code_w
    # LLM-side SpatialMerger/Unmerger: the LM emits one image token per
    # mms×mms block, and image_head's SpatialUnmerger fans each LM token back
    # out to mms_unit RVQ positions. So the AR loop runs n_codes/mms_unit LM
    # steps to produce n_codes RVQ positions. mms==1 -> n_lm_steps==n_codes.
    mms = int(getattr(cfg.visual_config, "merger_merge_size", 1) or 1)
    mms_unit = mms * mms
    n_lm_steps = n_codes // mms_unit
    img_start_id = int(cfg.visual_config.image_start_token_id)
    device = next(model.parameters()).device

    cond_rows = [[int(cls_base_id) + int(c), img_start_id] for c in class_ids]
    cond_ids = torch.tensor(cond_rows, device=device, dtype=torch.long)

    if cfg_scale != 1.0:
        uncond_rows = [[int(uncond_token_id), img_start_id] for _ in class_ids]
        uncond_ids = torch.tensor(uncond_rows, device=device, dtype=torch.long)
        input_ids = torch.cat([cond_ids, uncond_ids], dim=0)
    else:
        input_ids = cond_ids

    attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=device)
    images_grid = [(1, code_h * mgs, code_w * mgs)]

    # Toggle use_cache for the duration of this call — training defaults set
    # config.use_cache=False, but generate() needs it on.
    prev_use_cache = getattr(model.config, "use_cache", None)
    was_training = model.training
    model.config.use_cache = True
    model.eval()
    try:
        out = GenerationImageHead.generate(
            model,
            input_ids=input_ids,
            attention_mask=attention_mask,
            images=None,
            patch_nums=None,
            images_grid=images_grid,
            labels=None,
            past_key_values=None,
            max_new_tokens=n_lm_steps,
            finish_until_max_new_tokens=True,
            num_beams=1,
            do_sample=do_sample,
            top_k=int(top_k) if top_k and top_k > 0 else 0,
            top_p=float(top_p),
            temperature=float(temperature),
            num_return_sequences=1,
            output_hidden_states=False,
            return_dict_in_generate=True,
            use_cache=True,
            cfg_scale=cfg_scale,
            refine_gamma=refine_gamma,
        )
    finally:
        model.config.use_cache = prev_use_cache if prev_use_cache is not None else False
        if was_training:
            model.train()

    vision_sequences = out.vision_sequences  # [B_total, n_codes, depth]
    # CFG path returns cond+uncond rows; keep cond half only.
    if cfg_scale != 1.0:
        vision_sequences = vision_sequences[: len(class_ids)]

    # Clamp any oob ids (EOS row id == codebook_size) before codebook lookup —
    # mirrors modeling_vision._decode_vision_sequences_to_features safeguard.
    cb_sizes = list(cfg.visual_quantizer_config.codebook_sizes)
    vs = vision_sequences.clone()
    for i, n in enumerate(cb_sizes):
        vs[..., i].clamp_(max=int(n) - 1)

    # Decode: id grid -> pixels via the tokenizer's decode_to_pixels contract
    # (BAR / RVQ 都实现；内部各自 embed_code + 像素解码，不再由 eval 伸手摸 .visual_quantizer)。
    tok = _resolve_visual_tokenizer(model, visual_tokenizer)
    # With mms>1 the generated vision_sequences are block-major (each LM step
    # emitted an mms×mms sub-block; the SpatialUnmerger output is block-major).
    # decode_to_pixels expects a raster [B, H, W, depth] grid, so invert
    # spatial_merge_reorder here. For mms==1 this is identity.
    depth = vs.shape[-1]
    mh, mw = code_h // mms, code_w // mms
    indices = (
        vs.view(len(class_ids), mh, mw, mms, mms, depth)
        .permute(0, 1, 3, 2, 4, 5)
        .reshape(len(class_ids), code_h, code_w, depth)
    )
    target_h = code_h * patch_size * mgs
    target_w = code_w * patch_size * mgs

    images: List[Image.Image] = []
    _ac = getattr(tok, "_needs_bf16_autocast", False)
    _dctx = torch.autocast("cuda", dtype=torch.bfloat16) if (_ac and device.type == "cuda") \
        else contextlib.nullcontext()
    for b in range(len(class_ids)):
        with _dctx:
            rec = tok.decode_to_pixels(
                indices[b : b + 1], code_h, code_w, target_h, target_w,
            )
        images.append(_denorm_to_pil(rec))
    return images


@torch.no_grad()
def generate_t2i_batch(
    model,
    tokenizer,
    prompts: Sequence[str],
    *,
    image_size: int = 256,
    image_height: Optional[int] = None,
    image_width: Optional[int] = None,
    cfg_scale: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    temperature: float = 1.0,
    do_sample: bool = True,
    max_prompt_tokens: int = 384,
    visual_tokenizer: object = None,
    user_token_id: Optional[int] = None,
    assistant_token_id: Optional[int] = None,
) -> List[Image.Image]:
    """Text-to-image analog of generate_c2i_batch.

    Same KV-cache AR + block-major decode as c2i; the ONLY difference is the
    prompt: instead of a single <cls_K> token the cond row is the tokenized
    caption followed by <img_start>. mms (SpatialMerger) and block_ar are
    handled transparently by GenerationImageHead / image_head.forward, exactly
    as in c2i — this is why we route through the same pipeline instead of the
    hand-rolled depth loop that ignored merger_merge_size.

    Pass user_token_id and assistant_token_id for SFT checkpoints. Both CFG
    branches then use the role tokens. Any-resolution checkpoints also receive
    the processor's size prefix in the compressed language-model grid.

    Processes one prompt per generate() call (batch=1, plus an empty-caption
    uncond row when cfg_scale != 1.0) to avoid variable-length text padding.
    """
    cfg = model.config
    patch_size = cfg.visual_config.patch_size
    sm = cfg.visual_config.spatial_merge_size
    # Token-grid factor is merge_size (the tokenizer's patch->code downsample),
    # NOT spatial_merge_size. They coincide for 16x-nobridge (both 1) and
    # 32x tokenizers (both 2), but diverge for repar_16x (sm=2, merge_size=1):
    # the BAR ViT still groups patches 2x2 (spatial_merge_size) yet quantizes
    # before the merge (merger_after_quant=false) so the code grid stays at the
    # full patch grid. This mirrors the encode/token-count formula in
    # processor_omni.py (patch_nums = grid.prod() // merge_size**2 // mms**2).
    mgs = int(getattr(cfg.visual_config, "merge_size", None) or sm)
    requested_h = int(image_height) if image_height is not None else int(image_size)
    requested_w = int(image_width) if image_width is not None else int(image_size)
    factor = patch_size * mgs
    requested_h = max(factor, (requested_h // factor) * factor)
    requested_w = max(factor, (requested_w // factor) * factor)
    code_h = requested_h // patch_size // mgs
    code_w = requested_w // patch_size // mgs
    n_codes = code_h * code_w
    mms = int(getattr(cfg.visual_config, "merger_merge_size", 1) or 1)
    mms_unit = mms * mms
    n_lm_steps = n_codes // mms_unit
    img_start_id = int(cfg.visual_config.image_start_token_id)
    device = next(model.parameters()).device

    cb_sizes = list(cfg.visual_quantizer_config.codebook_sizes)
    tok = _resolve_visual_tokenizer(model, visual_tokenizer)
    mh, mw = code_h // mms, code_w // mms
    target_h = code_h * patch_size * mgs
    target_w = code_w * patch_size * mgs
    images_grid = [(1, code_h * mgs, code_w * mgs)]
    use_size_prefix = bool(getattr(getattr(model, "processor", None),
                                   "enable_image_token_size_prefix", False))
    prefix_ids = []
    if use_size_prefix:
        start_tag = tokenizer.convert_ids_to_tokens(cfg.visual_config.image_token_size_prefix_start_id)
        end_tag = tokenizer.convert_ids_to_tokens(cfg.visual_config.image_token_size_prefix_end_id)
        prefix_ids = tokenizer.encode(f"{start_tag}{mh} {mw}{end_tag}", add_special_tokens=False)
    if (user_token_id is None) != (assistant_token_id is None):
        raise ValueError("Provide both user_token_id and assistant_token_id for SFT inference")
    sft_wrap = user_token_id is not None

    prev_use_cache = getattr(model.config, "use_cache", None)
    was_training = model.training
    model.config.use_cache = True
    model.eval()
    images: List[Image.Image] = []
    try:
        for prompt in prompts:
            text_ids = tokenizer.encode(prompt, add_special_tokens=False)[:max_prompt_tokens]
            suffix = prefix_ids + [img_start_id]
            if sft_wrap:
                cond_row = [int(user_token_id)] + text_ids + [int(assistant_token_id)] + suffix
                uncond_row = [int(user_token_id), int(assistant_token_id)] + suffix
            else:
                cond_row = text_ids + suffix
                uncond_row = suffix
            if cfg_scale != 1.0:
                # uncond = empty caption; left-pad to cond length so <img_start>
                # is the final token in both rows (generation starts aligned).
                L = max(len(cond_row), len(uncond_row))
                pad_id = int(getattr(cfg, "pad_token_id", 0) or 0)
                cond_p = [pad_id] * (L - len(cond_row)) + cond_row
                uncond_p = [pad_id] * (L - len(uncond_row)) + uncond_row
                input_ids = torch.tensor([cond_p, uncond_p], device=device, dtype=torch.long)
                attention_mask = torch.tensor(
                    [[0] * (L - len(cond_row)) + [1] * len(cond_row),
                     [0] * (L - len(uncond_row)) + [1] * len(uncond_row)],
                    device=device, dtype=torch.long,
                )
            else:
                input_ids = torch.tensor([cond_row], device=device, dtype=torch.long)
                attention_mask = torch.ones_like(input_ids)

            out = GenerationImageHead.generate(
                model,
                input_ids=input_ids,
                attention_mask=attention_mask,
                images=None,
                patch_nums=None,
                images_grid=images_grid,
                labels=None,
                past_key_values=None,
                max_new_tokens=n_lm_steps,
                finish_until_max_new_tokens=True,
                num_beams=1,
                do_sample=do_sample,
                top_k=int(top_k) if top_k and top_k > 0 else 0,
                top_p=float(top_p),
                temperature=float(temperature),
                num_return_sequences=1,
                output_hidden_states=use_size_prefix,
                return_dict_in_generate=True,
                use_cache=True,
                cfg_scale=cfg_scale,
            )
            if out.vision_sequences.shape[1] != n_codes:
                raise RuntimeError(f"Generated {out.vision_sequences.shape[1]} codes, expected {n_codes}")
            vs = out.vision_sequences[:1].clone()  # cond row -> [1, n_codes, depth]
            for i, n in enumerate(cb_sizes):
                vs[..., i].clamp_(max=int(n) - 1)
            depth = vs.shape[-1]
            indices = (
                vs.view(1, mh, mw, mms, mms, depth)
                .permute(0, 1, 3, 2, 4, 5)
                .reshape(1, code_h, code_w, depth)
            )
            _ac = getattr(tok, "_needs_bf16_autocast", False)
            _dctx = torch.autocast("cuda", dtype=torch.bfloat16) \
                if (_ac and device.type == "cuda") else contextlib.nullcontext()
            with _dctx:
                rec = tok.decode_to_pixels(
                    indices, code_h, code_w, target_h, target_w,
                )
            images.append(_denorm_to_pil(rec))
    finally:
        model.config.use_cache = prev_use_cache if prev_use_cache is not None else False
        if was_training:
            model.train()
    return images
