"""Training/inference glue between the frozen visual tokenizer and the LLM.

The visual tokenizer is a standalone trust_remote_code model.  Non-cached data
paths (T2I pretrain / SFT) encode images on the fly: this module loads the
frozen tokenizer and turns one batch of images into block-major reordered
vision_tokens before the LLM forward.

Reordering happens on the encode side using the tokenizer-reported
out_grid_thw; the LLM only consumes already reordered indices.
"""

import os
import sys
import importlib

import numpy as np
import torch


def get_reorder_fn(model):
    """Fetch block_major_reorder_per_image from the loaded LLM's dynamic module.
    Returns None if unavailable (not needed when mms == 1)."""
    try:
        pkg = type(model).__module__.rsplit('.', 1)[0]
        sm = importlib.import_module(pkg + '.spatial_merger')
        return sm.block_major_reorder_per_image
    except Exception:
        return None


def _read_tokenizer_model_type(tok_path):
    """Return model_type from tok_path/config.json, or None if unreadable."""
    import json
    cfg_path = os.path.join(tok_path, "config.json")
    if not os.path.isfile(cfg_path):
        return None
    try:
        with open(cfg_path, "r", encoding="utf-8") as f:
            return json.load(f).get("model_type")
    except Exception:
        return None


def _needs_external_autocast(model_type):
    """Whether an AutoModel tokenizer needs a caller-owned bf16 autocast."""
    return model_type != "repar_ideogram4_unified"


def _disable_optional_refiner_color_fix(tok):
    """Disable exported wavelet/AdaIN color-fix hooks when explicitly requested."""
    value = os.environ.get("OMNI_DISABLE_WAVELET_COLOR_FIX", "")
    if value.strip().lower() not in {"1", "true", "yes", "on"}:
        return

    module_name = type(tok).__module__
    module = sys.modules.get(module_name) or importlib.import_module(module_name)
    disabled = []
    for name in ("wavelet_color_fix", "adain_color_fix"):
        if hasattr(module, name):
            setattr(module, name, None)
            disabled.append(name)
    tok._wavelet_color_fix_disabled = True
    print(
        "[visual-tokenizer] wavelet/AdaIN color fix disabled: "
        f"module={module_name}, hooks={disabled or ['not-present']}",
        flush=True,
    )


def load_frozen_visual_tokenizer(tok_path, llm_config, device, dtype=torch.bfloat16):
    """Load a frozen, self-describing visual tokenizer export.

    tok_path must be a Hugging Face directory whose config.json carries the
    tokenizer's own model_type and an auto_map; it is loaded with
    AutoModel.from_pretrained(trust_remote_code=True).  Any implementation that
    follows the encode/decode_to_pixels contract works without LLM changes.
    """
    model_type = _read_tokenizer_model_type(tok_path)
    if not model_type or model_type == "omni":
        raise ValueError(
            f"{tok_path} is not a self-describing visual tokenizer export: expected "
            "config.json with the tokenizer model_type and an auto_map"
        )
    from transformers import AutoModel
    tok = AutoModel.from_pretrained(
        tok_path, trust_remote_code=True, torch_dtype=dtype
    )
    # External tokenizers may keep bf16 weights with fp32 intermediates, which
    # needs a bf16 autocast around encode/decode.  The unified RepAR refiner
    # manages autocast internally; an outer autocast would leak into its
    # DiT/VAE and change refiner numerics.
    needs_autocast = _needs_external_autocast(model_type)
    _disable_optional_refiner_color_fix(tok)
    tok = tok.to(device).eval()
    tok.requires_grad_(False)
    tok._needs_bf16_autocast = needs_autocast
    return tok


@torch.no_grad()
def encode_images_to_vision_tokens(tokenizer, images, images_grid, mms=1, reorder_fn=None):
    """One batch of raw images -> block-major reordered vision_tokens.

    images:      List of float tensors, flattened patches per image (from the collator).
    images_grid: List of (1, h_patch, w_patch) per image.
    Returns:     [N_total, depth] int64 tensor, same format as cached vision_tokens.
    """
    dev = next(tokenizer.parameters()).device
    dtype = next(tokenizer.parameters()).dtype
    visual_input = torch.cat([img.to(dev, dtype=dtype) for img in images], dim=0)
    visual_grid = torch.tensor(np.array(images_grid), dtype=torch.long, device=dev)
    import contextlib
    ac = getattr(tokenizer, "_needs_bf16_autocast", False)
    ctx = torch.autocast("cuda", dtype=torch.bfloat16) if (ac and dev.type == "cuda") \
        else contextlib.nullcontext()
    with ctx:
        out = tokenizer.encode(visual_input, visual_grid, fake_input=False)
    idx = out.indices
    # Reorder with the tokenizer-reported out_grid_thw (= input grid // bridge
    # merge_size) instead of re-deriving the tokenizer's internal merge_size.
    if mms > 1 and reorder_fn is not None:
        idx = reorder_fn(idx, out.out_grid_thw, mms)
    return idx.to(torch.int64)
