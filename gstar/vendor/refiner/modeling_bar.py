"""HF PreTrainedModel exposing the vendored BAR tokenizer under the
``BaseVisualTokenizer`` contract (encode -> EncodeOutput(indices, out_grid_thw),
decode_to_pixels, decode_to_features, geometry properties).

Loadable via AutoModel.from_pretrained(dir, trust_remote_code=True). Self-contained:
it defines its own ``EncodeOutput`` (duck-typed, matching the LLM-side contract)
and does NOT import any external tokenizer package.
"""
from dataclasses import dataclass
from typing import Optional, List, Dict

import torch
from omegaconf import OmegaConf
from transformers import PreTrainedModel
from transformers.modeling_outputs import ModelOutput

from .configuration_bar import BARTokenizerConfig
# NOTE: HF's dynamic-module loader only copies the FIRST-LEVEL relative imports
# of THIS file into its cache (it does not recurse). So every vendored module
# must be imported here directly, even if unused, or the transitive imports
# inside tokenizer.py (blocks/quantizer/base_model/res_mod) won't be present.
from .base_model import BaseModel  # noqa: F401  (force-copy into HF cache)
from .res_mod import ResMod  # noqa: F401  (force-copy into HF cache)
from .blocks import UnifiedEncoder, UnifiedDecoder  # noqa: F401  (force-copy)
from .quantizer import BaseVQ, FSQ, VQ, RVQ, ProductVQ  # noqa: F401  (force-copy)
from .tokenizer import (
    FSQTokenizer,
    RVQTokenizer,
    VQTokenizer,
    AETokenizer,
    AliTokTokenizer,
)

_REGISTRY = {
    "FSQ": FSQTokenizer,
    "RVQ": RVQTokenizer,
    "pq": RVQTokenizer,
    "PQ": RVQTokenizer,
    "VQ": VQTokenizer,
    "AE": AETokenizer,
    "AliTok": AliTokTokenizer,
}


@dataclass
class EncodeOutput(ModelOutput):
    """Mirror of the LLM-side EncodeOutput contract (duck-typed, self-contained)."""
    indices: Optional[torch.LongTensor] = None       # [N_tokens, num_levels], raster per out_grid_thw
    out_grid_thw: Optional[torch.LongTensor] = None   # [num_images, 3], quantized-token grid
    quants: Optional[torch.FloatTensor] = None        # z_quantized_merged (out_proj / decoder space)
    codebook_quant: Optional[torch.FloatTensor] = None  # full-depth cumulative code in CODEBOOK space
    cmt_loss: Optional[torch.Tensor] = None
    codebook_usage: Optional[Dict] = None


class BARTokenizer(PreTrainedModel):
    """BAR RVQ tokenizer as a BaseVisualTokenizer-compatible model.

    Contract methods (duck-typed against the visual tokenizer interface):
    encode / decode_to_pixels / decode_to_features + codebook_sizes /
    num_codebook_levels / codebook_dim. get_uncond_indices / fake_input are optional.
    """

    config_class = BARTokenizerConfig

    # transformers>=5.x reads `self.all_tied_weights_keys` during
    # `from_pretrained`'s meta→device move. This tokenizer ties no weights, so a
    # class-level empty mapping is a safe compat shim (no numerical effect).
    all_tied_weights_keys: dict = {}

    def __init__(self, config):
        super().__init__(config)
        oc = OmegaConf.create(config.bar_config)
        self._oc = oc
        vq_type = oc.model.vq_model.get("type", "FSQ")
        if vq_type not in _REGISTRY:
            raise ValueError(f"Unknown tokenizer type {vq_type!r}. Supported: {list(_REGISTRY)}")
        self.tokenizer = _REGISTRY[vq_type](oc)

    # ---- geometry (LLM builds bridge/head from these) -----------------------
    @property
    def num_codebook_levels(self) -> int:
        return int(self.tokenizer.quantize.code_depth)

    @property
    def codebook_sizes(self) -> List[int]:
        return [int(self.tokenizer.quantize.codebook_size)] * self.num_codebook_levels

    @property
    def codebook_dim(self) -> int:
        return int(self._oc.model.vq_model.get("code_dim"))

    # ---- contract: encode ---------------------------------------------------
    def encode(self, pixel_values, grid_thw, fake_input: bool = False) -> EncodeOutput:
        """pixel_values: Qwen flatten patches [ΣN_patch, patch_dim] (or a list of
        per-image CHW tensors); grid_thw: [num_images, 3] (t,h,w) input patch grid.
        Returns discrete indices [N_tokens, depth] laid out raster per out_grid_thw,
        where out_grid_thw = grid_thw with h,w divided by the internal merge size."""
        if not torch.is_tensor(grid_thw):
            grid_thw = torch.as_tensor(grid_thw, dtype=torch.long)
        _, result = self.tokenizer.encode(pixel_values, image_grid_thw=grid_thw)
        indices = result["min_encoding_indices"]        # (N_merged, depth) long
        if indices.dim() == 1:                           # single-depth (VQ) -> add level axis
            indices = indices.unsqueeze(-1)
        out_grid_thw = self.tokenizer._merged_grid_thw(grid_thw).to(indices.device)
        # Full-depth cumulative quantized code in CODEBOOK space (pre out_proj):
        # `quant_at_depth_with_grad[-1]` = Σ_{d} codebook_d[code_d], dim=code_dim
        # (1152). This is the discrete-feature condition RepAR-HiDream consumes
        # — distinct from `quants` (= z_quantized_merged, the out_proj/decoder
        # space, dim=1024).
        qad = result.get("quant_at_depth_with_grad", None)
        codebook_quant = qad[-1] if qad else None
        return EncodeOutput(
            indices=indices.long(),
            out_grid_thw=out_grid_thw.long(),
            quants=result.get("z_quantized_merged", None),
            codebook_quant=codebook_quant,
        )

    # ---- contract: decode_to_pixels ----------------------------------------
    def decode_to_pixels(self, indices, grid_h, grid_w, target_height, target_width):
        """indices: raster [B, grid_h, grid_w, depth] code ids -> pixels [B,3,H,W].
        Returns None if this tokenizer has no pixel decoder (feature_recon_only)."""
        tok = self.tokenizer
        if getattr(tok, "decoder", None) is None:
            return None
        idx = indices.reshape(indices.shape[0], grid_h * grid_w, indices.shape[-1])
        B = idx.shape[0]
        z = tok.quantize.get_codebook_entry(idx).reshape(B, grid_h * grid_w, tok.decoder.width)
        if getattr(tok, "merger_after_quant", False):
            z = tok.post_quant_unmerger(z)               # -> [B, grid_h*grid_w*ms^2, C]
        # spatial_merge_size is None when merger_after_quant is off (the 16x /
        # no-merger variant: quant tokens already sit at the decoder's patch grid,
        # so no unmerge and ms=1). ``int(None)`` would crash -> coerce to 1.
        ms = int(getattr(tok, "spatial_merge_size", None) or 1)
        uh, uw = grid_h * ms, grid_w * ms
        gs = getattr(tok.decoder, "grid_size", None)
        if gs is not None and uh == gs and uw == gs:     # fixed square -> static decode
            imgs = tok.decode(z)[0]
        else:                                            # anyres -> varlen decode
            g = torch.tensor([[1, uh, uw]] * B, device=z.device, dtype=torch.long)
            out = tok.decode(z.reshape(B * uh * uw, z.shape[-1]), image_grid_thw=g)[0]
            imgs = torch.stack(list(out), dim=0) if isinstance(out, (list, tuple)) else out
        return imgs

    # ---- contract: decode_to_features (feature-distill; usually unused) -----
    def decode_to_features(self, quants, grid_thw):
        return None

    # ---- optional contract methods -----------------------------------------
    def get_uncond_indices(self, num_images, out_grid_thw, device):
        raise NotImplementedError("BARTokenizer does not provide CFG null indices.")

    def fake_input(self, device):
        raise NotImplementedError("BARTokenizer does not provide a ZeRO3 fake input.")
