"""Shared building blocks for RepAR-conditioned DiTs (Krea2 / others).

Unified *zero-init additive* conditioning:

    z_q (RepAR feature) ──► ConditionEncoder ──► cond_tokens (B, N, C_cond)
                                                     │  (spatial grid aligned
                                                     │   to the DiT image-token
                                                     │   grid, done ONCE and
                                                     │   shared by all inject pts)
                                                     ▼
    per inject point:  InjectionHead(cond_tokens, t) ──► residual (B, N, D)
                       (adaLN time-modulated + zero-init Linear)  ★ step0 = 0

`only_first_layer_inject` on the wrapper decides whether there is ONE inject
point (DiT input, ≈ SketchColour channel-concat) or K points every N layers
(≈ multi-layer ControlNet). No sigma gate — the residual is forced additive.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    """Standard RMSNorm (weight init 1). Independent of krea2_dit.RMSNorm."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x * self.weight.float()).to(dtype)


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1 + scale) + shift


# ---- fused condition-path pieces (opt-out GSTAR_FAST_REFINER=0; same function, other rounding) ----
import os as _os
from ...fast import enabled as _fast_on, sm90 as _sm90


def dit_fast2_enabled():
    return _fast_on("REFINER")


def _inj_rms(x, nw, eps: float, shift, scale, lw, lb):
    xf = x.float()
    xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    xn = (xf * nw.float()).to(x.dtype)
    return F.linear(modulate(xn, shift, scale), lw, lb)


def _inj_ln(x, eps: float, shift, scale, lw, lb):
    xn = F.layer_norm(x, (x.shape[-1],), eps=eps)
    return F.linear(modulate(xn, shift, scale), lw, lb)


def _resblock(x, eps: float, shift, scale, gate, w1, b1, w2, b2):
    h = modulate(F.layer_norm(x, (x.shape[-1],), eps=eps), shift, scale)
    h = F.linear(F.silu(F.linear(h, w1, b1)), w2, b2)
    return x + gate * h


def _film_fp32(h, s, b, g):
    """FiLM in fp32 on the (fp32) resized code, rounded once to the module dtype (that of s)."""
    return ((h * (1 + s.float()) + b.float()) * torch.sigmoid(g.float())).to(s.dtype)


_C2 = {}


def _c2(name, fn):
    f = _C2.get(name)
    if f is None:
        if _os.environ.get("GSTAR_FAST_REFINER_AUTOTUNE", "0") == "1":
            f = torch.compile(fn, dynamic=False, fullgraph=True,
                              options={"max_autotune": True, "max_autotune_gemm_backends": "TRITON,ATEN",
                                       "triton.enable_persistent_tma_matmul": True})
        else:
            f = torch.compile(fn, dynamic=False, fullgraph=True)
        _C2[name] = f
    return f


class ResBlock(nn.Module):
    """DiT-style time-modulated residual MLP (adopted from UNet-diffusion ResBlock).

    x (B, N, ch) + time y (B, time_dim) or (B, N, time_dim) → x + gate·MLP(...).
    adaLN takes the time embedding directly (time_dim), not `ch`, so the trunk's
    time vector (the DiT `features` dim) can drive it without a pre-projection.
    adaLN last layer is zero-init ⇒ gate=0 at start ⇒ the block is IDENTITY, so
    adding it never perturbs a healthy init; it learns its refinement from there.
    """

    def __init__(self, channels: int, time_dim: int):
        super().__init__()
        self.in_ln = nn.LayerNorm(channels, elementwise_affine=False, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels, bias=True),
            nn.SiLU(),
            nn.Linear(channels, channels, bias=True),
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(time_dim, 3 * channels, bias=True))
        nn.init.zeros_(self.adaLN_modulation[1].weight)
        nn.init.zeros_(self.adaLN_modulation[1].bias)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        shift, scale, gate = self.adaLN_modulation(y).chunk(3, dim=-1)
        if shift.dim() == 2:
            shift = shift.unsqueeze(1); scale = scale.unsqueeze(1); gate = gate.unsqueeze(1)
        if dit_fast2_enabled() and not torch.is_grad_enabled() and isinstance(self.mlp[1], nn.SiLU):
            return _c2("resblock", _resblock)(x, self.in_ln.eps, shift, scale, gate, self.mlp[0].weight,
                                               self.mlp[0].bias, self.mlp[2].weight, self.mlp[2].bias)
        h = modulate(self.in_ln(x), shift, scale)
        h = self.mlp(h)
        return x + gate * h


class ConditionEncoder(nn.Module):
    """z_q (B, L_z, code_dim) → cond tokens (B, tH*tW, cond_width), TIME-SELECTIVE.

    A 3-stage shared trunk (ControlNet-style: shared feature extractor, per-inject
    zero-init out head lives in `InjectionHead`):

        h = RMSNorm(z_q)                              # full code_dim (rich RepAR code)
        s, b, g = time_mlp(t_emb)                     # each code_dim
        h = (h*(1+s) + b) * sigmoid(g)                # ① time-selective FiLM+SE gate
        h = interpolate(h grid → DiT grid)            #    align to DiT image-token grid
        h = in_proj(code_dim → cond_width)            # ② projection (time-picked feats)
        h = ResBlock(h, t_emb)                        # ③ time-modulated residual refine
        → cond tokens (B, N, cond_width)

    Stage ① lets time *pick which of the code_dim channels matter at this noise
    level* BEFORE the projection — effective proj `W·diag(g_t)` is a time-conditioned
    subspace, not a fixed one. `cond_width` is deliberately NOT a tiny bottleneck:
    z_q is a rich learned reconstruction code injected into a wide DiT backbone, so
    hard-compressing it (e.g. →64) throttles detail for no param saving. Default it
    to `code_dim` (no compression); shrink only if you *want* a bottleneck.

    Init: `time_mlp` last layer zero + gate bias large ⇒ stage ① ≈ passthrough at
    step-0; `ResBlock` adaLN zero-init ⇒ stage ③ ≈ identity. Step-0 identity to the
    frozen base is guaranteed independently by the zero-init Linear in `InjectionHead`.
    """

    def __init__(self, code_dim: int, cond_width: int, time_dim: int,
                 num_resblocks: int = 1, resize_mode: str = "bilinear"):
        super().__init__()
        self.code_dim = code_dim
        self.cond_width = cond_width
        self.resize_mode = resize_mode
        self.norm = RMSNorm(code_dim)
        self.time_mlp = nn.Sequential(
            nn.SiLU(), nn.Linear(time_dim, 3 * code_dim, bias=True))
        self.in_proj = nn.Linear(code_dim, cond_width)
        self.resblocks = nn.ModuleList(
            [ResBlock(cond_width, time_dim) for _ in range(max(1, num_resblocks))])
        # Start ≈ time-agnostic passthrough: no scale/shift, gate open (~1).
        nn.init.zeros_(self.time_mlp[1].weight)
        nn.init.zeros_(self.time_mlp[1].bias)
        with torch.no_grad():
            self.time_mlp[1].bias[2 * code_dim:].fill_(4.0)   # sigmoid(4)≈0.982

    @staticmethod
    def _film_apply(h, s, b, g):
        s = s.unsqueeze(1); b = b.unsqueeze(1); g = g.unsqueeze(1)   # (B,1,code_dim)
        return (h * (1 + s) + b) * torch.sigmoid(g)

    def film(self, h: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        s, b, g = self.time_mlp(t_emb).chunk(3, dim=-1)        # each (B, code_dim)
        return self._film_apply(h, s, b, g)

    def refine(self, h: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """in_proj → ResBlock stack. `y` is the per-token (or per-sample) time
        embedding (time_dim). Exposed so the packed path can call it ONCE over the
        concatenated cond tokens (ZeRO-3 call-count invariant)."""
        h = self.in_proj(h)
        for rb in self.resblocks:
            h = rb(h, y)
        return h

    def forward(self, z_q: torch.Tensor,
                z_grid: tuple[int, int],
                target_grid: tuple[int, int],
                t_emb: torch.Tensor) -> torch.Tensor:
        B, L, C = z_q.shape
        zH, zW = int(z_grid[0]), int(z_grid[1])
        tH, tW = int(target_grid[0]), int(target_grid[1])
        assert zH * zW == L, f"z_grid {zH}x{zW} != L_z {L}"
        h = self.norm(z_q)                                     # (B, L, C)
        h = self.film(h, t_emb)                                # ① time-selective (B, L, C)
        h = h.transpose(1, 2).reshape(B, C, zH, zW)            # (B, C, zH, zW)
        if (zH, zW) != (tH, tW):
            align = None if self.resize_mode == "nearest" else False
            h = F.interpolate(h.float(), size=(tH, tW),
                              mode=self.resize_mode, align_corners=align).to(z_q.dtype)
        h = h.reshape(B, C, tH * tW).transpose(1, 2)           # (B, N, C)
        return self.refine(h, t_emb)                           # ②③ → (B, N, cond_width)


class InjectionHead(nn.Module):
    """adaLN(time) + zero-init Linear → additive residual for one inject point.

    cond_tokens (B, N, cond_dim) + time c (B, time_dim) or (B, N, time_dim)
    → (B, N, out_dim). `linear` is zero-init so the residual is 0 at step-0
    (identity to the frozen base regardless of adaLN init).
    """

    def __init__(self, cond_dim: int, time_dim: int, out_dim: int,
                 use_rmsnorm: bool = False):
        super().__init__()
        self.norm_final = (RMSNorm(cond_dim) if use_rmsnorm
                           else nn.LayerNorm(cond_dim, elementwise_affine=False, eps=1e-6))
        self.linear = nn.Linear(cond_dim, out_dim, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(time_dim, 2 * cond_dim, bias=True))
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        mod = self.adaLN_modulation(c)                         # (B,2cond) or (B,N,2cond)
        shift, scale = mod.chunk(2, dim=-1)
        if shift.dim() == 2:
            shift = shift.unsqueeze(1)
            scale = scale.unsqueeze(1)
        if dit_fast2_enabled() and not torch.is_grad_enabled():
            if isinstance(self.norm_final, RMSNorm):
                return _c2("inj_rms", _inj_rms)(x, self.norm_final.weight, self.norm_final.eps, shift, scale,
                                                self.linear.weight, self.linear.bias)
            if isinstance(self.norm_final, nn.LayerNorm) and self.norm_final.weight is None:
                return _c2("inj_ln", _inj_ln)(x, self.norm_final.eps, shift, scale,
                                              self.linear.weight, self.linear.bias)
        x = modulate(self.norm_final(x), shift, scale)
        return self.linear(x)


# ===========================================================================
# In-Context adapter: RepAR z_q tokens → pseudo-text context tokens
# ===========================================================================
#
# Turns the frozen RepAR feature z_q (B, L, code_dim) into "pseudo-text" context
# tokens of dim `features` (D) that ride the Krea2 DiT's PRETRAINED joint-
# attention conditioning route — fed in place of the empty prompt. This gives
# the condition a strong, immediately-effective control path (unlike the zero-
# init additive spatial injection, which starts at 0 and needs a long ramp).
#
# Token count is PRESERVED (L_ctx == L_z), so the caller sizes pos/mask by z_q's
# length. Packed training runs all samples' z_q as ONE concatenated sequence
# with a BLOCK-DIAGONAL attention mask (built from per-image lengths) so images
# never attend across each other, while every parameterized op is called exactly
# ONCE regardless of pack size n — required for DeepSpeed ZeRO-3 (per-sample
# param calls with n varying across ranks would mismatch collectives → deadlock).
class _AdapterBlock(nn.Module):
    """Pre-norm transformer block: self-attention (optional block-diagonal mask)
    + MLP. No positional encoding — the DiT gives context tokens position 0, and
    the additive spatial-injection path handles pixel alignment; this block is a
    semantic/content re-organizer of z_q."""

    def __init__(self, dim: int, heads: int, mlp_ratio: int = 4):
        super().__init__()
        assert dim % heads == 0, f"dim {dim} not divisible by heads {heads}"
        self.heads = heads
        self.norm1 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio), nn.GELU(),
            nn.Linear(dim * mlp_ratio, dim))

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor = None) -> torch.Tensor:
        B, S, D = x.shape
        h = self.norm1(x)
        qkv = self.qkv(h).reshape(B, S, 3, self.heads, D // self.heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)        # each (B, heads, S, hd)
        o = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        o = o.transpose(1, 2).reshape(B, S, D)
        x = x + self.proj(o)
        x = x + self.mlp(self.norm2(x))
        return x


class IncontextAdapter(nn.Module):
    """z_q (B, L, code_dim) → context tokens (B, L, features). Token count kept.

    Bottleneck design: attention/MLP blocks run at `inner_dim` (≪ features), with
    `in_proj` (code_dim→inner) and `out_proj` (inner→features) at the ends. Keeps
    the param/activation cost far below running the blocks at the full DiT width
    `features` (6144), while still mixing tokens via real self-attention."""

    def __init__(self, code_dim: int, features: int, depth: int = 2,
                 heads: int = 8, mlp_ratio: int = 4, inner_dim: int = 2048):
        super().__init__()
        self.in_proj = nn.Linear(code_dim, inner_dim)
        self.blocks = nn.ModuleList(
            [_AdapterBlock(inner_dim, heads, mlp_ratio) for _ in range(depth)])
        self.norm_out = nn.LayerNorm(inner_dim)
        self.out_proj = nn.Linear(inner_dim, features)
        # start the output mild so early context isn't wildly OOD for the DiT's
        # pretrained cross-attention (it still has non-zero effect via the
        # residual path — NOT zero-init, we WANT immediate control).
        nn.init.zeros_(self.out_proj.bias)
        nn.init.normal_(self.out_proj.weight, std=0.02)

    @staticmethod
    def _block_diag_mask(seqlens, device) -> torch.Tensor:
        """(S, S) bool mask, True where two tokens share the same image block."""
        S = int(sum(seqlens))
        m = torch.zeros(S, S, dtype=torch.bool, device=device)
        off = 0
        for L in seqlens:
            L = int(L)
            m[off:off + L, off:off + L] = True
            off += L
        return m

    def forward(self, z_cat: torch.Tensor, seqlens=None) -> torch.Tensor:
        """z_cat: (1, S, code_dim) — ONE sample (dense) or all packed samples
        concatenated. `seqlens`: per-image token counts for the packed block-
        diagonal mask; None or a single-element list → full attention."""
        h = self.in_proj(z_cat)
        attn_mask = None
        if seqlens is not None and len(seqlens) > 1:
            attn_mask = self._block_diag_mask(seqlens, z_cat.device)[None, None]  # (1,1,S,S)
        for blk in self.blocks:
            h = blk(h, attn_mask=attn_mask)
        return self.out_proj(self.norm_out(h))
