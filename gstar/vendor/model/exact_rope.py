"""Bit-exact fused RoPE for the decode step (on by default; opt-out GSTAR_FAST_LM_KERNELS=0).

Reproduces modeling_omni.apply_rotary_pos_emb for bf16 q/k and fp32 cos/sin tables in one kernel:

    c, s  = cos[pos], sin[pos]                                   (gathers, exact)
    q_out = ((q.float() * c) + (rotate_half(q.float()) * s)).to(bf16)
    k_out = ((k.float() * c) + (rotate_half(k.float()) * s)).to(bf16)

Each product and the sum are separate round-to-nearest fp32 operations in the eager chain
(three kernels), so the kernel uses explicit mul_rn / add_rn (no FMA contraction); the negation
of rotate_half is exact, the final cast is round-to-nearest-even like c10::BFloat16.
"""
import os
from ...fast import enabled as _fast_on, sm90 as _sm90

import torch

try:
    import triton
    import triton.language as tl
    from triton.language.extra.cuda import libdevice as _ld
    from .exact_ops import mul_rn as _xmul_rn, add_rn as _xadd_rn, sub_rn as _xsub_rn
    _TRITON = True
except Exception:  # pragma: no cover
    _TRITON = False


def fused_rope_enabled():
    return _TRITON and _fast_on("LM_KERNELS")


if _TRITON:
    @triton.jit
    def _rope_kernel(q_ptr, k_ptr, cos_ptr, sin_ptr, pos_ptr, qo_ptr, ko_ptr,
                     sqb, sqh, sqs, skb, skh, sks, ps_b, ps_s,
                     HQ: tl.constexpr, HK: tl.constexpr, S: tl.constexpr, D: tl.constexpr):
        pid = tl.program_id(0)              # one program per (batch, position)
        b = pid // S
        sp = pid % S
        pos = tl.load(pos_ptr + b * ps_b + sp * ps_s)
        half: tl.constexpr = D // 2
        d = tl.arange(0, D)
        rot = tl.where(d < half, d + half, d - half)
        c = tl.load(cos_ptr + pos * D + d)
        s = tl.load(sin_ptr + pos * D + d)
        hq = tl.arange(0, HQ)
        qp = q_ptr + b * sqb + sp * sqs + hq[:, None] * sqh
        x = tl.load(qp + d[None, :]).to(tl.float32)
        xr = tl.load(qp + rot[None, :]).to(tl.float32)
        xr = tl.where(d[None, :] < half, -xr, xr)
        out = _xadd_rn(_xmul_rn(x, c[None, :]), _xmul_rn(xr, s[None, :]))
        tl.store(qo_ptr + ((b * S + sp) * HQ + hq[:, None]) * D + d[None, :], out.to(tl.bfloat16))
        hk = tl.arange(0, HK)
        kp = k_ptr + b * skb + sp * sks + hk[:, None] * skh
        x = tl.load(kp + d[None, :]).to(tl.float32)
        xr = tl.load(kp + rot[None, :]).to(tl.float32)
        xr = tl.where(d[None, :] < half, -xr, xr)
        out = _xadd_rn(_xmul_rn(x, c[None, :]), _xmul_rn(xr, s[None, :]))
        tl.store(ko_ptr + ((b * S + sp) * HK + hk[:, None]) * D + d[None, :], out.to(tl.bfloat16))


def _pow2(n):
    return n > 0 and (n & (n - 1)) == 0


def rope_supported(q, k, cos_, sin_, position_ids):
    if not (_TRITON and q.is_cuda and q.dtype == torch.bfloat16 and k.dtype == torch.bfloat16
            and cos_.dtype == torch.float32 and sin_.dtype == torch.float32):
        return False
    if q.dim() != 4 or k.dim() != 4 or q.shape[0] != k.shape[0] or q.shape[2] != k.shape[2]:
        return False
    B, HQ, S, D = q.shape
    if not (_pow2(D) and _pow2(HQ) and _pow2(k.shape[1]) and q.stride(-1) == 1 and k.stride(-1) == 1):
        return False
    if position_ids.shape != (B, S) or cos_.numel() % D or not cos_.is_contiguous() or not sin_.is_contiguous():
        return False
    return True


def fused_rope(q, k, cos_, sin_, position_ids):
    """Same outputs as apply_rotary_pos_emb(q, k, cos_, sin_, position_ids): (q_embed, k_embed) with the
    shapes of q and k ([B, H, S, D]); outputs are contiguous in [B, S, H, D] order viewed as [B, H, S, D]."""
    B, HQ, S, D = q.shape
    HK = k.shape[1]
    qo = torch.empty(B, S, HQ, D, dtype=q.dtype, device=q.device)
    ko = torch.empty(B, S, HK, D, dtype=k.dtype, device=k.device)
    _rope_kernel[(B * S,)](q, k, cos_, sin_, position_ids, qo, ko,
                           q.stride(0), q.stride(1), q.stride(2), k.stride(0), k.stride(1), k.stride(2),
                           position_ids.stride(0), position_ids.stride(1),
                           HQ=HQ, HK=HK, S=S, D=D, num_warps=4)
    if S == 1:   # [B, 1, H, D] memory == contiguous [B, H, 1, D]: the reference's strides
        return qo.view(B, HQ, 1, D), ko.view(B, HK, 1, D)
    return qo.transpose(1, 2), ko.transpose(1, 2)
