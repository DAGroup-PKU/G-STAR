"""Bit-exact cheaper decode attention for the CFG decode step (on by default; opt-out GSTAR_FAST_LM_KERNELS=0).

Reference (Attention.forward with use_flash=False, torch 2.6 math SDPA, bf16 inputs):
    k_rep, v_rep = repeat_kv(K[:, :, :L]), repeat_kv(V[:, :, :L])          # bf16 copies [B, H, L, D]
    q_acc, k_acc, v_acc = q.float(), k_rep.float(), v_rep.float()           # fp32 copies
    s = sqrt(1 / sqrt(D)) (double, then float in the kernels)
    attn = matmul(q_acc * s, k_acc.transpose(-2, -1) * s)                   # cuBLAS fp32
    attn.add_(mask); attn = _safe_softmax(attn, -1)
    out = matmul(attn, v_acc).to(bf16); (attn.to(bf16) is computed and discarded)
Here the five copies and two scalings become two Triton kernels that write exactly the fp32 tensors
k_acc * s (same contiguous [B, H, L, D] memory, viewed transposed: the strides the reference's mul
produces) and v_acc, straight from the bf16 cache (head h reads kv head h // n_rep); q_acc * s is a third
tiny kernel.  The matmuls, the mask add and the softmax are the same ATen calls on identical tensors.
"""
import math
import os
from ...fast import enabled as _fast_on, sm90 as _sm90

import numpy as np
import torch

try:
    import triton
    import triton.language as tl
    from triton.language.extra.cuda import libdevice as _ld
    from .exact_ops import mul_rn as _xmul_rn, add_rn as _xadd_rn, sub_rn as _xsub_rn
    _TRITON = True
except Exception:  # pragma: no cover
    _TRITON = False


def fast_decode_attn_enabled():
    return _TRITON and _fast_on("LM_KERNELS")


def sdpa_math_scale(head_dim):
    # sdp::calculate_scale(q, None).sqrt(): SymFloat (double) 1 / sqrt(D), then sqrt; float in the kernel
    return float(np.float32(math.sqrt(1.0 / math.sqrt(head_dim))))


if _TRITON:
    @triton.jit
    def _expand_f32_kernel(src, dst, scale, L, s_b, s_h, s_l, HQ: tl.constexpr, REP: tl.constexpr,
                           D: tl.constexpr, BLOCK_L: tl.constexpr, SCALE: tl.constexpr):
        bh = tl.program_id(0)
        lb = tl.program_id(1)
        b = bh // HQ
        h = bh % HQ
        rows = lb * BLOCK_L + tl.arange(0, BLOCK_L)
        d = tl.arange(0, D)
        m = rows[:, None] < L
        x = tl.load(src + b * s_b + (h // REP) * s_h + rows[:, None] * s_l + d[None, :], mask=m).to(tl.float32)
        if SCALE:
            x = _xmul_rn(x, scale)
        tl.store(dst + (bh * L + rows[:, None]) * D + d[None, :], x, mask=m)

    @triton.jit
    def _q_scale_kernel(src, dst, scale, s_b, s_h, HQ: tl.constexpr, D: tl.constexpr):
        b = tl.program_id(0)
        h = tl.arange(0, HQ)
        d = tl.arange(0, D)
        x = tl.load(src + b * s_b + h[:, None] * s_h + d[None, :]).to(tl.float32)
        tl.store(dst + (b * HQ + h[:, None]) * D + d[None, :], _xmul_rn(x, scale))


def _expand(buf, L, hq, scale):
    B, HKV, _, D = buf.shape
    out = torch.empty(B, hq, L, D, dtype=torch.float32, device=buf.device)
    BLOCK_L = 64
    _expand_f32_kernel[(B * hq, triton.cdiv(L, BLOCK_L))](
        buf, out, scale if scale is not None else 1.0, L, buf.stride(0), buf.stride(1), buf.stride(2),
        HQ=hq, REP=hq // HKV, D=D, BLOCK_L=BLOCK_L, SCALE=scale is not None, num_warps=4)
    return out


def supported(q, kbuf):
    if not _TRITON or q.dtype != torch.bfloat16 or kbuf.dtype != torch.bfloat16 or not q.is_cuda:
        return False
    B, HQ, S, D = q.shape
    return (S == 1 and kbuf.shape[0] == B and HQ % kbuf.shape[1] == 0 and kbuf.shape[3] == D
            and (HQ & (HQ - 1)) == 0 and (D & (D - 1)) == 0 and q.stride(-1) == 1 and kbuf.stride(-1) == 1)


def decode_attention(q, kbuf, vbuf, L, mask):
    """q: [B, HQ, 1, D] bf16; kbuf/vbuf: [B, HKV, Lcap, D] bf16 caches (first L rows valid);
    mask: additive mask broadcastable to [B, HQ, 1, L].  Returns what
    F.scaled_dot_product_attention(q, repeat_kv(k), repeat_kv(v), attn_mask=mask) returns (math backend)."""
    B, HQ, _, D = q.shape
    s = sdpa_math_scale(D)
    q_s = torch.empty(B, HQ, 1, D, dtype=torch.float32, device=q.device)
    _q_scale_kernel[(B,)](q, q_s, s, q.stride(0), q.stride(1), HQ=HQ, D=D, num_warps=4)
    k_s = _expand(kbuf, L, HQ, s)                 # == k_acc * s, memory of the reference's transposed mul
    v_acc = _expand(vbuf, L, HQ, None)            # == v_acc
    attn = torch.matmul(q_s, k_s.transpose(-2, -1))
    attn.add_(mask)
    attn = torch.ops.aten._safe_softmax(attn, -1)
    return torch.matmul(attn, v_acc).to(q.dtype)


# ---------------------------------------------------------------------------------------------
# Lean variant: persistent fp32 caches k_acc*s and v_acc (all query heads),
# one row appended per decode step inside the pre-attention graph, so the per-step expansion of the
# whole cache disappears.  cuBLAS on the [:, :, :L] views of these caches gives the same results as
# on contiguous copies (checked for every L in 18..1120); softmax replaces _safe_softmax, identical
# because no row is fully masked (the current token is always attended).
# ---------------------------------------------------------------------------------------------
def lm_lean_enabled():
    return _TRITON and _fast_on("LM_KERNELS")


if _TRITON:
    @triton.jit
    def _expand_into_kernel(src, dst, scale, L, s_b, s_h, s_l, d_b, d_h, d_l, HQ: tl.constexpr,
                            REP: tl.constexpr, D: tl.constexpr, BLOCK_L: tl.constexpr, SCALE: tl.constexpr):
        bh = tl.program_id(0)
        lb = tl.program_id(1)
        b = bh // HQ
        h = bh % HQ
        rows = lb * BLOCK_L + tl.arange(0, BLOCK_L)
        d = tl.arange(0, D)
        m = rows[:, None] < L
        x = tl.load(src + b * s_b + (h // REP) * s_h + rows[:, None] * s_l + d[None, :], mask=m).to(tl.float32)
        if SCALE:
            x = _xmul_rn(x, scale)
        tl.store(dst + b * d_b + h * d_h + rows[:, None] * d_l + d[None, :], x, mask=m)

    @triton.jit
    def _step_rows_kernel(q, k, v, qs_out, ks_buf, vf_buf, idx_ptr, scale,
                          q_sb, q_sh, k_sb, k_sh, v_sb, v_sh, c_b, c_h, c_l,
                          HQ: tl.constexpr, REP: tl.constexpr, D: tl.constexpr):
        b = tl.program_id(0)
        h = tl.arange(0, HQ)
        d = tl.arange(0, D)
        row = tl.load(idx_ptr)
        xq = tl.load(q + b * q_sb + h[:, None] * q_sh + d[None, :]).to(tl.float32)
        tl.store(qs_out + (b * HQ + h[:, None]) * D + d[None, :], _xmul_rn(xq, scale))
        xk = tl.load(k + b * k_sb + (h[:, None] // REP) * k_sh + d[None, :]).to(tl.float32)
        tl.store(ks_buf + b * c_b + h[:, None] * c_h + row * c_l + d[None, :], _xmul_rn(xk, scale))
        xv = tl.load(v + b * v_sb + (h[:, None] // REP) * v_sh + d[None, :]).to(tl.float32)
        tl.store(vf_buf + b * c_b + h[:, None] * c_h + row * c_l + d[None, :], xv)


def expand_into(src, dst, L, scale):
    """dst[:, :, :L] = float(repeat_kv(src[:, :, :L])) (* scale); dst is [B, HQ, Lcap, D] fp32."""
    B, HKV, _, D = src.shape
    HQ = dst.shape[1]
    BLOCK_L = 64
    _expand_into_kernel[(B * HQ, triton.cdiv(L, BLOCK_L))](
        src, dst, scale if scale is not None else 1.0, L, src.stride(0), src.stride(1), src.stride(2),
        dst.stride(0), dst.stride(1), dst.stride(2), HQ=HQ, REP=HQ // HKV, D=D, BLOCK_L=BLOCK_L,
        SCALE=scale is not None, num_warps=4)


def step_rows(q, k, v, ks_buf, vf_buf, idx):
    """q_s = float(q) * s (new tensor [B, HQ, 1, D]); rows idx of ks_buf / vf_buf = float(k) * s / float(v)
    for every query head (head h reads kv head h // rep).  q: [B, HQ, 1, D]; k, v: [B, HKV, 1, D] bf16."""
    B, HQ, _, D = q.shape
    s = sdpa_math_scale(D)
    q_s = torch.empty(B, HQ, 1, D, dtype=torch.float32, device=q.device)
    _step_rows_kernel[(B,)](q, k, v, q_s, ks_buf, vf_buf, idx, s,
                            q.stride(0), q.stride(1), k.stride(0), k.stride(1), v.stride(0), v.stride(1),
                            ks_buf.stride(0), ks_buf.stride(1), ks_buf.stride(2),
                            HQ=HQ, REP=HQ // k.shape[1], D=D, num_warps=4)
    return q_s


def decode_attention_cached(q_s, ks_buf, vf_buf, L, mask, out_dtype):
    """The math-SDPA result from q_s = q_acc * s and the persistent caches (first L rows)."""
    attn = torch.matmul(q_s, ks_buf[:, :, :L].transpose(-2, -1))
    attn.add_(mask)
    attn = torch.softmax(attn, -1)
    return torch.matmul(attn, vf_buf[:, :, :L]).to(out_dtype)
