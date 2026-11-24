"""Refiner q/k RMSNorm + RoPE in one Triton kernel (on by default; opt-out GSTAR_FAST_REFINER_KERNELS=0; other rounding).

Replaces the compiled q/k chain of the fast DiT block (inductor's kernel ran at ~2.3 TB/s): reads q and k
straight from the QKV GEMM output ([S, 3, H, D], row stride 3*H*D), normalizes each head row over D in fp32,
rounds to bf16 like F.rms_norm's output, applies rotate_half RoPE in fp32 and writes contiguous q, k.
"""
import os
from ...fast import enabled as _fast_on, sm90 as _sm90

import torch
import triton
import triton.language as tl


def dit_qkr_enabled():
    return _fast_on("REFINER_KERNELS")


@triton.jit
def _qk_norm_rope_kernel(qkv_ptr, nq_ptr, nk_ptr, cos_ptr, sin_ptr, q_out, k_out, n_rows, eps,
                         H: tl.constexpr, D: tl.constexpr, ROW_STRIDE: tl.constexpr, R: tl.constexpr):
    # rows r = s * H + h of q (and k); qkv: [S, 3, H, D] bf16 (row stride ROW_STRIDE = 3*H*D)
    HALF: tl.constexpr = D // 2
    pid = tl.program_id(0)
    r = pid * R + tl.arange(0, R)
    rm = r < n_rows
    s = r // H
    h = r % H
    d = tl.arange(0, HALF)
    base = s[:, None] * ROW_STRIDE + h[:, None] * D + d[None, :]
    c1 = tl.load(cos_ptr + s[:, None] * D + d[None, :], mask=rm[:, None], other=0.0).to(tl.float32)
    c2 = tl.load(cos_ptr + s[:, None] * D + HALF + d[None, :], mask=rm[:, None], other=0.0).to(tl.float32)
    s1 = tl.load(sin_ptr + s[:, None] * D + d[None, :], mask=rm[:, None], other=0.0).to(tl.float32)
    s2 = tl.load(sin_ptr + s[:, None] * D + HALF + d[None, :], mask=rm[:, None], other=0.0).to(tl.float32)
    o_base = r[:, None] * D + d[None, :]
    for which in tl.static_range(2):
        off = which * H * D
        x1 = tl.load(qkv_ptr + base + off, mask=rm[:, None], other=0.0).to(tl.float32)
        x2 = tl.load(qkv_ptr + base + off + HALF, mask=rm[:, None], other=0.0).to(tl.float32)
        if which == 0:
            w1 = tl.load(nq_ptr + d).to(tl.float32)
            w2 = tl.load(nq_ptr + HALF + d).to(tl.float32)
        else:
            w1 = tl.load(nk_ptr + d).to(tl.float32)
            w2 = tl.load(nk_ptr + HALF + d).to(tl.float32)
        ms = (tl.sum(x1 * x1, axis=1) + tl.sum(x2 * x2, axis=1)) / D
        rs = 1.0 / tl.sqrt(ms + eps)
        # F.rms_norm output (rounded to bf16 like the module's output), then RoPE in fp32
        n1 = (x1 * rs[:, None] * w1[None, :]).to(tl.bfloat16).to(tl.float32)
        n2 = (x2 * rs[:, None] * w2[None, :]).to(tl.bfloat16).to(tl.float32)
        o1 = n1 * c1 - n2 * s1
        o2 = n2 * c2 + n1 * s2
        optr = q_out if which == 0 else k_out
        tl.store(optr + o_base, o1.to(tl.bfloat16), mask=rm[:, None])
        tl.store(optr + o_base + HALF, o2.to(tl.bfloat16), mask=rm[:, None])

def qk_norm_rope(qkv, nq_w, nk_w, cos, sin, eps, H, D, R=4, warps=2):
    """qkv: [1, S, 3*H*D] bf16 (contiguous GEMM output); cos/sin: [1, S, D]. Returns q, k [S, H, D] contiguous."""
    S = qkv.shape[1]
    q = torch.empty(S, H, D, dtype=qkv.dtype, device=qkv.device)
    k = torch.empty_like(q)
    n = S * H
    _qk_norm_rope_kernel[(triton.cdiv(n, R),)](qkv, nq_w, nk_w, cos.contiguous(), sin.contiguous(), q, k, n, float(eps),
                                               H=H, D=D, ROW_STRIDE=3 * H * D, R=R, num_warps=warps)
    return q, k
