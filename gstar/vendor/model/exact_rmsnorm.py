"""Bit-exact fused RMSNorm (on by default; opt-out GSTAR_FAST_LM_KERNELS=0).

Reproduces, in one Triton kernel, the eager chain of audio_modeling_omni.RMSNorm for bf16 inputs:

    var = x.float().pow(2).mean(-1, keepdim=True)        # ATen reduce_kernel (MeanOps)
    h   = (x * torch.rsqrt(var + eps)).to(bf16)           # fp32 math, RNE to bf16
    out = weight * h                                      # bf16 x bf16 -> fp32 -> RNE bf16

The only order-sensitive part is the mean.  ATen reduces a contiguous last dim of N fp32 values with
T threads per row (ReduceConfig::set_block_dimension, Reduce.cuh):
  * N > 128 ("vectorized input", 4 lanes): thread t keeps 4 accumulators, acc_i += x[4(t + T j) + i]
    for j = 0, 1, ...; then ((a0 + a1) + a2) + a3;
  * N <= 128 (vt0 = 4 unrolled loads): acc_i += x[t + T i + 4T m]; then ((a0 + a1) + a2) + a3;
  * block_x_reduce: shared-memory halving v[t] += v[t + off] for off = T/2 ... 32, then warp shuffles
    v[t] += v[t + off] for off = 1, 2, 4, 8, 16 (lane 0 holds the result);
  * mean = sum * (rows / numel), a power of two here (exact).
T depends on N and on the number of rows R exactly as in set_block_dimension.  Every addition here is a
two-operand fp32 add in the same tree, so the result is bitwise identical.  A one-time live check per
(N, R) against the eager module disables the fused path for that shape if it ever differs.
"""
import os
from ...fast import enabled as _fast_on, sm90 as _sm90

import torch

try:
    import triton
    import triton.language as tl
    _TRITON = True
except Exception:  # pragma: no cover
    _TRITON = False


def fused_rmsnorm_enabled():
    return _TRITON and _fast_on("LM_KERNELS")


def _last_pow2(n):
    p = 1
    while p * 2 <= n:
        p *= 2
    return p


def aten_threads_per_row(n, rows, max_threads=512, warp=32):
    """block_width of ATen's ReduceConfig for a contiguous last-dim float reduction."""
    vectorized = n > 128
    dim0 = n // 4 if vectorized else n
    dim1 = rows
    dim0_pow2 = _last_pow2(dim0) if dim0 < max_threads else max_threads
    dim1_pow2 = _last_pow2(dim1) if dim1 < max_threads else max_threads
    bw = min(dim0_pow2, warp)
    bh = min(dim1_pow2, max_threads // bw)
    bw = min(dim0_pow2, max_threads // bh)
    return bw, vectorized


if _TRITON:
    from triton.language.extra.cuda import libdevice as _ld
    from .exact_ops import mul_rn as _xmul_rn, add_rn as _xadd_rn, sub_rn as _xsub_rn

    @triton.jit
    def _tree_reduce(v, T: tl.constexpr):
        # shared-memory phase: v[t] + v[t + off] for off = T/2 ... 32
        if T > 256:
            v = tl.sum(tl.reshape(v, (2, 256)), axis=0)
        if T > 128:
            v = tl.sum(tl.reshape(v, (2, 128)), axis=0)
        if T > 64:
            v = tl.sum(tl.reshape(v, (2, 64)), axis=0)
        if T > 32:
            v = tl.sum(tl.reshape(v, (2, 32)), axis=0)
        # warp-shuffle phase (lane 0): pairs of adjacent lanes, offsets 1, 2, 4, 8, 16
        v = tl.sum(tl.reshape(v, (16, 2)), axis=1)
        v = tl.sum(tl.reshape(v, (8, 2)), axis=1)
        v = tl.sum(tl.reshape(v, (4, 2)), axis=1)
        v = tl.sum(tl.reshape(v, (2, 2)), axis=1)
        v = tl.sum(tl.reshape(v, (1, 2)), axis=1)
        return v

    @triton.jit
    def _rmsnorm_kernel(x_ptr, w_ptr, out_ptr, eps, inv_n,
                        N: tl.constexpr, T: tl.constexpr, VEC: tl.constexpr):
        row = tl.program_id(0)
        base = x_ptr + row * N
        t = tl.arange(0, T)
        a0 = tl.zeros((T,), tl.float32)
        a1 = tl.zeros((T,), tl.float32)
        a2 = tl.zeros((T,), tl.float32)
        a3 = tl.zeros((T,), tl.float32)
        if VEC:
            for j in tl.static_range(N // (4 * T)):
                off = 4 * (t + T * j)
                x0 = tl.load(base + off).to(tl.float32)
                x1 = tl.load(base + off + 1).to(tl.float32)
                x2 = tl.load(base + off + 2).to(tl.float32)
                x3 = tl.load(base + off + 3).to(tl.float32)
                # pow(2) is rounded to fp32 before the reduction adds it: no FMA contraction
                a0 = a0 + _xmul_rn(x0, x0)
                a1 = a1 + _xmul_rn(x1, x1)
                a2 = a2 + _xmul_rn(x2, x2)
                a3 = a3 + _xmul_rn(x3, x3)
        else:
            for m in tl.static_range(N // (4 * T)):
                off = t + 4 * T * m
                x0 = tl.load(base + off).to(tl.float32)
                x1 = tl.load(base + off + T).to(tl.float32)
                x2 = tl.load(base + off + 2 * T).to(tl.float32)
                x3 = tl.load(base + off + 3 * T).to(tl.float32)
                a0 = a0 + _xmul_rn(x0, x0)
                a1 = a1 + _xmul_rn(x1, x1)
                a2 = a2 + _xmul_rn(x2, x2)
                a3 = a3 + _xmul_rn(x3, x3)
        v = ((a0 + a1) + a2) + a3
        total = tl.sum(_tree_reduce(v, T), axis=0)
        var = _xmul_rn(total, inv_n)
        r = tl.math.rsqrt(var + eps)
        cols = tl.arange(0, N)
        x = tl.load(base + cols).to(tl.float32)
        h = (x * r).to(tl.bfloat16)
        w = tl.load(w_ptr + cols)
        out = (w.to(tl.float32) * h.to(tl.float32)).to(tl.bfloat16)
        tl.store(out_ptr + row * N + cols, out)


_CHECKED = {}


def _eligible(module, x):
    w = module.weight
    n = x.shape[-1]
    if not (x.is_cuda and x.dtype == torch.bfloat16 and w.dtype == torch.bfloat16 and x.is_contiguous()
            and w.is_contiguous() and w.numel() == n and n >= 128 and (n & (n - 1)) == 0):
        return False
    rows = x.numel() // n
    if rows < 1:
        return False
    return True


def _launch(module, x):
    n = x.shape[-1]
    rows = x.numel() // n
    T, vec = aten_threads_per_row(n, rows)
    if n % (4 * T):
        return None
    out = torch.empty_like(x)
    nw = 4 if n <= 2048 else 8
    _rmsnorm_kernel[(rows,)](x, module.weight, out, float(module.variance_epsilon), float(1.0 / n),
                             N=n, T=T, VEC=vec, num_warps=nw)
    return out


def rmsnorm_forward(module, x, eager):
    """Fused RMSNorm when eligible and verified for this shape, else eager(x)."""
    if not _eligible(module, x):
        return eager(x)
    n = x.shape[-1]
    rows = x.numel() // n
    key = (n, rows, x.device)
    state = _CHECKED.get(key)
    if state is False:
        return eager(x)
    if state is None:
        if torch.cuda.is_current_stream_capturing():
            return eager(x)          # verify outside a capture first
        ref = eager(x)
        out = _launch(module, x)
        ok = out is not None and torch.equal(out.view(torch.int16), ref.view(torch.int16))
        _CHECKED[key] = bool(ok)
        if not ok:
            print(f"[gstar.fast] fused RMSNorm: live check failed for N={n} rows={rows}, reference ops kept", flush=True)
        return ref
    out = _launch(module, x)
    return eager(x) if out is None else out
