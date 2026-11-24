"""Bit-exact decode GEMMs for the backbone MLP (on by default; opt-out GSTAR_FAST_LM_GEMM=0).

At the decode step (CFG batch: M = 2 rows) cuBLAS runs the MLP linears gate_proj / up_proj (12288 x 4096) and
down_proj (4096 x 12288) with the sm90 64x64 kernel without split-K: every output is one fp32 accumulation over K
in k16 tensor-core steps, in order, rounded once to bf16.  A Triton tensor-core GEMM that walks K in the same
order gives the same bits (checked: torch.equal on random inputs for every tile configuration tried, mma.sync
and wgmma), so the result does not depend on the N tiling.  These kernels stream the weights with more CTAs
in flight (25.8 us vs 29.0 us for gate/up, 27.0 vs 29.6 us for down in isolation).  gate and up are computed by
one kernel that also applies the reference's SiLU(gate) * up (each GEMM output rounded to bf16 first, then the
exact silu_mul arithmetic of exact_fused2), so the [2, 12288] gate/up tensors are never written.
q/k/v/o use cuBLAS split-K kernels (a different summation order) and stay on cuBLAS.

Every (kernel, shape) is checked against the cuBLAS reference on random inputs at its first use (live check,
eagerly, before graph capture); a mismatch switches that shape back to the reference ops.
"""
import os
from ...fast import enabled as _fast_on, sm90 as _sm90

import torch
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
    from triton.language.extra.cuda import libdevice as _ld
    from .exact_ops import mul_rn as _xmul_rn
    _TRITON = True
except Exception:  # pragma: no cover
    _TRITON = False


def lm_tgemm_enabled():
    return _TRITON and _fast_on("LM_GEMM")


if _TRITON:
    @triton.jit
    def _m2_gemm_kernel(x_ptr, w_ptr, o_ptr, M, K, sxm, swn, som,
                        BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
        pid = tl.program_id(0)
        rm = tl.arange(0, BM)
        rn = pid * BN + tl.arange(0, BN)
        rk = tl.arange(0, BK)
        acc = tl.zeros((BM, BN), dtype=tl.float32)
        for k0 in range(0, K, BK):
            a = tl.load(x_ptr + rm[:, None] * sxm + (k0 + rk)[None, :], mask=rm[:, None] < M, other=0.0)
            b = tl.load(w_ptr + rn[None, :] * swn + (k0 + rk)[:, None])
            acc = tl.dot(a, b, acc)
        tl.store(o_ptr + rm[:, None] * som + rn[None, :], acc.to(tl.bfloat16), mask=rm[:, None] < M)

    @triton.jit
    def _m2_gate_up_silu_kernel(x_ptr, wg_ptr, wu_ptr, o_ptr, M, K, sxm, swg, swu, som,
                                BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
        pid = tl.program_id(0)
        rm = tl.arange(0, BM)
        rn = pid * BN + tl.arange(0, BN)
        rk = tl.arange(0, BK)
        acc_g = tl.zeros((BM, BN), dtype=tl.float32)
        acc_u = tl.zeros((BM, BN), dtype=tl.float32)
        for k0 in range(0, K, BK):
            a = tl.load(x_ptr + rm[:, None] * sxm + (k0 + rk)[None, :], mask=rm[:, None] < M, other=0.0)
            bg = tl.load(wg_ptr + rn[None, :] * swg + (k0 + rk)[:, None])
            bu = tl.load(wu_ptr + rn[None, :] * swu + (k0 + rk)[:, None])
            acc_g = tl.dot(a, bg, acc_g)
            acc_u = tl.dot(a, bu, acc_u)
        g = acc_g.to(tl.bfloat16).to(tl.float32)
        # == ATen's silu for every finite bf16 input (the expression of exact_fused2._silu_mul_kernel)
        s = (g / (1.0 + _ld.exp(-g))).to(tl.bfloat16).to(tl.float32)
        u = acc_u.to(tl.bfloat16).to(tl.float32)
        tl.store(o_ptr + rm[:, None] * som + rn[None, :], _xmul_rn(s, u).to(tl.bfloat16), mask=rm[:, None] < M)


# (BN, BK, num_warps, num_stages), from a sweep on the H200 (M = 2)
_CFG_GEMM = {(4096, 12288): (32, 256, 4, 6), (12288, 4096): (32, 128, 2, 4)}
_CFG_GATE_UP = (32, 128, 2, 3)
_OK = {}


def _rows(x):
    K = x.shape[-1]
    return x.reshape(-1, K), x.shape[:-1]


def _gemm_launch(x2, w, cfg):
    M, K = x2.shape
    N = w.shape[0]
    BN, BK, warps, stages = cfg
    out = torch.empty(M, N, dtype=x2.dtype, device=x2.device)
    _m2_gemm_kernel[(N // BN,)](x2, w, out, M, K, x2.stride(0), w.stride(0), out.stride(0),
                                BM=16, BN=BN, BK=BK, num_warps=warps, num_stages=stages)
    return out


def _gate_up_launch(x2, wg, wu, cfg):
    M, K = x2.shape
    N = wg.shape[0]
    BN, BK, warps, stages = cfg
    out = torch.empty(M, N, dtype=x2.dtype, device=x2.device)
    _m2_gate_up_silu_kernel[(N // BN,)](x2, wg, wu, out, M, K, x2.stride(0), wg.stride(0), wu.stride(0),
                                        out.stride(0), BM=16, BN=BN, BK=BK, num_warps=warps, num_stages=stages)
    return out


def _shape_ok(x2, *ws):
    M, K = x2.shape
    return (x2.is_cuda and x2.dtype == torch.bfloat16 and x2.is_contiguous() and M <= 16
            and all(w.dtype == torch.bfloat16 and w.is_contiguous() and w.dim() == 2 and w.shape[1] == K
                    for w in ws))


def _live_check(key, fn, ref, K, device, M, trials=4):
    """Bitwise check of a replica against the reference op on random [M, K] inputs, once per key (the key holds
    the weight shape and M: cuBLAS chooses its kernel, hence its summation order, per problem shape)."""
    key = key + (M,)
    if key not in _OK:
        ok = True
        g = torch.Generator(device=device)
        g.manual_seed(1234)
        for t in range(trials):
            x = torch.randn(M, K, device=device, dtype=torch.bfloat16, generator=g) * (0.5 + t)
            if not torch.equal(fn(x), ref(x)):
                ok = False
                break
        _OK[key] = ok
        if not ok:
            print(f"[gstar.fast] decode GEMM replica: live check failed for {key}, reference ops kept", flush=True)
    return _OK[key]


def m2_linear(x, w):
    """F.linear(x, w) (no bias) for decode inputs with <= 16 rows, bitwise; cuBLAS where not verified."""
    x2, lead = _rows(x)
    cfg = _CFG_GEMM.get(tuple(w.shape))
    if cfg is None or not _shape_ok(x2, w) or w.shape[0] % cfg[0]:
        return F.linear(x, w)
    key = ("gemm", tuple(w.shape))
    if not _live_check(key, lambda t: _gemm_launch(t, w, cfg), lambda t: F.linear(t, w), w.shape[1], x.device,
                       x2.shape[0]):
        return F.linear(x, w)
    return _gemm_launch(x2, w, cfg).view(*lead, w.shape[0])


def m2_gate_up_silu(x, wg, wu, silu_mul_ref):
    """silu_mul_ref(F.linear(x, wg), F.linear(x, wu)) in one kernel, bitwise."""
    x2, lead = _rows(x)
    cfg = _CFG_GATE_UP
    if (not _shape_ok(x2, wg, wu) or wg.shape != wu.shape or wg.shape[0] % cfg[0]
            or x2.shape[1] % cfg[1]):
        return silu_mul_ref(F.linear(x, wg), F.linear(x, wu))
    key = ("gate_up", tuple(wg.shape))
    if not _live_check(key, lambda t: _gate_up_launch(t, wg, wu, cfg),
                       lambda t: silu_mul_ref(F.linear(t, wg), F.linear(t, wu)), wg.shape[1], x.device,
                       x2.shape[0]):
        return silu_mul_ref(F.linear(x, wg), F.linear(x, wu))
    return _gate_up_launch(x2, wg, wu, cfg).view(*lead, wg.shape[0])


# ---------------------------------------------------------------------------------------------
# Split-K shapes.  For the attention projections at M = 2 cuBLAS runs cutlass sm80 split-K
# kernels + cublasLt::splitKreduce: K is cut into chunks of ceil(k_tiles / splits) tiles, every chunk is one
# in-order fp32 accumulation, stored as a partial (bf16-rounded for q_proj / o_proj (4096 x 4096): 4 chunks of
# 1024; fp32 for k_proj / v_proj (1024 x 4096): 16 chunks of 256), and the reduce adds the partials in chunk order
# into an fp32 sum that starts at 0, rounded once to bf16.  Found by hypothesis search and checked adversarially
# (partials of very different magnitudes: the sequential order matches 20/20, a tree order does not).
# One CTA here walks K once and folds its accumulator into the running sum at every chunk boundary - the same
# additions in the same order; q, k and v (same input) are computed by one launch.
# ---------------------------------------------------------------------------------------------
def lm_tgemm2_enabled():
    return _TRITON and _fast_on("LM_GEMM")


# (N, K) -> (chunk length along K, bf16-rounded partials), measured for these shapes; any other shape is looked
# up once at its first use (_discover_splitk), None where no configuration reproduces cuBLAS
_SPLITK = {(4096, 4096): (1024, True), (1024, 4096): (256, False)}


if _TRITON:
    @triton.jit
    def _splitk_body(x_ptr, w_ptr, o_ptr, n0, M, K, sxm, swn, som, chunk_tiles, rnd,
                     BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
        rm = tl.arange(0, BM)
        rn = n0 + tl.arange(0, BN)
        rk = tl.arange(0, BK)
        acc = tl.zeros((BM, BN), dtype=tl.float32)
        total = tl.zeros((BM, BN), dtype=tl.float32)
        nk = K // BK
        for kt in range(0, nk):
            k0 = kt * BK
            a = tl.load(x_ptr + rm[:, None] * sxm + (k0 + rk)[None, :], mask=rm[:, None] < M,
                        other=0.0).to(tl.bfloat16)          # fp32 A (o_proj on the fp32 attention output): the exact .to(bf16) of the reference
            b = tl.load(w_ptr + rn[None, :] * swn + (k0 + rk)[:, None])
            acc = tl.dot(a, b, acc)
            if (kt + 1) % chunk_tiles == 0:
                if rnd != 0:
                    total = total + acc.to(tl.bfloat16).to(tl.float32)
                else:
                    total = total + acc
                acc = tl.zeros((BM, BN), dtype=tl.float32)
        tl.store(o_ptr + rm[:, None] * som + rn[None, :], total.to(tl.bfloat16), mask=rm[:, None] < M)

    @triton.jit
    def _m2_splitk_kernel(x_ptr, w_ptr, o_ptr, M, K, sxm, swn, som, chunk_tiles, rnd,
                          BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
        _splitk_body(x_ptr, w_ptr, o_ptr, tl.program_id(0) * BN, M, K, sxm, swn, som, chunk_tiles, rnd,
                     BM=BM, BN=BN, BK=BK)

    @triton.jit
    def _m2_qkv_kernel(x_ptr, wq_ptr, wk_ptr, wv_ptr, oq_ptr, ok_ptr, ov_ptr, M, K, sxm, swq, swk, swv,
                       soq, sok, sov, nbq, nbk, ctq, rq, ctk, rk_, BM: tl.constexpr, BN: tl.constexpr,
                       BK: tl.constexpr):
        pid = tl.program_id(0)
        if pid < nbq:
            _splitk_body(x_ptr, wq_ptr, oq_ptr, pid * BN, M, K, sxm, swq, soq, ctq, rq, BM=BM, BN=BN, BK=BK)
        elif pid < nbq + nbk:
            _splitk_body(x_ptr, wk_ptr, ok_ptr, (pid - nbq) * BN, M, K, sxm, swk, sok, ctk, rk_,
                         BM=BM, BN=BN, BK=BK)
        else:
            _splitk_body(x_ptr, wv_ptr, ov_ptr, (pid - nbq - nbk) * BN, M, K, sxm, swv, sov, ctk, rk_,
                         BM=BM, BN=BN, BK=BK)


_CFG_SPLITK = (32, 256, 4, 6)       # (BN, BK, num_warps, num_stages), o_proj (sweep on the H200)
_CFG_QKV = (64, 256, 4, 4)          # q/k/v launch


def _splitk_launch(x2, w, cfg):
    M, K = x2.shape
    N = w.shape[0]
    chunk, rnd = _SPLITK[(N, K)]
    BN, BK, warps, stages = cfg
    out = torch.empty(M, N, dtype=w.dtype, device=x2.device)      # bf16 (also for an fp32 A)
    _m2_splitk_kernel[(N // BN,)](x2, w, out, M, K, x2.stride(0), w.stride(0), out.stride(0), chunk // BK,
                                  int(rnd), BM=16, BN=BN, BK=BK, num_warps=warps, num_stages=stages)
    return out


def _qkv_launch(x2, wq, wk, wv, cfg):
    M, K = x2.shape
    BN, BK, warps, stages = cfg
    cq, rq = _SPLITK[tuple(wq.shape)]
    ck, rk = _SPLITK[tuple(wk.shape)]
    oq = torch.empty(M, wq.shape[0], dtype=x2.dtype, device=x2.device)
    ok = torch.empty(M, wk.shape[0], dtype=x2.dtype, device=x2.device)
    ov = torch.empty(M, wv.shape[0], dtype=x2.dtype, device=x2.device)
    nbq, nbk = wq.shape[0] // BN, wk.shape[0] // BN
    _m2_qkv_kernel[(nbq + 2 * nbk,)](x2, wq, wk, wv, oq, ok, ov, M, K, x2.stride(0), wq.stride(0), wk.stride(0),
                                     wv.stride(0), oq.stride(0), ok.stride(0), ov.stride(0), nbq, nbk, cq // BK,
                                     int(rq), ck // BK, int(rk), BM=16, BN=BN, BK=BK, num_warps=warps,
                                     num_stages=stages)
    return oq, ok, ov


def _discover_splitk(w, M):
    """cuBLAS's split-K configuration for an unlisted weight shape: the (chunk, bf16 partials) whose replica equals
    F.linear bitwise on random [M, K] inputs (three scales and a cancellation-heavy one), if exactly one candidate
    does - candidates are the uniform partitions of K into chunks of whole replica K tiles; else None (cuBLAS
    is kept).  The live check verifies the shape again before the replica is used."""
    N, K = w.shape
    g = torch.Generator(device=w.device)
    g.manual_seed(4321)
    xs = [torch.randn(M, K, device=w.device, generator=g) * (0.5 + t) for t in range(3)]
    xc = torch.randn(M, K, device=w.device, generator=g)
    xc[:, ::7] *= 1000.0
    xs = [x.to(torch.bfloat16) for x in xs + [xc]]
    refs = [F.linear(x, w) for x in xs]
    bk = _CFG_SPLITK[1]
    hits = []
    for chunk in range(bk, K + 1, bk):
        if K % chunk:
            continue
        for rnd in ((True,) if chunk == K else (True, False)):      # one chunk: the rounding flag is moot
            _SPLITK[(N, K)] = (chunk, rnd)
            if all(torch.equal(_splitk_launch(x, w, _CFG_SPLITK), r) for x, r in zip(xs, refs)):
                hits.append((chunk, rnd))
    _SPLITK[(N, K)] = hits[0] if len(hits) == 1 else None
    return _SPLITK[(N, K)]


def _splitk_ok(x2, w, cfg):
    if not (_shape_ok(x2, w) and w.shape[0] % cfg[0] == 0 and w.shape[0] % _CFG_SPLITK[0] == 0):
        return False
    key = tuple(w.shape)
    if key not in _SPLITK:
        _discover_splitk(w, x2.shape[0])
    e = _SPLITK[key]
    return e is not None and e[0] % cfg[1] == 0 and x2.shape[1] % e[0] == 0


def m2_linear_splitk(x, w):
    """F.linear(x, w) (no bias, <= 16 rows) for the split-K shapes, bitwise; cuBLAS where not verified."""
    x2, lead = _rows(x)
    cfg = _CFG_SPLITK
    if not _splitk_ok(x2, w, cfg):
        return F.linear(x, w)
    key = ("splitk", tuple(w.shape))
    if not _live_check(key, lambda t: _splitk_launch(t, w, cfg), lambda t: F.linear(t, w), w.shape[1], x.device,
                       x2.shape[0]):
        return F.linear(x, w)
    return _splitk_launch(x2, w, cfg).view(*lead, w.shape[0])


def m2_linear_splitk_f32a(x32, w):
    """F.linear(x32.to(bf16), w) for an fp32 x32 (the decode attention output), bitwise, in one kernel: the
    bf16 rounding of the reference's .to(bf16) happens on the A load."""
    x2, lead = _rows(x32)
    cfg = _CFG_SPLITK
    key = tuple(w.shape)
    ok = (_SPLITK.get(key) is not None and x2.is_cuda and x2.dtype == torch.float32 and x2.is_contiguous() and x2.shape[0] <= 16
          and w.dtype == torch.bfloat16 and w.is_contiguous() and w.shape[1] == x2.shape[1]
          and w.shape[0] % cfg[0] == 0 and _SPLITK[key][0] % cfg[1] == 0 and x2.shape[1] % _SPLITK[key][0] == 0)
    if not ok:
        return F.linear(x32.to(torch.bfloat16), w)
    ckey = ("splitk_f32a", key, x2.shape[0])
    if ckey not in _OK:
        good = True
        g = torch.Generator(device=x32.device)
        g.manual_seed(999)
        for t in range(4):
            xt = torch.randn(x2.shape[0], w.shape[1], device=x32.device, dtype=torch.float32, generator=g) * (0.5 + t)
            if not torch.equal(_splitk_launch(xt, w, cfg), F.linear(xt.to(torch.bfloat16), w)):
                good = False
                break
        _OK[ckey] = good
        if not good:
            print(f"[gstar.fast] decode GEMM replica: live check failed for splitk_f32a {key}, reference ops kept", flush=True)
    if not _OK[ckey]:
        return F.linear(x32.to(torch.bfloat16), w)
    return _splitk_launch(x2, w, cfg).view(*lead, w.shape[0])


def m2_qkv(x, wq, wk, wv):
    """(F.linear(x, wq), F.linear(x, wk), F.linear(x, wv)) in one launch, bitwise (no biases)."""
    x2, lead = _rows(x)
    cfg = _CFG_QKV
    if not (_splitk_ok(x2, wq, cfg) and _splitk_ok(x2, wk, cfg) and wv.shape == wk.shape
            and _splitk_ok(x2, wv, cfg)):
        return F.linear(x, wq), F.linear(x, wk), F.linear(x, wv)
    key = ("qkv", tuple(wq.shape), tuple(wk.shape))
    if not _live_check(key, lambda t: torch.cat(_qkv_launch(t, wq, wk, wv, cfg), -1),
                       lambda t: torch.cat([F.linear(t, wq), F.linear(t, wk), F.linear(t, wv)], -1),
                       wq.shape[1], x.device, x2.shape[0]):
        return F.linear(x, wq), F.linear(x, wk), F.linear(x, wv)
    q, k, v = _qkv_launch(x2, wq, wk, wv, cfg)
    return q.view(*lead, -1), k.view(*lead, -1), v.view(*lead, -1)


# ---------------------------------------------------------------------------------------------
# The DSpark head's linears.  With a bias, F.linear at M = 2 goes to cublasLt gemm_and_bias:
#   q_proj / v_proj / out_proj (2048 x 2048): K chunks [704, 704, 640]; input_proj (2048 x 2304): 3 x 768
#   (fc2 (2048 x 8192): [1408 x 5, 1152] - left on cuBLAS: without a parallel split the replica is slower);
#   partials rounded to bf16, added in chunk order from 0, the sum rounded to bf16, then + bias, rounded again;
#   fc1 (8192 x 2048): no split, bf16(acc + bias);  k_proj (no bias): 8 chunks of 256, fp32 partials.
# Found by hypothesis search and checked adversarially (30/30 with per-chunk magnitudes 2^±10).
# ---------------------------------------------------------------------------------------------
def lm_tgemm3_enabled():
    return _TRITON and _fast_on("LM_GEMM")


# (N, K, has_bias) -> (chunk length, bf16 partials, bias mode: 0 none, 1 bf16(acc + b), 2 bf16(bf16(sum) + b))
_HEAD_SPLITK = {
    (2048, 2304, True): (768, True, 2),
    (2048, 2048, True): (704, True, 2),
    (2048, 2048, False): (256, False, 0),
    (8192, 2048, True): (2048, False, 1),
}
_HEAD_CFG = {(2048, 2304, True): (16, 256, 4, 4), (2048, 2048, True): (16, 64, 2, 3),
             (8192, 2048, True): (32, 128, 2, 6)}
_HEAD_QKV_CFG = (32, 64, 2, 8)


if _TRITON:
    @triton.jit
    def _sk2_body(x_ptr, w_ptr, b_ptr, o_ptr, n0, M, K, sxm, swn, som, chunk_tiles, rnd, bias_mode,
                  BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
        rm = tl.arange(0, BM)
        rn = n0 + tl.arange(0, BN)
        rk = tl.arange(0, BK)
        acc = tl.zeros((BM, BN), dtype=tl.float32)
        total = tl.zeros((BM, BN), dtype=tl.float32)
        nk = K // BK
        for kt in range(0, nk):
            k0 = kt * BK
            a = tl.load(x_ptr + rm[:, None] * sxm + (k0 + rk)[None, :], mask=rm[:, None] < M, other=0.0)
            b = tl.load(w_ptr + rn[None, :] * swn + (k0 + rk)[:, None])
            acc = tl.dot(a, b, acc)
            if ((kt + 1) % chunk_tiles == 0) | (kt + 1 == nk):
                if rnd != 0:
                    total = total + acc.to(tl.bfloat16).to(tl.float32)
                else:
                    total = total + acc
                acc = tl.zeros((BM, BN), dtype=tl.float32)
        if bias_mode == 0:
            out = total.to(tl.bfloat16)
        else:
            bias = tl.load(b_ptr + rn).to(tl.float32)
            if bias_mode == 1:
                out = (total + bias[None, :]).to(tl.bfloat16)
            else:
                out = (total.to(tl.bfloat16).to(tl.float32) + bias[None, :]).to(tl.bfloat16)
        tl.store(o_ptr + rm[:, None] * som + rn[None, :], out, mask=rm[:, None] < M)

    @triton.jit
    def _sk2_kernel(x_ptr, w_ptr, b_ptr, o_ptr, M, K, sxm, swn, som, chunk_tiles, rnd, bias_mode,
                    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
        _sk2_body(x_ptr, w_ptr, b_ptr, o_ptr, tl.program_id(0) * BN, M, K, sxm, swn, som, chunk_tiles, rnd,
                  bias_mode, BM=BM, BN=BN, BK=BK)

    @triton.jit
    def _sk2_qkv_kernel(x_ptr, wq_ptr, wk_ptr, wv_ptr, bq_ptr, bk_ptr, bv_ptr, oq_ptr, ok_ptr, ov_ptr, M, K, sxm,
                        sw, so, nb, ctq, rq, bmq, ctk, rk_, bmk, ctv, rv, bmv,
                        BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
        pid = tl.program_id(0)
        if pid < nb:
            _sk2_body(x_ptr, wq_ptr, bq_ptr, oq_ptr, pid * BN, M, K, sxm, sw, so, ctq, rq, bmq,
                      BM=BM, BN=BN, BK=BK)
        elif pid < 2 * nb:
            _sk2_body(x_ptr, wk_ptr, bk_ptr, ok_ptr, (pid - nb) * BN, M, K, sxm, sw, so, ctk, rk_, bmk,
                      BM=BM, BN=BN, BK=BK)
        else:
            _sk2_body(x_ptr, wv_ptr, bv_ptr, ov_ptr, (pid - 2 * nb) * BN, M, K, sxm, sw, so, ctv, rv, bmv,
                      BM=BM, BN=BN, BK=BK)


def _hspec(w, b):
    return _HEAD_SPLITK.get((w.shape[0], w.shape[1], b is not None))


def _head_ok(x2, w, b, BN, BK):
    sp = _hspec(w, b)
    return (sp is not None and _shape_ok(x2, w) and w.shape[0] % BN == 0 and sp[0] % BK == 0
            and x2.shape[1] % BK == 0 and (b is None or (b.dtype == torch.bfloat16 and b.is_contiguous())))


def _head_launch(x2, w, b, cfg):
    M, K = x2.shape
    N = w.shape[0]
    chunk, rnd, bm = _hspec(w, b)
    BN, BK, warps, stages = cfg
    out = torch.empty(M, N, dtype=x2.dtype, device=x2.device)
    _sk2_kernel[(N // BN,)](x2, w, b if b is not None else w, out, M, K, x2.stride(0), w.stride(0), out.stride(0),
                            chunk // BK, int(rnd), bm, BM=16, BN=BN, BK=BK, num_warps=warps, num_stages=stages)
    return out


def _head_qkv_launch(x2, wq, bq, wk, bk, wv, bv, cfg):
    M, K = x2.shape
    N = wq.shape[0]
    BN, BK, warps, stages = cfg
    (cq, rq, mq), (ck, rk, mk), (cv, rv, mv) = _hspec(wq, bq), _hspec(wk, bk), _hspec(wv, bv)
    oq, ok, ov = (torch.empty(M, N, dtype=x2.dtype, device=x2.device) for _ in range(3))
    _sk2_qkv_kernel[(3 * (N // BN),)](x2, wq, wk, wv, bq if bq is not None else wq, bk if bk is not None else wq,
                                      bv if bv is not None else wq, oq, ok, ov, M, K, x2.stride(0), wq.stride(0),
                                      oq.stride(0), N // BN, cq // BK, int(rq), mq, ck // BK, int(rk), mk, cv // BK,
                                      int(rv), mv, BM=16, BN=BN, BK=BK, num_warps=warps, num_stages=stages)
    return oq, ok, ov


def _live_check_shaped(key, fn, ref, shape, device, trials=4):
    if key not in _OK:
        ok = True
        g = torch.Generator(device=device)
        g.manual_seed(4321)
        for t in range(trials):
            x = torch.randn(*shape, device=device, dtype=torch.bfloat16, generator=g) * (0.5 + t)
            r = ref(x)
            o = fn(x)
            if not all(torch.equal(a, c) for a, c in zip(o, r)):
                ok = False
                break
        _OK[key] = ok
        if not ok:
            print(f"[gstar.fast] decode GEMM replica: live check failed for {key}, reference ops kept", flush=True)
    return _OK[key]


def head_linear(x, w, b):
    """F.linear(x, w, b) for the DSpark head's decode inputs (<= 16 rows), bitwise; cuBLAS where not verified."""
    x2, lead = _rows(x)
    cfg = _HEAD_CFG.get((w.shape[0], w.shape[1], b is not None))
    if cfg is None or not _head_ok(x2, w, b, cfg[0], cfg[1]):
        return F.linear(x, w, b)
    key = ("head", tuple(w.shape), b is not None, tuple(x.shape))
    if not _live_check_shaped(key, lambda t: (_head_launch(t.reshape(-1, t.shape[-1]), w, b, cfg).view(*t.shape[:-1], -1),),
                              lambda t: (F.linear(t, w, b),), tuple(x.shape), x.device):
        return F.linear(x, w, b)
    return _head_launch(x2, w, b, cfg).view(*lead, w.shape[0])


def head_qkv(x, wq, bq, wk, bk, wv, bv):
    """(F.linear(x, wq, bq), F.linear(x, wk, bk), F.linear(x, wv, bv)) in one launch, bitwise."""
    x2, lead = _rows(x)
    cfg = _HEAD_QKV_CFG
    if not (wq.shape == wk.shape == wv.shape and all(_head_ok(x2, w, b, cfg[0], cfg[1])
                                                     for w, b in ((wq, bq), (wk, bk), (wv, bv)))):
        return F.linear(x, wq, bq), F.linear(x, wk, bk), F.linear(x, wv, bv)
    key = ("head_qkv", tuple(wq.shape), bq is not None, bk is not None, bv is not None, tuple(x.shape))

    def fn(t):
        return tuple(o.view(*t.shape[:-1], -1) for o in _head_qkv_launch(t.reshape(-1, t.shape[-1]), wq, bq, wk, bk,
                                                                        wv, bv, cfg))

    if not _live_check_shaped(key, fn, lambda t: (F.linear(t, wq, bq), F.linear(t, wk, bk), F.linear(t, wv, bv)),
                              tuple(x.shape), x.device):
        return F.linear(x, wq, bq), F.linear(x, wk, bk), F.linear(x, wv, bv)
    q, k, v = _head_qkv_launch(x2, wq, bq, wk, bk, wv, bv, cfg)
    return q.view(*lead, -1), k.view(*lead, -1), v.view(*lead, -1)
