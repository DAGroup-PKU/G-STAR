"""More bit-exact fusions for the decode step (on by default with the lean caches; opt-out GSTAR_FAST_LM_KERNELS=0).

(A) qk_rope_rows: q_norm / k_norm (audio_modeling_omni.RMSNorm, N = 128 -> ATen tree with T = 32 lanes,
    4 unrolled loads per lane), RoPE (fp32 products and sum rounded separately, RNE to bf16), and the
    lean attention operands (q_s = float(q) * s, KS row = float(k) * s, VF row = float(v), every query
    head reading kv head h // rep) in one kernel instead of four.
(B) add_rmsnorm: h = bf16(float(a) + float(b)) (the bf16 residual add) followed by the exact RMSNorm of h
    (same tree as exact_rmsnorm), both outputs written.
(C) silu_mul: bf16(silu(g)) * u with ATen's silu (x / (1 + expf(-x)) in fp32, IEEE division, libdevice
    expf = the CUDA math library's expf) and the bf16 x bf16 product.
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

from .exact_rmsnorm import aten_threads_per_row

if _TRITON:
    from .exact_rmsnorm import _tree_reduce


def lm_fuse2_enabled():
    return _TRITON and _fast_on("LM_KERNELS")


if _TRITON:
    @triton.jit
    def _norm128(base, w_ptr, eps):
        """Exact RMSNorm of one 128-element bf16 row (T = 32 lanes, loads t, t+32, t+64, t+96).
        Returns the four normalized bf16 quarters (as fp32 values) a, b, c, d = n[t], n[t+32], n[t+64], n[t+96]."""
        t = tl.arange(0, 32)
        a = tl.load(base + t).to(tl.float32)
        b = tl.load(base + t + 32).to(tl.float32)
        c = tl.load(base + t + 64).to(tl.float32)
        d = tl.load(base + t + 96).to(tl.float32)
        v = ((_xmul_rn(a, a) + _xmul_rn(b, b)) + _xmul_rn(c, c)) + _xmul_rn(d, d)
        total = tl.sum(_tree_reduce(v, 32), axis=0)
        r = tl.math.rsqrt(_xmul_rn(total, 0.0078125) + eps)
        wa = tl.load(w_ptr + t).to(tl.float32)
        wb = tl.load(w_ptr + t + 32).to(tl.float32)
        wc = tl.load(w_ptr + t + 64).to(tl.float32)
        wd = tl.load(w_ptr + t + 96).to(tl.float32)
        na = (wa * (a * r).to(tl.bfloat16).to(tl.float32)).to(tl.bfloat16).to(tl.float32)
        nb = (wb * (b * r).to(tl.bfloat16).to(tl.float32)).to(tl.bfloat16).to(tl.float32)
        nc = (wc * (c * r).to(tl.bfloat16).to(tl.float32)).to(tl.bfloat16).to(tl.float32)
        nd = (wd * (d * r).to(tl.bfloat16).to(tl.float32)).to(tl.bfloat16).to(tl.float32)
        return na, nb, nc, nd

    @triton.jit
    def _rope128(na, nb, nc, nd, cos_ptr, sin_ptr, pos):
        """RoPE (rotate_half) of a normalized row: quarters [0:32], [32:64] pair with [64:96], [96:128]."""
        t = tl.arange(0, 32)
        cb = cos_ptr + pos * 128
        sb = sin_ptr + pos * 128
        oa = _xadd_rn(_xmul_rn(na, tl.load(cb + t)), _xmul_rn(-nc, tl.load(sb + t)))
        ob = _xadd_rn(_xmul_rn(nb, tl.load(cb + t + 32)), _xmul_rn(-nd, tl.load(sb + t + 32)))
        oc = _xadd_rn(_xmul_rn(nc, tl.load(cb + t + 64)), _xmul_rn(na, tl.load(sb + t + 64)))
        od = _xadd_rn(_xmul_rn(nd, tl.load(cb + t + 96)), _xmul_rn(nb, tl.load(sb + t + 96)))
        return (oa.to(tl.bfloat16).to(tl.float32), ob.to(tl.bfloat16).to(tl.float32),
                oc.to(tl.bfloat16).to(tl.float32), od.to(tl.bfloat16).to(tl.float32))

    @triton.jit
    def _qk_rope_rows_kernel(q_ptr, k_ptr, v_ptr, qw_ptr, kw_ptr, eps, cos_ptr, sin_ptr, pos_ptr, idx_ptr,
                             qs_ptr, ks_ptr, vf_ptr, scale, c_b, c_h, c_l, kb_ptr, vb_ptr, b_b, b_h, b_l,
                             HQ: tl.constexpr, HK: tl.constexpr, BF16_ROWS: tl.constexpr):
        pid = tl.program_id(0)           # b * (HQ + HK) + head ; q rows first, then k (with v) rows
        b = pid // (HQ + HK)
        r = pid % (HQ + HK)
        pos = tl.load(pos_ptr + b)
        t = tl.arange(0, 32)
        if r < HQ:
            na, nb, nc, nd = _norm128(q_ptr + (b * HQ + r) * 128, qw_ptr, eps)
            oa, ob, oc, od = _rope128(na, nb, nc, nd, cos_ptr, sin_ptr, pos)
            ob_ = qs_ptr + (b * HQ + r) * 128
            tl.store(ob_ + t, _xmul_rn(oa, scale))
            tl.store(ob_ + t + 32, _xmul_rn(ob, scale))
            tl.store(ob_ + t + 64, _xmul_rn(oc, scale))
            tl.store(ob_ + t + 96, _xmul_rn(od, scale))
        else:
            hk = r - HQ
            na, nb, nc, nd = _norm128(k_ptr + (b * HK + hk) * 128, kw_ptr, eps)
            oa, ob, oc, od = _rope128(na, nb, nc, nd, cos_ptr, sin_ptr, pos)
            row = tl.load(idx_ptr)
            vb = v_ptr + (b * HK + hk) * 128
            va = tl.load(vb + t).to(tl.float32)
            vb2 = tl.load(vb + t + 32).to(tl.float32)
            vc = tl.load(vb + t + 64).to(tl.float32)
            vd = tl.load(vb + t + 96).to(tl.float32)
            if BF16_ROWS:   # fused attention: the bf16 cache rows k_embed / v (what the reference's KV cache holds)
                kbr = kb_ptr + b * b_b + hk * b_h + row * b_l
                tl.store(kbr + t, oa.to(tl.bfloat16))
                tl.store(kbr + t + 32, ob.to(tl.bfloat16))
                tl.store(kbr + t + 64, oc.to(tl.bfloat16))
                tl.store(kbr + t + 96, od.to(tl.bfloat16))
                vbr = vb_ptr + b * b_b + hk * b_h + row * b_l
                tl.store(vbr + t, va.to(tl.bfloat16))
                tl.store(vbr + t + 32, vb2.to(tl.bfloat16))
                tl.store(vbr + t + 64, vc.to(tl.bfloat16))
                tl.store(vbr + t + 96, vd.to(tl.bfloat16))
            for j in tl.static_range(HQ // HK):
                h = hk * (HQ // HK) + j
                kb = ks_ptr + b * c_b + h * c_h + row * c_l
                tl.store(kb + t, _xmul_rn(oa, scale))
                tl.store(kb + t + 32, _xmul_rn(ob, scale))
                tl.store(kb + t + 64, _xmul_rn(oc, scale))
                tl.store(kb + t + 96, _xmul_rn(od, scale))
                fb = vf_ptr + b * c_b + h * c_h + row * c_l
                tl.store(fb + t, va)
                tl.store(fb + t + 32, vb2)
                tl.store(fb + t + 64, vc)
                tl.store(fb + t + 96, vd)

    @triton.jit
    def _add_rmsnorm_kernel(a_ptr, b_ptr, w_ptr, h_ptr, out_ptr, eps, inv_n,
                            N: tl.constexpr, T: tl.constexpr):
        row = tl.program_id(0)
        t = tl.arange(0, T)
        a0 = tl.zeros((T,), tl.float32)
        a1 = tl.zeros((T,), tl.float32)
        a2 = tl.zeros((T,), tl.float32)
        a3 = tl.zeros((T,), tl.float32)
        for j in tl.static_range(N // (4 * T)):           # vectorized ATen tree (N > 128)
            off = row * N + 4 * (t + T * j)
            h0 = _xadd_rn(tl.load(a_ptr + off).to(tl.float32), tl.load(b_ptr + off).to(tl.float32)).to(tl.bfloat16)
            h1 = _xadd_rn(tl.load(a_ptr + off + 1).to(tl.float32), tl.load(b_ptr + off + 1).to(tl.float32)).to(tl.bfloat16)
            h2 = _xadd_rn(tl.load(a_ptr + off + 2).to(tl.float32), tl.load(b_ptr + off + 2).to(tl.float32)).to(tl.bfloat16)
            h3 = _xadd_rn(tl.load(a_ptr + off + 3).to(tl.float32), tl.load(b_ptr + off + 3).to(tl.float32)).to(tl.bfloat16)
            tl.store(h_ptr + off, h0)
            tl.store(h_ptr + off + 1, h1)
            tl.store(h_ptr + off + 2, h2)
            tl.store(h_ptr + off + 3, h3)
            x0 = h0.to(tl.float32)
            x1 = h1.to(tl.float32)
            x2 = h2.to(tl.float32)
            x3 = h3.to(tl.float32)
            a0 = a0 + _xmul_rn(x0, x0)
            a1 = a1 + _xmul_rn(x1, x1)
            a2 = a2 + _xmul_rn(x2, x2)
            a3 = a3 + _xmul_rn(x3, x3)
        v = ((a0 + a1) + a2) + a3
        total = tl.sum(_tree_reduce(v, T), axis=0)
        r = tl.math.rsqrt(_xmul_rn(total, inv_n) + eps)
        cols = tl.arange(0, N)
        hh = _xadd_rn(tl.load(a_ptr + row * N + cols).to(tl.float32),
                        tl.load(b_ptr + row * N + cols).to(tl.float32)).to(tl.bfloat16).to(tl.float32)
        hn = (hh * r).to(tl.bfloat16)
        w = tl.load(w_ptr + cols)
        tl.store(out_ptr + row * N + cols, (w.to(tl.float32) * hn.to(tl.float32)).to(tl.bfloat16))

    @triton.jit
    def _silu_mul_kernel(g_ptr, u_ptr, out_ptr, n, BLOCK: tl.constexpr):
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        m = offs < n
        g = tl.load(g_ptr + offs, mask=m).to(tl.float32)
        # == ATen's silu for every finite bf16 input (checked exhaustively over all 65,280 values)
        s = (g / (1.0 + _ld.exp(-g))).to(tl.bfloat16).to(tl.float32)
        u = tl.load(u_ptr + offs, mask=m).to(tl.float32)
        tl.store(out_ptr + offs, _xmul_rn(s, u).to(tl.bfloat16), mask=m)


def qk_rope_rows(q, k, v, q_norm, k_norm, cos_, sin_, pos, idx, ks_buf, vf_buf, scale, kb_buf=None, vb_buf=None):
    """q: [B, 1, HQ*128] q_proj output, k/v: [B, 1, HK*128] (contiguous bf16), pos: [B, 1] positions,
    idx: [1] cache row. Returns q_s [B, HQ, 1, 128] fp32 and writes row idx of ks_buf/vf_buf."""
    B = q.shape[0]
    HQ = q.shape[-1] // 128
    HK = k.shape[-1] // 128
    q_s = torch.empty(B, HQ, 1, 128, dtype=torch.float32, device=q.device)
    _qk_rope_rows_kernel[(B * (HQ + HK),)](q, k, v, q_norm.weight, k_norm.weight, float(q_norm.variance_epsilon),
                                           cos_, sin_, pos, idx, q_s, ks_buf, vf_buf, scale,
                                           ks_buf.stride(0), ks_buf.stride(1), ks_buf.stride(2),
                                           kb_buf if kb_buf is not None else ks_buf,
                                           vb_buf if vb_buf is not None else vf_buf,
                                           kb_buf.stride(0) if kb_buf is not None else 0,
                                           kb_buf.stride(1) if kb_buf is not None else 0,
                                           kb_buf.stride(2) if kb_buf is not None else 0,
                                           HQ=HQ, HK=HK, BF16_ROWS=kb_buf is not None, num_warps=1)
    return q_s


def add_rmsnorm(a, b, norm):
    """(h, norm(h)) with h = a + b (bf16), both exactly as the eager ops."""
    n = a.shape[-1]
    rows = a.numel() // n
    T, vec = aten_threads_per_row(n, rows)
    h = torch.empty_like(a)
    out = torch.empty_like(a)
    _add_rmsnorm_kernel[(rows,)](a, b, norm.weight, h, out, float(norm.variance_epsilon), float(1.0 / n),
                                 N=n, T=T, num_warps=4 if n <= 2048 else 8)
    return h, out


def add_rmsnorm_supported(a, b, norm):
    n = a.shape[-1]
    if not (_TRITON and a.is_cuda and a.dtype == torch.bfloat16 and b.dtype == torch.bfloat16
            and norm.weight.dtype == torch.bfloat16 and a.is_contiguous() and b.is_contiguous()
            and a.shape == b.shape and n > 128 and (n & (n - 1)) == 0):
        return False
    T, vec = aten_threads_per_row(n, a.numel() // n)
    return vec and n % (4 * T) == 0


def silu_mul(g, u):
    out = torch.empty_like(g)
    n = g.numel()
    BLOCK = 2048
    _silu_mul_kernel[(triton.cdiv(n, BLOCK),)](g, u, out, n, BLOCK=BLOCK, num_warps=4)
    return out


# ---------------------------------------------------------------------------------------------
# The DSpark head's per-slot glue.  next_slot_input writes, in one kernel, the sampled id
# into tokens[:, slot] (both CFG rows), and the next slot's input_proj input [base_hidden[:, slot+1], emb[id]]
# (what torch.cat([base_hidden[:, slot + 1, :], token_emb_layers[level](sampled)], -1) holds): copies only.
# ---------------------------------------------------------------------------------------------
def lm_fuse3_enabled():
    return _TRITON and _fast_on("LM_KERNELS")


if _TRITON:
    @triton.jit
    def _next_slot_kernel(sampled_ptr, tokens_ptr, tok_stride, slot, emb_ptr, emb_stride, base_ptr, b_stride_r,
                          b_stride_s, inp_ptr, inp_stride, next_slot, ROWS: tl.constexpr, D: tl.constexpr,
                          E: tl.constexpr, HAS_NEXT: tl.constexpr):
        r = tl.program_id(0)
        sid = tl.load(sampled_ptr)
        tl.store(tokens_ptr + r * tok_stride + slot, sid)
        if HAS_NEXT:
            d = tl.arange(0, D)
            tl.store(inp_ptr + r * inp_stride + d, tl.load(base_ptr + r * b_stride_r + next_slot * b_stride_s + d))
            e = tl.arange(0, E)
            tl.store(inp_ptr + r * inp_stride + D + e, tl.load(emb_ptr + sid * emb_stride + e))


def next_slot_input(sampled, tokens, slot, emb_weight, base_hidden, inp, has_next):
    rows = tokens.shape[0]
    D = base_hidden.shape[-1]
    E = emb_weight.shape[-1]
    _next_slot_kernel[(rows,)](sampled, tokens, tokens.stride(0), slot, emb_weight, emb_weight.stride(0),
                               base_hidden, base_hidden.stride(0), base_hidden.stride(1), inp, inp.stride(0),
                               slot + 1, ROWS=rows, D=D, E=E, HAS_NEXT=has_next, num_warps=4)
