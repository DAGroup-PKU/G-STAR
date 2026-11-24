"""Bit-exact faster pieces of the DSpark head's per-slot sampling (on by default; opt-out GSTAR_FAST_LM_KERNELS=0).

Reference (image_head._sample_scores with the HF warpers Temperature -> TopK -> TopP):
    s  = logits[:, :V].float(); s = u + cfg * (c - u)          # slice/cast, sub, mul(scalar), add
    s  = s / T                                                   # ATen: s * (1.0f / T) for a scalar divisor
    s  = s.masked_fill(s < topk(s, k)[0][..., -1:], -inf)        # top-k
    s  = TopP(s); p = softmax(s); multinomial(p)                 # unchanged
Here the first two lines are one kernel with the same fp32 roundings (explicit round-to-nearest, no FMA
contraction), and the top-k threshold, the k-th largest value, comes from torch.kthvalue (the same value
as topk's last element; only the threshold is used).  Everything after the top-k mask is the reference.
"""
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


def fast_sampling_enabled():
    return _TRITON and _fast_on("LM_KERNELS")


if _TRITON:
    @triton.jit
    def _cfg_temp_kernel(lg_ptr, out_ptr, stride_row, half, V, cfg, inv_t, BLOCK: tl.constexpr):
        row = tl.program_id(0)
        blk = tl.program_id(1)
        offs = blk * BLOCK + tl.arange(0, BLOCK)
        m = offs < V
        c = tl.load(lg_ptr + row * stride_row + offs, mask=m).to(tl.float32)
        u = tl.load(lg_ptr + (row + half) * stride_row + offs, mask=m).to(tl.float32)
        s = _xadd_rn(u, _xmul_rn(_xsub_rn(c, u), cfg))
        s = _xmul_rn(s, inv_t)
        tl.store(out_ptr + row * V + offs, s, mask=m)


def _f32(x):
    return float(np.float32(x))


def cfg_temperature(logits, V, cfg_scale, temperature):
    """(u + cfg (c - u)) / T on logits[:, :V] -> fp32 [half, V], rounding exactly like the eager ops."""
    rows = logits.shape[0]
    half = rows // 2
    out = torch.empty(half, V, dtype=torch.float32, device=logits.device)
    inv_t = _f32(np.float32(1.0) / np.float32(temperature))    # ATen: opmath(1) / opmath(T)
    BLOCK = 4096
    _cfg_temp_kernel[(half, triton.cdiv(V, BLOCK))](logits, out, logits.stride(0), half, V,
                                                    _f32(cfg_scale), inv_t, BLOCK=BLOCK, num_warps=8)
    return out


def match_warpers(logits_processor):
    """(temperature, top_k, topp_warper) when the processor list is exactly Temperature, TopK, TopP."""
    if logits_processor is None:
        return None
    try:
        procs = list(logits_processor)
    except TypeError:          # a single callable, not a LogitsProcessorList: the generic path
        return None
    names = [type(p).__name__ for p in procs]
    if names != ["TemperatureLogitsWarper", "TopKLogitsWarper", "TopPLogitsWarper"]:
        return None
    t, k, p = procs
    if not (isinstance(t.temperature, float) and t.temperature > 0) or k.filter_value != -float("inf"):
        return None
    return t.temperature, k, p


def topk_mask(scores, k_warper):
    top_k = min(k_warper.top_k, scores.size(-1))
    n = scores.size(-1)
    thr = torch.kthvalue(scores, n - top_k + 1, dim=-1, keepdim=True)[0]
    return scores.masked_fill(scores < thr, k_warper.filter_value)


# ---------------------------------------------------------------------------------------------
# The elementwise glue around the top-k / top-p decisions in single kernels
# (identical results: comparisons and selections only; sort / softmax / cumsum stay the reference ops).
#   top-k:  s.masked_fill(s < thr, -inf)
#   top-p:  remove = cumprobs <= (1 - top_p) (fp32 compare), remove[..., -keep:] = False,
#           s.masked_fill(remove.scatter(1, sorted_idx, remove), -inf)  ==  out[idx[j]] = remove[j] ? -inf : s[idx[j]]
# ---------------------------------------------------------------------------------------------
def lean2_enabled():
    return _TRITON and _fast_on("LM_KERNELS")


if _TRITON:
    @triton.jit
    def _mask_below_kernel(s_ptr, thr_ptr, out_ptr, N, fill, BLOCK: tl.constexpr):
        row = tl.program_id(0)
        offs = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        m = offs < N
        x = tl.load(s_ptr + row * N + offs, mask=m)
        t = tl.load(thr_ptr + row)
        tl.store(out_ptr + row * N + offs, tl.where(x < t, fill, x), mask=m)

    @triton.jit
    def _topp_apply_kernel(s_ptr, cum_ptr, idx_ptr, out_ptr, N, thr, keep, fill, BLOCK: tl.constexpr):
        row = tl.program_id(0)
        j = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        m = j < N
        c = tl.load(cum_ptr + row * N + j, mask=m)
        idx = tl.load(idx_ptr + row * N + j, mask=m)
        rem = (c <= thr) & (j < N - keep)
        x = tl.load(s_ptr + row * N + idx, mask=m)
        tl.store(out_ptr + row * N + idx, tl.where(rem, fill, x), mask=m)


def topk_mask_lean(scores, k_warper):
    top_k = min(k_warper.top_k, scores.size(-1))
    R, N = scores.shape
    thr = torch.kthvalue(scores, N - top_k + 1, dim=-1, keepdim=True)[0].contiguous()
    out = torch.empty_like(scores)
    BLOCK = 4096
    _mask_below_kernel[(R, triton.cdiv(N, BLOCK))](scores, thr, out, N, float(k_warper.filter_value),
                                                   BLOCK=BLOCK, num_warps=8)
    return out


def top_p_lean(scores, p_warper):
    """TopPLogitsWarper(scores) with the post-scan glue in one kernel (scores: contiguous fp32 [R, N])."""
    sorted_logits, sorted_indices = torch.sort(scores, descending=False)
    cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
    R, N = scores.shape
    out = torch.empty_like(scores)
    thr = float(np.float32(1 - p_warper.top_p))                # ATen compares fp32 with the scalar as fp32
    BLOCK = 4096
    _topp_apply_kernel[(R, triton.cdiv(N, BLOCK))](scores, cumulative_probs, sorted_indices, out, N, thr,
                                                   int(p_warper.min_tokens_to_keep), float(p_warper.filter_value),
                                                   BLOCK=BLOCK, num_warps=8)
    return out


# ---------------------------------------------------------------------------------------------
# One stable sort serves top-k and top-p.  With sorted_all, idx = sort(scores) (ascending,
# stable): the top-k threshold is sorted_all[N - k] (the k-th largest, as topk(...)[0][..., -1]); the array the
# top-p warper sorts, sort(masked_fill(scores < thr, -inf)), has the values where(sorted_all < thr, -inf,
# sorted_all) at the same positions (kept values keep their stable order, the removed ones become the leading
# -inf block), so softmax and cumsum see identical inputs.  Only the order of the indices inside the -inf block
# differs, and every entry of that block is removed (cumprob 0 <= 1 - top_p) whatever its order, so the result
# of the scatter / masked_fill is identical.
# ---------------------------------------------------------------------------------------------
def sort1_enabled():
    return _TRITON and _fast_on("LM_KERNELS")


if _TRITON:
    @triton.jit
    def _mask_sorted_kernel(sorted_ptr, out_ptr, thr_ptr, N, fill, BLOCK: tl.constexpr):
        row = tl.program_id(0)
        offs = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        m = offs < N
        x = tl.load(sorted_ptr + row * N + offs, mask=m)
        t = tl.load(thr_ptr + row)
        tl.store(out_ptr + row * N + offs, tl.where(x < t, fill, x), mask=m)

    @triton.jit
    def _topkp_apply_kernel(s_ptr, cum_ptr, idx_ptr, thr_ptr, out_ptr, N, pthr, keep, kfill, pfill,
                            BLOCK: tl.constexpr):
        row = tl.program_id(0)
        j = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        m = j < N
        c = tl.load(cum_ptr + row * N + j, mask=m)
        idx = tl.load(idx_ptr + row * N + j, mask=m)
        t = tl.load(thr_ptr + row)
        x = tl.load(s_ptr + row * N + idx, mask=m)
        x = tl.where(x < t, kfill, x)                          # top-k mask
        rem = (c <= pthr) & (j < N - keep)                     # top-p removal
        tl.store(out_ptr + row * N + idx, tl.where(rem, pfill, x), mask=m)


def topk_topp_sort1(scores, k_warper, p_warper):
    """TopP(TopK(scores)) of the HF warpers with one sort (scores: contiguous fp32 [R, N])."""
    R, N = scores.shape
    top_k = min(k_warper.top_k, N)
    sorted_all, idx = torch.sort(scores, descending=False)
    thr = sorted_all[:, N - top_k:N - top_k + 1].contiguous()
    # 512-wide programs for the two elementwise kernels (33 CTAs instead of 4; same values)
    BLOCK, nw = (512, 4) if _fast_on("LM_KERNELS") else (4096, 8)
    grid = (R, triton.cdiv(N, BLOCK))
    sorted_masked = torch.empty_like(sorted_all)
    _mask_sorted_kernel[grid](sorted_all, sorted_masked, thr, N, float(k_warper.filter_value), BLOCK=BLOCK, num_warps=nw)
    cumulative_probs = sorted_masked.softmax(dim=-1).cumsum(dim=-1)
    out = torch.empty_like(scores)
    _topkp_apply_kernel[grid](scores, cumulative_probs, idx, thr, out, N, float(np.float32(1 - p_warper.top_p)),
                              int(p_warper.min_tokens_to_keep), float(k_warper.filter_value),
                              float(p_warper.filter_value), BLOCK=BLOCK, num_warps=nw)
    return out
