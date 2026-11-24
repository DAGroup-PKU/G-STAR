import torch, os, inspect
from ...fast import enabled as _fast_on, sm90 as _sm90
from einops import rearrange, repeat


try:
    import flash_attn_interface
    FLASH_ATTN_3_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_3_AVAILABLE = False

try:
    import flash_attn
    FLASH_ATTN_2_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_2_AVAILABLE = False

try:
    from sageattention import sageattn
    SAGE_ATTN_AVAILABLE = True
except ModuleNotFoundError:
    SAGE_ATTN_AVAILABLE = False

try:
    import xformers.ops as xops
    XFORMERS_AVAILABLE = True
except ModuleNotFoundError:
    XFORMERS_AVAILABLE = False

try:
    if "enable_gqa" in inspect.signature(torch.nn.functional.scaled_dot_product_attention).parameters:
        TORCH_SUPPORT_GQA = True
    else:
        TORCH_SUPPORT_GQA = False
except:
    TORCH_SUPPORT_GQA = False


def initialize_attention_priority():
    if os.environ.get('DIFFSYNTH_ATTENTION_IMPLEMENTATION') is not None:
        return os.environ.get('DIFFSYNTH_ATTENTION_IMPLEMENTATION').lower()
    elif FLASH_ATTN_3_AVAILABLE:
        return "flash_attention_3"
    elif FLASH_ATTN_2_AVAILABLE:
        return "flash_attention_2"
    elif SAGE_ATTN_AVAILABLE:
        return "sage_attention"
    elif XFORMERS_AVAILABLE:
        return "xformers"
    else:
        return "torch"


ATTENTION_IMPLEMENTATION = initialize_attention_priority()


def rearrange_qkv(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, q_pattern="b n s d", k_pattern="b n s d", v_pattern="b n s d", required_in_pattern="b n s d", dims=None):
    dims = {} if dims is None else dims
    if q_pattern != required_in_pattern:
        q = rearrange(q, f"{q_pattern} -> {required_in_pattern}", **dims)
    if k_pattern != required_in_pattern:
        k = rearrange(k, f"{k_pattern} -> {required_in_pattern}", **dims)
    if v_pattern != required_in_pattern:
        v = rearrange(v, f"{v_pattern} -> {required_in_pattern}", **dims)
    return q, k, v


def rearrange_out(out: torch.Tensor, out_pattern="b n s d", required_out_pattern="b n s d", dims=None):
    dims = {} if dims is None else dims
    if out_pattern != required_out_pattern:
        out = rearrange(out, f"{required_out_pattern} -> {out_pattern}", **dims)
    return out


def torch_sdpa(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, q_pattern="b n s d", k_pattern="b n s d", v_pattern="b n s d", out_pattern="b n s d", dims=None, attn_mask=None, scale=None, is_causal=False):
    required_in_pattern, required_out_pattern= "b n s d", "b n s d"
    q, k, v = rearrange_qkv(q, k, v, q_pattern, k_pattern, v_pattern, required_in_pattern, dims)
    if q.shape[1] != k.shape[1] or q.shape[1] != v.shape[1]:
        # Grouped Query Attention
        if TORCH_SUPPORT_GQA:
            out = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask, scale=scale, is_causal=is_causal, enable_gqa=True)
        else:
            # In low-version torch, `enable_gqa` is not supported.
            k = repeat(k, "b n s d -> b (n m) s d", m=q.shape[1]//k.shape[1])
            v = repeat(v, "b n s d -> b (n m) s d", m=q.shape[1]//v.shape[1])
            out = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask, scale=scale, is_causal=is_causal)
    else:
        out = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask, scale=scale, is_causal=is_causal)
    out = rearrange_out(out, out_pattern, required_out_pattern, dims)
    return out


def torch_sdpa_sliding_window(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, sliding_window: int, q_pattern="b n s d", k_pattern="b n s d", v_pattern="b n s d", out_pattern="b n s d", dims=None, scale=None):
    required_in_pattern, required_out_pattern = "b n s d", "b n s d"
    q, k, v = rearrange_qkv(q, k, v, q_pattern, k_pattern, v_pattern, required_in_pattern, dims)

    B, N, S, D = q.shape
    W = sliding_window
    chunk_size = W
    num_chunks = (S + chunk_size - 1) // chunk_size

    output = torch.empty_like(q)
    dtype = q.dtype
    device = q.device
    min_val = torch.finfo(dtype).min

    for i in range(num_chunks):
        q_start = i * chunk_size
        q_end = min(q_start + chunk_size, S)
        actual_chunk_size = q_end - q_start

        kv_start = max(0, q_start - W)
        kv_end = min(S, q_end + W)

        q_chunk = q[:, :, q_start:q_end, :]
        k_chunk = k[:, :, kv_start:kv_end, :]
        v_chunk = v[:, :, kv_start:kv_end, :]

        q_indices = torch.arange(q_start, q_end, device=device)
        k_indices = torch.arange(kv_start, kv_end, device=device)
        diff = q_indices.unsqueeze(1) - k_indices.unsqueeze(0)
        valid = diff.abs() <= W

        local_mask = torch.zeros(actual_chunk_size, kv_end - kv_start, dtype=dtype, device=device)
        local_mask.masked_fill_(~valid, min_val)
        local_mask = local_mask.unsqueeze(0).unsqueeze(0)

        out_chunk = torch_sdpa(
            q_chunk, k_chunk, v_chunk, attn_mask=local_mask, scale=scale
        )
        output[:, :, q_start:q_end, :] = out_chunk

    output = rearrange_out(output, out_pattern, required_out_pattern, dims)
    return output


def flash_attention_3(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, q_pattern="b n s d", k_pattern="b n s d", v_pattern="b n s d", out_pattern="b n s d", dims=None, scale=None, is_causal=False, window_size=None):
    required_in_pattern, required_out_pattern= "b s n d", "b s n d"
    q, k, v = rearrange_qkv(q, k, v, q_pattern, k_pattern, v_pattern, required_in_pattern, dims)
    window_size = (window_size, window_size) if window_size is not None else (-1, -1)
    out = flash_attn_interface.flash_attn_func(q, k, v, softmax_scale=scale, window_size=window_size)
    if isinstance(out, tuple):
        out = out[0]
    out = rearrange_out(out, out_pattern, required_out_pattern, dims)
    return out


def flash_attention_2(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, q_pattern="b n s d", k_pattern="b n s d", v_pattern="b n s d", out_pattern="b n s d", dims=None, scale=None, is_causal=False, window_size=None):
    required_in_pattern, required_out_pattern= "b s n d", "b s n d"
    q, k, v = rearrange_qkv(q, k, v, q_pattern, k_pattern, v_pattern, required_in_pattern, dims)
    window_size = (window_size, window_size) if window_size is not None else (-1, -1)
    out = flash_attn.flash_attn_func(q, k, v, softmax_scale=scale, causal=is_causal, window_size=window_size)
    out = rearrange_out(out, out_pattern, required_out_pattern, dims)
    return out


def sage_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, q_pattern="b n s d", k_pattern="b n s d", v_pattern="b n s d", out_pattern="b n s d", dims=None, scale=None):
    required_in_pattern, required_out_pattern= "b n s d", "b n s d"
    q, k, v = rearrange_qkv(q, k, v, q_pattern, k_pattern, v_pattern, required_in_pattern, dims)
    out = sageattn(q, k, v, sm_scale=scale)
    out = rearrange_out(out, out_pattern, required_out_pattern, dims)
    return out


def xformers_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, q_pattern="b n s d", k_pattern="b n s d", v_pattern="b n s d", out_pattern="b n s d", dims=None, scale=None):
    required_in_pattern, required_out_pattern= "b s n d", "b s n d"
    q, k, v = rearrange_qkv(q, k, v, q_pattern, k_pattern, v_pattern, required_in_pattern, dims)
    out = xops.memory_efficient_attention(q, k, v, scale=scale)
    out = rearrange_out(out, out_pattern, required_out_pattern, dims)
    return out


def attention_forward(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, q_pattern="b n s d", k_pattern="b n s d", v_pattern="b n s d", out_pattern="b n s d", dims=None, attn_mask=None, scale=None, is_causal=False, compatibility_mode=False, window_size=None):
    if compatibility_mode or (attn_mask is not None) or ATTENTION_IMPLEMENTATION == "torch":
        if window_size is None:
            return torch_sdpa(q, k, v, q_pattern, k_pattern, v_pattern, out_pattern, dims, attn_mask=attn_mask, scale=scale, is_causal=is_causal)
        else:
            # Sliding Window Attention is not compatible with `is_causal` and `attn_mask`.
            assert is_causal == False and attn_mask is None
            return torch_sdpa_sliding_window(q, k, v, window_size, q_pattern, k_pattern, v_pattern, out_pattern, dims, scale=scale)
    elif ATTENTION_IMPLEMENTATION == "flash_attention_3":
        return flash_attention_3(q, k, v, q_pattern, k_pattern, v_pattern, out_pattern, dims, scale=scale, is_causal=is_causal, window_size=window_size)
    elif ATTENTION_IMPLEMENTATION == "flash_attention_2":
        return flash_attention_2(q, k, v, q_pattern, k_pattern, v_pattern, out_pattern, dims, scale=scale, is_causal=is_causal, window_size=window_size)
    elif ATTENTION_IMPLEMENTATION == "sage_attention":
        if window_size is not None or is_causal: return attention_forward(q, k, v, q_pattern, k_pattern, v_pattern, out_pattern, dims, attn_mask, scale, is_causal, compatibility_mode=True, window_size=window_size)
        return sage_attention(q, k, v, q_pattern, k_pattern, v_pattern, out_pattern, dims, scale=scale)
    elif ATTENTION_IMPLEMENTATION == "xformers":
        if window_size is not None or is_causal: return attention_forward(q, k, v, q_pattern, k_pattern, v_pattern, out_pattern, dims, attn_mask, scale, is_causal, compatibility_mode=True, window_size=window_size)
        return xformers_attention(q, k, v, q_pattern, k_pattern, v_pattern, out_pattern, dims, scale=scale)
    else:
        raise NotImplementedError("No available attention implementation.")


# ─────────────────────────────────────────────────────────────────────────────
# Variable-length (packed-sequence) attention.
#
# Used by NaViT-style sample packing where multiple samples of different lengths
# are concatenated along the sequence dim into ONE batch row (B=1, S=sum(S_i))
# and `cu_seqlens` records the [0, S_0, S_0+S_1, ...] boundaries so attention
# blocks across samples are forbidden.
#
# These entries are SEPARATE from `attention_forward` — they are NOT called
# unless the upstream model explicitly opts in (e.g. HiDream DiT only takes the
# varlen path when its forward receives a non-None `cu_seqlens`). When unused,
# all existing inference / training code paths are byte-identical to the legacy
# implementation in `attention_legacy.py`.
#
# Q / K / V layout convention here: `(total_tokens, n_heads, head_dim)` —
# the native flash-attn varlen layout. The caller (e.g. _custom_flash_attn in
# hidream_o1_image_dit.py) is responsible for unpadding to this shape and
# repadding the output afterwards.
# ─────────────────────────────────────────────────────────────────────────────

def flash_attention_2_varlen(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, scale=None, is_causal=False, window_size=None):
    """flash-attn 2 varlen wrapper. q/k/v: (total, n_heads, head_dim)."""
    if not FLASH_ATTN_2_AVAILABLE:
        raise RuntimeError("flash_attention_2_varlen requires the `flash_attn` package.")
    ws = (window_size, window_size) if window_size is not None else (-1, -1)
    return flash_attn.flash_attn_varlen_func(
        q, k, v,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=int(max_seqlen_q),
        max_seqlen_k=int(max_seqlen_k),
        softmax_scale=scale,
        causal=is_causal,
        window_size=ws,
    )


def flash_attention_3_varlen(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, scale=None, is_causal=False, window_size=None):
    """flash-attn 3 varlen wrapper. q/k/v: (total, n_heads, head_dim)."""
    if not FLASH_ATTN_3_AVAILABLE:
        raise RuntimeError("flash_attention_3_varlen requires `flash_attn_interface` (FA3).")
    ws = (window_size, window_size) if window_size is not None else (-1, -1)
    out = flash_attn_interface.flash_attn_varlen_func(
        q, k, v,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=int(max_seqlen_q),
        max_seqlen_k=int(max_seqlen_k),
        softmax_scale=scale,
        causal=is_causal,
        window_size=ws,
    )
    if isinstance(out, tuple):
        out = out[0]
    return out


def _torch_sdpa_varlen(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, scale=None, is_causal=False):
    """Pure-torch reference for varlen attention — used when flash_attn is
    unavailable. Loops over samples (slow); intended as a fallback only.
    q/k/v: (total, n_heads, head_dim) -> out: (total, n_heads, head_dim)."""
    outs = []
    B = cu_seqlens_q.numel() - 1
    for i in range(B):
        qs, qe = int(cu_seqlens_q[i]), int(cu_seqlens_q[i + 1])
        ks, ke = int(cu_seqlens_k[i]), int(cu_seqlens_k[i + 1])
        # SDPA expects (B, H, S, D).
        q_i = q[qs:qe].transpose(0, 1).unsqueeze(0)
        k_i = k[ks:ke].transpose(0, 1).unsqueeze(0)
        v_i = v[ks:ke].transpose(0, 1).unsqueeze(0)
        gqa = (q_i.shape[1] != k_i.shape[1])
        if gqa and not TORCH_SUPPORT_GQA:
            rep = q_i.shape[1] // k_i.shape[1]
            k_i = repeat(k_i, "b n s d -> b (n m) s d", m=rep)
            v_i = repeat(v_i, "b n s d -> b (n m) s d", m=rep)
            gqa = False
        kwargs = {"scale": scale, "is_causal": is_causal}
        if gqa:
            kwargs["enable_gqa"] = True
        o_i = torch.nn.functional.scaled_dot_product_attention(q_i, k_i, v_i, **kwargs)
        outs.append(o_i.squeeze(0).transpose(0, 1))
    return torch.cat(outs, dim=0) if outs else q.new_zeros(q.shape)


def attention_forward_varlen(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, scale=None, is_causal=False, window_size=None):
    """Packed-sequence attention dispatcher.

    q / k / v shape: (total_tokens, n_heads, head_dim).
    cu_seqlens_*    : int32 1-D of shape (batch + 1,), cumulative seq lengths.

    Routes to FA3 → FA2 → torch fallback. Output: (total_tokens, n_heads, head_dim).
    """
    if ATTENTION_IMPLEMENTATION == "flash_attention_3" and FLASH_ATTN_3_AVAILABLE:
        return flash_attention_3_varlen(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, scale=scale, is_causal=is_causal, window_size=window_size)
    if FLASH_ATTN_2_AVAILABLE:
        return flash_attention_2_varlen(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, scale=scale, is_causal=is_causal, window_size=window_size)
    return _torch_sdpa_varlen(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, scale=scale, is_causal=is_causal)



# ─────────────────────────────────────────────────────────────────────────────
# Triton flash-attention forward for one packed sequence (on by default; opt-out GSTAR_FAST_REFINER_ATTN=0).
#
# Non-causal, (S, H, D) bf16, D in {64, 128, 256}, S a multiple of the tile.  Same
# online-softmax algorithm as flash-attn 2 (fp32 scores / statistics / accumulator,
# bf16 P for the PV product), but Hopper wgmma tiles of 128 x 64: ≈ 410 TFLOP/s at
# the refiner's shape (S 17,408, H 18, D 256) vs ≈ 325 for flash-attn 2's 64 x 64
# hdim-256 kernel on an H200.  Other rounding than flash-attn.
# ─────────────────────────────────────────────────────────────────────────────
try:
    import triton
    import triton.language as tl
    _TRITON_AVAILABLE = True
except Exception:  # pragma: no cover
    _TRITON_AVAILABLE = False


def triton_attention_enabled():
    return _TRITON_AVAILABLE and _fast_on("REFINER_ATTN")


if _TRITON_AVAILABLE:
    @triton.jit
    def _gstar_attn_fwd(Q, K, V, O, qk_scale, sqs, sqh, sks, skh, svs, svh, sos, soh, S,
                     D: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, STAGES: tl.constexpr):
        pid_m = tl.program_id(0)
        h = tl.program_id(1)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = tl.arange(0, BLOCK_N)
        offs_d = tl.arange(0, D)
        q = tl.load(Q + h * sqh + offs_m[:, None] * sqs + offs_d[None, :])
        m_i = tl.full([BLOCK_M], float("-inf"), tl.float32)
        l_i = tl.zeros([BLOCK_M], tl.float32)
        acc = tl.zeros([BLOCK_M, D], tl.float32)
        kp = K + h * skh + offs_n[None, :] * sks + offs_d[:, None]
        vp = V + h * svh + offs_n[:, None] * svs + offs_d[None, :]
        for start in tl.range(0, S, BLOCK_N, num_stages=STAGES):
            kt = tl.load(kp + start * sks)                    # [D, BLOCK_N]
            qk = tl.dot(q, kt)
            m_ij = tl.maximum(m_i, tl.max(qk, 1) * qk_scale)
            p = tl.math.exp2(qk * qk_scale - m_ij[:, None])
            alpha = tl.math.exp2(m_i - m_ij)
            l_i = l_i * alpha + tl.sum(p, 1)
            acc = acc * alpha[:, None]
            v = tl.load(vp + start * svs)                     # [BLOCK_N, D]
            acc = tl.dot(p.to(tl.bfloat16), v, acc)
            m_i = m_ij
        acc = acc / l_i[:, None]
        tl.store(O + h * soh + offs_m[:, None] * sos + offs_d[None, :], acc.to(tl.bfloat16))


def triton_attention_supported(q, k, v):
    if not _TRITON_AVAILABLE or q.dtype != torch.bfloat16 or k.dtype != q.dtype or v.dtype != q.dtype:
        return False
    if q.dim() != 3 or q.shape != k.shape or q.shape != v.shape or not q.is_cuda:
        return False
    S, H, D = q.shape
    return D in (64, 128, 256) and S % 128 == 0 and all(t.stride(-1) == 1 for t in (q, k, v))


def triton_attention(q, k, v, scale=None):
    """q/k/v: (S, H, D) one sequence, full (non-causal) attention -> (S, H, D)."""
    S, H, D = q.shape
    scale = D ** -0.5 if scale is None else scale
    o = torch.empty(S, H, D, dtype=q.dtype, device=q.device)
    bm, bn, warps, stages = (128, 64, 8, 2)
    _gstar_attn_fwd[(S // bm, H)](q, k, v, o, scale * 1.4426950408889634,
                               q.stride(0), q.stride(1), k.stride(0), k.stride(1), v.stride(0), v.stride(1),
                               o.stride(0), o.stride(1), S, D=D, BLOCK_M=bm, BLOCK_N=bn, STAGES=stages,
                               num_warps=warps)
    return o



# ─────────────────────────────────────────────────────────────────────────────
# FlashInfer's sm90 attention (FlashAttention-3 derived: TMA K/V loads, warp-specialized wgmma pipeline) for
# one packed sequence, d = 256, bf16, no mask (on by default on sm90; opt-out GSTAR_FAST_REFINER_ATTN=0).
# Built from FlashInfer's and CUTLASS's header-only sources (an installed flashinfer-python package, or
# FLASHINFER_INCLUDE_DIR / CUTLASS_INCLUDE_DIR) with torch.utils.cpp_extension when this module is imported (part
# of loading the refiner), once per machine (gstar.fast.build_extension); falls back to the Triton / flash-attn 2
# paths if the headers are missing or the build fails.  ≈ 720 TFLOP/s at the refiner's shape on an H200
# (flash-attn 2 ≈ 320); other rounding than flash-attn.
# ─────────────────────────────────────────────────────────────────────────────
FA3_CUDA_SRC = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <flashinfer/attention/hopper/attention_updater.cuh>
#include <flashinfer/attention/hopper/variant_helper.cuh>
#include <flashinfer/math.cuh>
#include <flashinfer/layout.cuh>
#include <flashinfer/cutlass_utils.cuh>
#include <flashinfer/utils.cuh>
#include <flashinfer/pos_enc.cuh>

using namespace flashinfer;
using DTypeQ = cutlass_dtype_t<nv_bfloat16>;
using DTypeKV = cutlass_dtype_t<nv_bfloat16>;
using DTypeO = cutlass_dtype_t<nv_bfloat16>;
using IdType = cutlass_dtype_t<int32_t>;

struct Params {
  using DTypeQ = ::DTypeQ;
  using DTypeKV = ::DTypeKV;
  using DTypeO = ::DTypeO;
  using IdType = ::IdType;
  DTypeQ* q_ptr;
  DTypeKV* k_ptr;
  DTypeKV* v_ptr;
  DTypeO* o_ptr;
  float* lse_ptr;
  struct AdditionalParams {
    float sm_scale;
  } additional_params;
  int64_t q_stride_n;
  int64_t k_stride_n;
  int64_t v_stride_n;
  int64_t o_stride_n;
  int64_t q_stride_h;
  int64_t k_stride_h;
  int64_t v_stride_h;
  int64_t o_stride_h;
  int qo_len;
  int kv_len;
  int head_dim;
  int num_qo_heads;
  int num_kv_heads;
  int group_size;
  int window_left;
  bool causal;
};

#include <flashinfer/attention/hopper/variants.cuh>
#include <flashinfer/attention/hopper/prefill_sm90.cuh>

namespace flashinfer {
template cudaError_t SinglePrefillWithKVCacheDispatched<256, 256, MaskMode::kNone, false, StandardAttention, Params>(
    Params& params, cudaStream_t stream);
}

torch::Tensor fa3_fwd_d256(torch::Tensor q, torch::Tensor k, torch::Tensor v, double sm_scale) {
  TORCH_CHECK(q.is_cuda() && k.is_cuda() && v.is_cuda(), "cuda tensors expected");
  TORCH_CHECK(q.scalar_type() == at::kBFloat16 && k.scalar_type() == at::kBFloat16 && v.scalar_type() == at::kBFloat16);
  TORCH_CHECK(q.dim() == 3 && k.dim() == 3 && v.dim() == 3 && q.size(2) == 256 && k.size(2) == 256 && v.size(2) == 256);
  TORCH_CHECK(q.stride(2) == 1 && k.stride(2) == 1 && v.stride(2) == 1, "last dim must be contiguous");
  auto o = torch::empty({q.size(0), q.size(1), 256}, q.options());
  Params params;
  params.q_ptr = static_cast<DTypeQ*>(q.data_ptr());
  params.k_ptr = static_cast<DTypeKV*>(k.data_ptr());
  params.v_ptr = static_cast<DTypeKV*>(v.data_ptr());
  params.o_ptr = static_cast<DTypeO*>(o.data_ptr());
  params.lse_ptr = nullptr;
  params.additional_params.sm_scale = static_cast<float>(sm_scale);
  params.q_stride_n = q.stride(0);
  params.q_stride_h = q.stride(1);
  params.k_stride_n = k.stride(0);
  params.k_stride_h = k.stride(1);
  params.v_stride_n = v.stride(0);
  params.v_stride_h = v.stride(1);
  params.o_stride_n = o.stride(0);
  params.o_stride_h = o.stride(1);
  params.qo_len = q.size(0);
  params.kv_len = k.size(0);
  params.head_dim = 256;
  params.num_qo_heads = q.size(1);
  params.num_kv_heads = k.size(1);
  params.group_size = params.num_qo_heads / params.num_kv_heads;
  params.window_left = -1;
  params.causal = false;
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  cudaError_t status = SinglePrefillWithKVCacheDispatched<256, 256, MaskMode::kNone, false, StandardAttention, Params>(params, stream);
  TORCH_CHECK(status == cudaSuccess, "fa3_fwd_d256 failed: ", cudaGetErrorString(status));
  return o;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("fa3_fwd_d256", &fa3_fwd_d256, "FlashInfer sm90 (FA3-derived) single prefill, d=256, bf16, no mask");
}
'''

_FA3 = {"mod": None, "tried": False, "error": None}


def _header_dirs():
    """Include directories of FlashInfer's sm90 attention templates and of CUTLASS / CuTe, or None.

    Searched in order: FLASHINFER_INCLUDE_DIR and CUTLASS_INCLUDE_DIR; the headers bundled with an installed
    flashinfer-python package (located without importing it)."""
    from pathlib import Path
    cands = []
    fi, cu = os.environ.get("FLASHINFER_INCLUDE_DIR"), os.environ.get("CUTLASS_INCLUDE_DIR")
    if fi and cu:
        cands.append((Path(fi), Path(cu)))
    try:
        import importlib.util
        spec = importlib.util.find_spec("flashinfer")
        if spec is not None and spec.submodule_search_locations:
            for root in spec.submodule_search_locations:
                data = Path(root) / "data"
                cands.append((data / "include", data / "cutlass" / "include"))
    except Exception:  # pragma: no cover
        pass
    for fi_dir, cu_dir in cands:
        if ((fi_dir / "flashinfer" / "attention" / "hopper" / "prefill_sm90.cuh").is_file()
                and (cu_dir / "cute").is_dir() and (cu_dir / "cutlass").is_dir()):
            dirs = [str(fi_dir), str(cu_dir)]
            util = cu_dir.parent / "tools" / "util" / "include"
            if util.is_dir():
                dirs.append(str(util))
            return dirs
    return None


def _build_fa3():
    if _FA3["tried"]:
        return _FA3["mod"]
    _FA3["tried"] = True
    try:
        from ...fast import build_extension
        inc = _header_dirs()
        if inc is None:
            raise RuntimeError("FlashInfer / CUTLASS headers not found (install flashinfer-python or set "
                               "FLASHINFER_INCLUDE_DIR and CUTLASS_INCLUDE_DIR)")
        _FA3["mod"] = build_extension(
            "gstar_fa3_ext", FA3_CUDA_SRC, ".cu", arch="9.0a", extra_include_paths=inc,
            extra_cuda_cflags=["-O3", "-std=c++17", "--expt-relaxed-constexpr", "--expt-extended-lambda",
                               "-DNDEBUG", "-U__CUDA_NO_HALF_OPERATORS__", "-U__CUDA_NO_HALF_CONVERSIONS__",
                               "-U__CUDA_NO_BFLOAT16_CONVERSIONS__", "-U__CUDA_NO_HALF2_OPERATORS__"],
            extra_cflags=["-O3", "-std=c++17"])
    except Exception as exc:  # pragma: no cover
        _FA3["error"] = f"{type(exc).__name__}: {exc}"
        _FA3["mod"] = None
        print(f"[gstar.fast] FlashInfer sm90 attention unavailable, using the Triton kernel: {_FA3['error'][:300]}", flush=True)
    return _FA3["mod"]


def fa3_attention_enabled():
    return _fast_on("REFINER_ATTN") and _sm90() and _build_fa3() is not None


def fa3_attention_supported(q, k, v):
    return (q.is_cuda and q.dtype == torch.bfloat16 and k.dtype == q.dtype and v.dtype == q.dtype
            and q.dim() == 3 and q.shape == k.shape and q.shape == v.shape and q.shape[-1] == 256
            and all(t.stride(-1) == 1 for t in (q, k, v)))


def fa3_attention(q, k, v, scale=None):
    """q/k/v: (S, H, 256), one sequence, full attention -> (S, H, 256)."""
    D = q.shape[-1]
    return _FA3["mod"].fa3_fwd_d256(q, k, v, float(D ** -0.5 if scale is None else scale))


if _fast_on("REFINER_ATTN") and _sm90():
    _build_fa3()          # at import: the build is part of loading the refiner
