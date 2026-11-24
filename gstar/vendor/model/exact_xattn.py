"""Bit-exact fused decode attention for the lean CFG decode step (on by default on sm90; opt-out GSTAR_FAST_LM_ATTN=0).

The reference decode attention per layer (math SDPA on the lean fp32 caches) is four kernels:
  attn = matmul(q_s, KS[:, :, :L]^T)          cuBLAS gemv2T for L >= 97 (gemvx below)
  attn.add_(mask); attn = softmax(attn, -1)   ATen elementwise add + softmax_warp_forward (L <= 1024)
  out  = matmul(attn, VF[:, :, :L])           cuBLAS gemvx
Their summation orders were measured with FPRev-style probes (±2^40 at a pair of positions, ones elsewhere: the
result is the size of the subtree joining the pair) and confirmed bitwise on random data for every L:
  gemv2T: 16 lanes t, lane t = FMA chain over d = t, t+16, ..., t+112 (from 0), lane partials added in order;
  softmax_warp_forward: element lane + 32*it, per-lane max / sum over it (from 0), xor butterfly 16..1, x / sum,
    expf (libdevice / CUDA expf, bitwise equal to torch.exp on all 2^31 inputs <= 0);
  gemvx: L <= 94: 2 lanes, contiguous chunks of ceil(L/2); L <= 256: 4 lanes, ceil(L/4); L > 256: 4 lanes, lane t
    takes [256k + 64t, 256k + 64t + 64) of every 256-tile; FMA chains, partials combined (a0+a2)+(a1+a3) / a0+a1.
This extension computes the same values in two kernels from the bf16 per-kv-head caches (KS = float(k) * s and
VF = float(v) on the load, exact): 8x fewer bytes than the expanded fp32 caches.  Used for 97 <= L <= 1024
(gemv2T + warp softmax); other L keep the reference ops.  Built at import (part of loading, fresh temp dir).
"""
import os
from ...fast import enabled as _fast_on, sm90 as _sm90

import torch

XATTN_SRC = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <math.h>

// Exact decode attention for the lean CFG decode step (97 <= L <= 1024): reproduces
//   attn = matmul(q_s, KS[:, :, :L]^T)  (cuBLAS gemv2T: 16 lanes, FMA over d = t + 16c, lanes added in order)
//   attn += mask;  attn = softmax(attn, -1)  (ATen softmax_warp_forward)
//   out = matmul(attn, VF[:, :, :L])  (cuBLAS gemvx: P = 2 / 4 lanes, contiguous ceil(L/P) chunks for L <= 256,
//   lane t takes [256k + 64t, +64) above; FMA chains; lane partials combined as the xor butterfly)
// with KS = float(KB) * s and VF = float(VB) computed on the load from the bf16 per-kv-head caches.

__global__ void __launch_bounds__(128) qk_mask_kernel(const float* __restrict__ q_s, const __nv_bfloat16* __restrict__ kb,
                               const __nv_bfloat16* __restrict__ mask, float* __restrict__ sc, const int* __restrict__ Lp,
                               float s, int HQ, int REP, long sq_b, long sq_h, long sk_b, long sk_h, long sk_l,
                               long sm_b, long ss_b, long ss_h) {
  // one thread per position l of one (batch, kv head): the whole K row in registers, the REP (= 4) query heads'
  // scores from it, each with gemv2T's order (16 lanes t: FMA over d = t + 16c, lanes added in order)
  const int bk = blockIdx.x;
  const int HKV = HQ / REP;
  const int b = bk / HKV, kvh = bk % HKV;
  const int L = *Lp;
  const int l = blockIdx.y * 128 + threadIdx.x;
  __shared__ float qs[4][128];
  for (int e = threadIdx.x; e < 4 * 128; e += 128)
    qs[e >> 7][e & 127] = q_s[b * sq_b + (kvh * REP + (e >> 7)) * sq_h + (e & 127)];
  __syncthreads();
  if (l >= L) return;
  const uint4* krow = reinterpret_cast<const uint4*>(kb + b * sk_b + kvh * sk_h + (long)l * sk_l);
  float k[128];
#pragma unroll
  for (int w = 0; w < 16; ++w) {
    uint4 u = krow[w];
    const __nv_bfloat162* h2 = reinterpret_cast<const __nv_bfloat162*>(&u);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      float2 f = __bfloat1622float2(h2[j]);
      k[w * 8 + 2 * j] = __fmul_rn(f.x, s);
      k[w * 8 + 2 * j + 1] = __fmul_rn(f.y, s);
    }
  }
  const float mk = __bfloat162float(mask[b * sm_b + l]);     // the reference's fp32 + bf16 promotion (exact)
#pragma unroll
  for (int g = 0; g < 4; ++g) {
    float tot = 0.f;
#pragma unroll
    for (int t = 0; t < 16; ++t) {
      float acc = 0.f;
#pragma unroll
      for (int c = 0; c < 8; ++c) acc = __fmaf_rn(qs[g][t + 16 * c], k[t + 16 * c], acc);
      tot = (t == 0) ? acc : __fadd_rn(tot, acc);
    }
    sc[b * ss_b + (kvh * REP + g) * ss_h + l] = __fadd_rn(tot, mk);
  }
}

__global__ void softmax_pv_kernel(const float* __restrict__ sc, const __nv_bfloat16* __restrict__ vb,
                                  float* __restrict__ out, const int* __restrict__ Lp, int HQ, int REP,
                                  long ss_b, long ss_h, long sv_b, long sv_h, long sv_l, long so_b, long so_h) {
  const int bh = blockIdx.x, b = bh / HQ, h = bh % HQ;
  const int L = *Lp;
  const int d = blockIdx.y * 32 + threadIdx.x;      // 32 d per block
  const int lane = threadIdx.y;                       // 0..3
  __shared__ float p[1024];
  __shared__ float red[4][32];
  // ---- softmax: warp 0 of the block (threadIdx.y == 0), ATen softmax_warp_forward arithmetic ----
  if (threadIdx.y == 0) {
    const int ln = threadIdx.x;
    const float* row = sc + b * ss_b + h * ss_h;
    float el[32];
#pragma unroll
    for (int it = 0; it < 32; ++it) {
      const int idx = ln + it * 32;
      el[it] = idx < L ? row[idx] : -INFINITY;
    }
    float mx = el[0];
#pragma unroll
    for (int it = 0; it < 32; ++it) mx = (mx > el[it]) ? mx : el[it];
#pragma unroll
    for (int off = 16; off > 0; off /= 2) {
      const float o = __shfl_xor_sync(0xffffffffu, mx, off, 32);
      mx = mx < o ? o : mx;
    }
    float sum = 0.f;
#pragma unroll
    for (int it = 0; it < 32; ++it) {
      el[it] = expf(el[it] - mx);
      sum += el[it];
    }
#pragma unroll
    for (int off = 16; off > 0; off /= 2) sum = sum + __shfl_xor_sync(0xffffffffu, sum, off, 32);
#pragma unroll
    for (int it = 0; it < 32; ++it) {
      const int idx = ln + it * 32;
      if (idx < L) p[idx] = el[it] / sum;
    }
  }
  __syncthreads();
  // ---- PV chains: thread (d, lane) ----
  const __nv_bfloat16* v = vb + b * sv_b + (h / REP) * sv_h + d;
  float acc = 0.f;
  const bool small = L <= 94;
  if (L <= 256) {
    const int P = small ? 2 : 4;
    const int csz = small ? (L + 1) / 2 : (L + 3) / 4;
    if (lane < P) {
      const int l0 = lane * csz;
      const int l1 = min(l0 + csz, L);
      int l = l0;
      for (; l + 8 <= l1; l += 8) {
        float pv[8], vv[8];
#pragma unroll
        for (int j = 0; j < 8; ++j) { pv[j] = p[l + j]; vv[j] = __bfloat162float(v[(long)(l + j) * sv_l]); }
#pragma unroll
        for (int j = 0; j < 8; ++j) acc = __fmaf_rn(pv[j], vv[j], acc);
      }
      for (; l < l1; ++l) acc = __fmaf_rn(p[l], __bfloat162float(v[(long)l * sv_l]), acc);
    }
  } else {
    for (int k0 = 0; k0 < L; k0 += 256) {
      const int l0 = k0 + 64 * lane;
      const int l1 = min(l0 + 64, L);
      int l = l0;
      for (; l + 8 <= l1; l += 8) {
        float pv[8], vv[8];
#pragma unroll
        for (int j = 0; j < 8; ++j) { pv[j] = p[l + j]; vv[j] = __bfloat162float(v[(long)(l + j) * sv_l]); }
#pragma unroll
        for (int j = 0; j < 8; ++j) acc = __fmaf_rn(pv[j], vv[j], acc);
      }
      for (; l < l1; ++l) acc = __fmaf_rn(p[l], __bfloat162float(v[(long)l * sv_l]), acc);
    }
  }
  red[lane][threadIdx.x] = acc;
  __syncthreads();
  if (lane == 0) {
    float r;
    if (small) r = __fadd_rn(red[0][threadIdx.x], red[1][threadIdx.x]);
    else r = __fadd_rn(__fadd_rn(red[0][threadIdx.x], red[2][threadIdx.x]), __fadd_rn(red[1][threadIdx.x], red[3][threadIdx.x]));
    out[b * so_b + h * so_h + d] = r;
  }
}

void exact_decode_attn(torch::Tensor q_s, torch::Tensor kb, torch::Tensor vb, torch::Tensor mask, torch::Tensor L_dev,
                       torch::Tensor sc, torch::Tensor out, double s) {
  const int B = q_s.size(0), HQ = q_s.size(1), D = q_s.size(3), HKV = kb.size(1);
  TORCH_CHECK(D == 128 && q_s.scalar_type() == at::kFloat && kb.scalar_type() == at::kBFloat16 && vb.scalar_type() == at::kBFloat16);
  auto stream = at::cuda::getCurrentCUDAStream();
  TORCH_CHECK(HQ / HKV == 4 && kb.stride(2) == 128 && kb.stride(3) == 1, "GQA 4, contiguous rows expected");
  dim3 ga(B * HKV, 8), ba(128);           // 8 chunks of 128 positions (L <= 1024)
  TORCH_CHECK(mask.scalar_type() == at::kBFloat16);
  qk_mask_kernel<<<ga, ba, 0, stream>>>(q_s.data_ptr<float>(), (const __nv_bfloat16*)kb.data_ptr(), (const __nv_bfloat16*)mask.data_ptr(),
      sc.data_ptr<float>(), L_dev.data_ptr<int>(), (float)s, HQ, HQ / HKV, q_s.stride(0), q_s.stride(1), kb.stride(0),
      kb.stride(1), kb.stride(2), mask.stride(0), sc.stride(0), sc.stride(1));
  dim3 gb(B * HQ, D / 32), bb(32, 4);
  softmax_pv_kernel<<<gb, bb, 0, stream>>>(sc.data_ptr<float>(), (const __nv_bfloat16*)vb.data_ptr(), out.data_ptr<float>(),
      L_dev.data_ptr<int>(), HQ, HQ / HKV, sc.stride(0), sc.stride(1), vb.stride(0), vb.stride(1), vb.stride(2),
      out.stride(0), out.stride(1));
}

// The DSpark head's per-slot attention (one causal layer, 16 heads x 128, n = slot + 1 <= 16 positions):
//   K32[:, :, slot] = float(k_new); V32[:, :, slot] = float(v_new)
//   scores = matmul(float(q), K32[:, :, :n]^T)    cuBLAS gemvx small-N: 16 lanes x 8 contiguous d, FMA, xor 8,4,2,1
//   scores = scores * scale; p = softmax(scores)   ATen softmax_warp_forward, warp width next_pow2(n), 1 element / lane
//   attn = matmul(p, V32[:, :, :n]).to(bf16)       gemvx: n <= 8: lane t: l = t, t+4 (rounded products added);
//                                                  9..16: per 8-tile pair (2t, 2t+1): fma(p1, v1, p0*v0), added;
//                                                  lanes (a0 + a2) + (a1 + a3)
//   out [B, 1, H*128] = attn.transpose(1, 2).reshape(...)
__global__ void __launch_bounds__(128) head_attn_kernel(const __nv_bfloat16* __restrict__ q, const __nv_bfloat16* __restrict__ kn,
    const __nv_bfloat16* __restrict__ vn, float* __restrict__ K32, float* __restrict__ V32, __nv_bfloat16* __restrict__ out,
    int slot, float scale, int H, long sq_b, long sk_b, long c_b, long c_h, long c_l, long so_b) {
  const int bh = blockIdx.x, b = bh / H, h = bh % H;
  const int d = threadIdx.x;              // 0..127
  const int n = slot + 1;
  __shared__ float qs[128];
  __shared__ float ks[16][129];
  __shared__ float chains[16][17];
  __shared__ float sc[16];
  __shared__ float ps[16];
  float* Kc = K32 + b * c_b + h * c_h;
  float* Vc = V32 + b * c_b + h * c_h;
  // the new cache rows (exact bf16 -> fp32 copies)
  const float knew = __bfloat162float(kn[b * sk_b + h * 128 + d]);
  const float vnew = __bfloat162float(vn[b * sk_b + h * 128 + d]);
  Kc[slot * c_l + d] = knew;
  Vc[slot * c_l + d] = vnew;
  qs[d] = __bfloat162float(q[b * sq_b + h * 128 + d]);
  for (int l = 0; l < n; ++l) ks[l][d] = (l == slot) ? knew : Kc[l * c_l + d];
  __syncthreads();
  // QK chains: (l, t) for l < n, t < 16: 256 chains over 128 threads
  for (int c = threadIdx.x; c < 16 * 16; c += 128) {
    const int l = c >> 4, t = c & 15;
    if (l < n) {
      float a = 0.f;
#pragma unroll
      for (int j = 0; j < 8; ++j) a = __fmaf_rn(qs[8 * t + j], ks[l][8 * t + j], a);
      chains[l][t] = a;
    }
  }
  __syncthreads();
  if (threadIdx.x < n) {
    const int l = threadIdx.x;
    float v1[8], v2[4], v3[2];
#pragma unroll
    for (int j = 0; j < 8; ++j) v1[j] = __fadd_rn(chains[l][j], chains[l][j + 8]);
#pragma unroll
    for (int j = 0; j < 4; ++j) v2[j] = __fadd_rn(v1[j], v1[j + 4]);
#pragma unroll
    for (int j = 0; j < 2; ++j) v3[j] = __fadd_rn(v2[j], v2[j + 2]);
    sc[l] = __fmul_rn(__fadd_rn(v3[0], v3[1]), scale);
  }
  __syncthreads();
  // softmax: warp 0, width W = next_pow2(n), lane i holds element i (or -inf)
  if (threadIdx.x < 32) {
    int W = 1;
    while (W < n) W <<= 1;
    const int ln = threadIdx.x;
    float x = (ln < n) ? sc[ln] : -INFINITY;
    float mx = x;
    for (int off = W / 2; off > 0; off /= 2) {
      const float o = __shfl_xor_sync(0xffffffffu, mx, off, W);
      mx = mx < o ? o : mx;
    }
    const float e = (ln < n) ? expf(x - mx) : 0.f;
    float sum = 0.f + e;
    for (int off = W / 2; off > 0; off /= 2) sum = sum + __shfl_xor_sync(0xffffffffu, sum, off, W);
    if (ln < n) ps[ln] = e / sum;
  }
  __syncthreads();
  // PV for d: 4 lanes in this thread
  float a[4] = {0.f, 0.f, 0.f, 0.f};
  if (n <= 8) {
#pragma unroll
    for (int t = 0; t < 4; ++t)
      for (int l = t; l < n; l += 4) {
        const float vv = (l == slot) ? vnew : Vc[l * c_l + d];
        a[t] = __fadd_rn(a[t], __fmul_rn(ps[l], vv));
      }
  } else {
#pragma unroll
    for (int t = 0; t < 4; ++t)
      for (int k0 = 0; k0 < n; k0 += 8) {
        const int l0 = k0 + 2 * t, l1 = l0 + 1;
        if (l0 >= n) continue;
        const float v0 = (l0 == slot) ? vnew : Vc[l0 * c_l + d];
        float tmp = __fmul_rn(ps[l0], v0);
        if (l1 < n) {
          const float v1_ = (l1 == slot) ? vnew : Vc[l1 * c_l + d];
          tmp = __fmaf_rn(ps[l1], v1_, tmp);
        }
        a[t] = __fadd_rn(a[t], tmp);
      }
  }
  const float r = __fadd_rn(__fadd_rn(a[0], a[2]), __fadd_rn(a[1], a[3]));
  out[b * so_b + h * 128 + d] = __float2bfloat16_rn(r);
}

void head_attn(torch::Tensor q, torch::Tensor kn, torch::Tensor vn, torch::Tensor K32, torch::Tensor V32, torch::Tensor out,
               int64_t slot, double scale) {
  const int B = K32.size(0), H = K32.size(1);
  TORCH_CHECK(K32.size(3) == 128 && K32.stride(3) == 1 && slot >= 0 && slot < 16 && slot < K32.size(2));
  head_attn_kernel<<<B * H, 128, 0, at::cuda::getCurrentCUDAStream()>>>(
      (const __nv_bfloat16*)q.data_ptr(), (const __nv_bfloat16*)kn.data_ptr(), (const __nv_bfloat16*)vn.data_ptr(),
      K32.data_ptr<float>(), V32.data_ptr<float>(), (__nv_bfloat16*)out.data_ptr(), (int)slot, (float)scale, H,
      q.stride(0), kn.stride(0), K32.stride(0), K32.stride(1), K32.stride(2), out.stride(0));
}

// argmax(softmax(x) / q) for one row, bitwise as the reference ops:
//   torch.softmax (cunn_SoftMaxForward, 1024 threads, ILP 4: per-thread sum over float4 groups at tid + k*1024 then the
//   tail; warp shfl-down tree; warp 0 tree over the warp sums; exp(x - max) / sum), torch.div(probs, q), argmax (first
//   index of the maximum).
__device__ __forceinline__ float warp_sum_down_sm(float v) {
#pragma unroll
  for (int off = 16; off > 0; off /= 2) v = v + __shfl_down_sync(0xffffffffu, v, off, 32);
  return v;
}

__global__ void __launch_bounds__(1024) softmax_div_argmax_kernel(const float* __restrict__ x, const float* __restrict__ q,
                                                                   int64_t* __restrict__ out, int N) {
  __shared__ float red[32];
  __shared__ float bval[32];
  __shared__ int bidx[32];
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int main_n = (N / 4096) * 4096;
  // max (order-independent)
  float mx = -INFINITY;
  for (int k = 0; 4 * (tid + k * 1024) < main_n; ++k) {
    const float4 v = reinterpret_cast<const float4*>(x)[tid + k * 1024];
    mx = fmaxf(mx, fmaxf(fmaxf(v.x, v.y), fmaxf(v.z, v.w)));
  }
  for (int i = main_n + tid; i < N; i += 1024) mx = fmaxf(mx, x[i]);
#pragma unroll
  for (int off = 16; off > 0; off /= 2) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, off, 32));
  if (lane == 0) red[warp] = mx;
  __syncthreads();
  if (warp == 0) {
    float m = red[lane];
#pragma unroll
    for (int off = 16; off > 0; off /= 2) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, off, 32));
    if (lane == 0) red[0] = m;
  }
  __syncthreads();
  const float m = red[0];
  __syncthreads();
  // sum in cunn_SoftMaxForward's order
  float s = 0.f;
  for (int k = 0; 4 * (tid + k * 1024) < main_n; ++k) {
    const float4 v = reinterpret_cast<const float4*>(x)[tid + k * 1024];
    s = s + expf(v.x - m); s = s + expf(v.y - m); s = s + expf(v.z - m); s = s + expf(v.w - m);
  }
  for (int i = main_n + tid; i < N; i += 1024) s = s + expf(x[i] - m);
  s = warp_sum_down_sm(s);
  if (lane == 0) red[warp] = s;
  __syncthreads();
  if (warp == 0) {
    float t = warp_sum_down_sm(red[lane]);
    if (lane == 0) red[0] = t;
  }
  __syncthreads();
  const float tot = red[0];
  // probs / q, argmax (largest value, smallest index)
  float best = -INFINITY;
  int bi = 0x7fffffff;
  for (int i = tid; i < N; i += 1024) {
    const float p = expf(x[i] - m) / tot;
    const float r = p / q[i];
    if (r > best || (r == best && i < bi)) { best = r; bi = i; }
  }
#pragma unroll
  for (int off = 16; off > 0; off /= 2) {
    const float ov = __shfl_xor_sync(0xffffffffu, best, off, 32);
    const int oi = __shfl_xor_sync(0xffffffffu, bi, off, 32);
    if (ov > best || (ov == best && oi < bi)) { best = ov; bi = oi; }
  }
  if (lane == 0) { bval[warp] = best; bidx[warp] = bi; }
  __syncthreads();
  if (warp == 0) {
    best = bval[lane]; bi = bidx[lane];
#pragma unroll
    for (int off = 16; off > 0; off /= 2) {
      const float ov = __shfl_xor_sync(0xffffffffu, best, off, 32);
      const int oi = __shfl_xor_sync(0xffffffffu, bi, off, 32);
      if (ov > best || (ov == best && oi < bi)) { best = ov; bi = oi; }
    }
    if (lane == 0) out[0] = bi;
  }
}

torch::Tensor softmax_div_argmax(torch::Tensor x, torch::Tensor q) {
  TORCH_CHECK(x.is_contiguous() && q.is_contiguous() && x.scalar_type() == at::kFloat && q.scalar_type() == at::kFloat);
  TORCH_CHECK(x.dim() == 2 && x.size(0) == 1 && (reinterpret_cast<uintptr_t>(x.data_ptr()) % 16) == 0);
  auto out = torch::empty({1}, x.options().dtype(at::kLong));
  softmax_div_argmax_kernel<<<1, 1024, 0, at::cuda::getCurrentCUDAStream()>>>(x.data_ptr<float>(), q.data_ptr<float>(),
                                                                             out.data_ptr<int64_t>(), (int)x.size(1));
  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("exact_decode_attn", &exact_decode_attn);
  m.def("head_attn", &head_attn);
  m.def("softmax_div_argmax", &softmax_div_argmax);
}
'''

_X = {"mod": None, "tried": False}


def _build():
    if _X["tried"]:
        return _X["mod"]
    _X["tried"] = True
    try:
        from ...fast import build_extension
        _X["mod"] = build_extension("gstar_xattn", XATTN_SRC, ".cu", arch="9.0", extra_cuda_cflags=["-O3"],
                                    extra_cflags=["-O2"])
    except Exception as exc:  # pragma: no cover
        print(f"[gstar.fast] fused attention kernels unavailable, reference ops kept: {type(exc).__name__}: {str(exc)[:300]}", flush=True)
        _X["mod"] = None
    return _X["mod"]


def xattn_enabled():
    return _fast_on("LM_ATTN") and _sm90() and _build() is not None


def hattn_enabled():
    """The DSpark head's per-slot attention (K32/V32 rows, QK, scale, softmax, PV, bf16 cast,
    transpose) in one kernel - gemvx small-N orders measured like the decode attention's (QK: 16 lanes x 8
    contiguous d, FMA, xor 8,4,2,1; PV: n <= 8 lane t = l = t, t+4 with rounded products, 9..16 per 8-tile pair
    fma(p1, v1, p0*v0); lanes (a0+a2)+(a1+a3)); warp softmax of width next_pow2(n).  Bitwise for every n 1..16."""
    return _fast_on("LM_ATTN") and _sm90() and _build() is not None


_HCHECK = {}


def head_attn(qx, kx, vx, K32, V32, slot, scale, ref_fn):
    """attn [rows, 1, H*128] bf16 of the head's slot; the first call per slot also runs ref_fn() (the reference
    ops, eager) and keeps the reference if the bits differ."""
    out = torch.empty_like(qx)
    if slot not in _HCHECK:
        ref = ref_fn()                         # writes the same K32 / V32 rows
        _X["mod"].head_attn(qx, kx, vx, K32, V32, out, int(slot), float(scale))
        _HCHECK[slot] = bool(torch.equal(out, ref))
        if not _HCHECK[slot]:
            print(f"[gstar.fast] head attention: live check failed for slot {slot}, reference ops kept", flush=True)
            return ref
        return out
    if not _HCHECK[slot]:
        return ref_fn()
    _X["mod"].head_attn(qx, kx, vx, K32, V32, out, int(slot), float(scale))
    return out


XATTN_LMIN, XATTN_LMAX = 97, 1024


def xattn(q_s, kb, vb, mask, L_dev, sc_buf, out, scale):
    _X["mod"].exact_decode_attn(q_s, kb, vb, mask, L_dev, sc_buf, out, float(scale))


def sample2_enabled():
    """argmax(softmax(scores) / q) of the head's draw in one kernel (ATen cunn_SoftMaxForward's order:
    per-thread float4 groups at tid + k*1024 then the tail, warp shfl-down trees, exp(x - max) / sum; IEEE division;
    first index of the maximum) after the reference's exponential_ draw; bitwise 2000/2000 on random rows."""
    return _fast_on("LM_KERNELS") and _sm90() and _build() is not None


_SCHECK = {}


def softmax_div_argmax(scores, q):
    out = _X["mod"].softmax_div_argmax(scores, q)
    key = tuple(scores.shape)          # the softmax's reduction order depends on the row length
    if key not in _SCHECK:
        probs = torch.softmax(scores, dim=-1)
        r = torch.div(probs, q)
        ref = torch.argmax(r, dim=-1, keepdim=True).squeeze(1)
        _SCHECK[key] = bool(torch.equal(out, ref))
        if not _SCHECK[key]:
            print("[gstar.fast] fused draw: live check failed, reference ops kept", flush=True)
            return ref
    if not _SCHECK[key]:
        probs = torch.softmax(scores, dim=-1)
        torch.div(probs, q, out=q)
        return torch.argmax(q, dim=-1, keepdim=True).squeeze(1)
    return out


if (_fast_on("LM_ATTN") or _fast_on("LM_KERNELS")) and _sm90():
    _build()
