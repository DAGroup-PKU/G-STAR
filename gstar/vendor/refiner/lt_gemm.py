"""Large-tile cuBLASLt GEMM for the refiner's DiT linears (on by default on sm90; opt-out GSTAR_FAST_REFINER_GEMM=0).

On an H200 the refiner phase runs at the GPU's 700 W power cap (clock ≈ 1.55–1.6 GHz instead of 1.98), so a GEMM's energy per
FLOP sets its speed.  cuBLAS's heuristic gives the QKV / o-proj / w1 / w3 linears (M = 17,408, K = 4,608) the
128×128 sm90 kernel on 112 CTAs, w2 (K = 12,288) the 128×256 one on all 132 SMs, which moves less operand data per
FLOP.  Under sustained load the 128×256 kernel runs the QKV shape in 3.34 ms instead of
3.50 ms.  This module asks cuBLASLt for that kernel for every DiT linear: the problem's own heuristic list first,
else the algorithm the heuristic picks for the w2-like problem (M, 4608, 12288), validated with
cublasLtMatmulAlgoCheck.  Same fp32 accumulation over K in the same order (no split-K): the outputs are
bitwise identical to F.linear for all four shapes (checked).  Built when the module is imported (part of loading), once per machine (gstar.fast.build_extension).
"""
import os
from ...fast import enabled as _fast_on, sm90 as _sm90

import torch

LT_CUDA_SRC = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cublasLt.h>
#include <cuda_bf16.h>
#include <map>
#include <tuple>

#define LT_CHECK(x) do { cublasStatus_t st = (x); TORCH_CHECK(st == CUBLAS_STATUS_SUCCESS, "cublasLt error ", (int)st, " at ", #x); } while (0)

static cublasLtHandle_t g_handle = nullptr;
static void* g_ws = nullptr;
static size_t g_ws_size = 64ull << 20;

struct Plan {
  cublasLtMatmulDesc_t op;
  cublasLtMatrixLayout_t a, b, c;
  cublasLtMatmulAlgo_t algo;
  int tile;
};
static std::map<std::tuple<int64_t,int64_t,int64_t>, Plan> g_plans;

static int algo_tile(const cublasLtMatmulAlgo_t& al) {
  int tile = 0; size_t w = 0;
  cublasLtMatmulAlgoConfigGetAttribute(&al, CUBLASLT_ALGO_CONFIG_TILE_ID, &tile, sizeof(tile), &w);
  return tile;
}

// Largest-tile algorithm (128x256 / 256x128 sm90 kernels) for C = X W^T: the problem's own heuristic list first, then
// the problem's top algorithm with its tile set to 128x256 (checked with cublasLtMatmulAlgoCheck); -1 if none.
static Plan& get_plan(int64_t M, int64_t N, int64_t K) {
  auto key = std::make_tuple(M, N, K);
  auto it = g_plans.find(key);
  if (it != g_plans.end()) return it->second;
  if (!g_handle) {
    LT_CHECK(cublasLtCreate(&g_handle));
    TORCH_CHECK(cudaMalloc(&g_ws, g_ws_size) == cudaSuccess);
  }
  Plan p;
  LT_CHECK(cublasLtMatmulDescCreate(&p.op, CUBLAS_COMPUTE_32F, CUDA_R_32F));
  cublasOperation_t ta = CUBLAS_OP_T, tb = CUBLAS_OP_N;
  LT_CHECK(cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_TRANSA, &ta, sizeof(ta)));
  LT_CHECK(cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_TRANSB, &tb, sizeof(tb)));
  LT_CHECK(cublasLtMatrixLayoutCreate(&p.a, CUDA_R_16BF, K, N, K));
  LT_CHECK(cublasLtMatrixLayoutCreate(&p.b, CUDA_R_16BF, K, M, K));
  LT_CHECK(cublasLtMatrixLayoutCreate(&p.c, CUDA_R_16BF, N, M, N));
  cublasLtMatmulPreference_t pref;
  LT_CHECK(cublasLtMatmulPreferenceCreate(&pref));
  LT_CHECK(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &g_ws_size, sizeof(g_ws_size)));
  cublasLtMatmulHeuristicResult_t res[32];
  int n = 0;
  LT_CHECK(cublasLtMatmulAlgoGetHeuristic(g_handle, p.op, p.a, p.b, p.c, p.c, pref, 32, res, &n));
  cublasLtMatmulPreferenceDestroy(pref);
  p.tile = -1;
  for (int i = 0; i < n && p.tile < 0; ++i) {
    if (res[i].state != CUBLAS_STATUS_SUCCESS) continue;
    int t = algo_tile(res[i].algo);
    if (t == CUBLASLT_MATMUL_TILE_128x256 || t == CUBLASLT_MATMUL_TILE_256x128) { p.algo = res[i].algo; p.tile = t; }
  }
  if (p.tile < 0 && !(N == 4608 && K == 12288)) {
    // donor: the w2-like problem (M, 4608, 12288), for which the heuristic itself picks the 128x256 kernel
    Plan& d = get_plan(M, 4608, 12288);
    if (d.tile >= 0) {
      cublasLtMatmulHeuristicResult_t chk;
      if (cublasLtMatmulAlgoCheck(g_handle, p.op, p.a, p.b, p.c, p.c, &d.algo, &chk) == CUBLAS_STATUS_SUCCESS &&
          chk.workspaceSize <= g_ws_size) { p.algo = d.algo; p.tile = d.tile; }
    }
  }
  return g_plans.emplace(key, p).first->second;
}

int64_t lt_tile(int64_t M, int64_t N, int64_t K) { return get_plan(M, N, K).tile; }

torch::Tensor lt_linear(torch::Tensor x, torch::Tensor w) {
  TORCH_CHECK(x.is_cuda() && w.is_cuda() && x.scalar_type() == at::kBFloat16 && w.scalar_type() == at::kBFloat16);
  TORCH_CHECK(x.is_contiguous() && w.is_contiguous() && w.dim() == 2 && x.size(-1) == w.size(1));
  int64_t K = x.size(-1), N = w.size(0), M = x.numel() / K;
  auto& p = get_plan(M, N, K);
  TORCH_CHECK(p.tile >= 0, "no large-tile algorithm");
  auto sizes = x.sizes().vec(); sizes.back() = N;
  auto out = torch::empty(sizes, x.options());
  float alpha = 1.f, beta = 0.f;
  LT_CHECK(cublasLtMatmul(g_handle, p.op, &alpha, w.data_ptr(), p.a, x.data_ptr(), p.b, &beta,
                          out.data_ptr(), p.c, out.data_ptr(), p.c, &p.algo, g_ws, g_ws_size,
                          at::cuda::getCurrentCUDAStream()));
  return out;
}

// One row (C = 4608 = 192 threads x 3 x 8 bf16) per block:
//   HAS_RES: xo = bf16(x + gate * (a * rsqrt(mean(a^2) + eps_a) * wa))
//   HAS_NEXT: mo = bf16(xn * rsqrt(mean(xn^2) + eps_n) * wn * scale)   (xn = the bf16-rounded new residual)
template <bool HAS_RES, bool HAS_NEXT>
__global__ void __launch_bounds__(192) res_norm_kernel(const __nv_bfloat16* __restrict__ x, const __nv_bfloat16* __restrict__ a,
    const __nv_bfloat16* __restrict__ gate, const __nv_bfloat16* __restrict__ wa, const __nv_bfloat16* __restrict__ wn,
    const __nv_bfloat16* __restrict__ scale, __nv_bfloat16* __restrict__ xo, __nv_bfloat16* __restrict__ mo,
    float eps_a, float eps_n, int C) {
  constexpr int V = 3;                     // 16-byte vectors per thread
  const long row = blockIdx.x;
  const int tid = threadIdx.x;
  __shared__ float red[6];
  const uint4* xr = reinterpret_cast<const uint4*>(x + row * C);
  float xv[V][8];
#pragma unroll
  for (int v = 0; v < V; ++v) {
    uint4 u = xr[tid + v * 192];
    const __nv_bfloat162* h = reinterpret_cast<const __nv_bfloat162*>(&u);
#pragma unroll
    for (int j = 0; j < 4; ++j) { float2 f = __bfloat1622float2(h[j]); xv[v][2 * j] = f.x; xv[v][2 * j + 1] = f.y; }
  }
  auto block_sum = [&](float s) -> float {
#pragma unroll
    for (int off = 16; off > 0; off /= 2) s += __shfl_xor_sync(0xffffffffu, s, off);
    __syncthreads();
    if ((tid & 31) == 0) red[tid >> 5] = s;
    __syncthreads();
    float t = 0.f;
#pragma unroll
    for (int w = 0; w < 6; ++w) t += red[w];
    return t;
  };
  if (HAS_RES) {
    const uint4* ar = reinterpret_cast<const uint4*>(a + row * C);
    float av[V][8];
    float ss = 0.f;
#pragma unroll
    for (int v = 0; v < V; ++v) {
      uint4 u = ar[tid + v * 192];
      const __nv_bfloat162* h = reinterpret_cast<const __nv_bfloat162*>(&u);
#pragma unroll
      for (int j = 0; j < 4; ++j) { float2 f = __bfloat1622float2(h[j]); av[v][2 * j] = f.x; av[v][2 * j + 1] = f.y; ss += f.x * f.x + f.y * f.y; }
    }
    const float ra = rsqrtf(block_sum(ss) / C + eps_a);
#pragma unroll
    for (int v = 0; v < V; ++v) {
      const int c0 = (tid + v * 192) * 8;
      uint4 gu = *reinterpret_cast<const uint4*>(gate + c0), wu = *reinterpret_cast<const uint4*>(wa + c0);
      const __nv_bfloat162* g2 = reinterpret_cast<const __nv_bfloat162*>(&gu);
      const __nv_bfloat162* w2 = reinterpret_cast<const __nv_bfloat162*>(&wu);
      uint4 outv;
      __nv_bfloat162* o2 = reinterpret_cast<__nv_bfloat162*>(&outv);
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        float2 g = __bfloat1622float2(g2[j]), w = __bfloat1622float2(w2[j]);
        float r0 = xv[v][2 * j] + g.x * (av[v][2 * j] * ra * w.x);
        float r1 = xv[v][2 * j + 1] + g.y * (av[v][2 * j + 1] * ra * w.y);
        o2[j] = __floats2bfloat162_rn(r0, r1);
        float2 back = __bfloat1622float2(o2[j]);
        xv[v][2 * j] = back.x; xv[v][2 * j + 1] = back.y;
      }
      reinterpret_cast<uint4*>(xo + row * C)[tid + v * 192] = outv;
    }
  }
  if (HAS_NEXT) {
    float ss = 0.f;
#pragma unroll
    for (int v = 0; v < V; ++v)
#pragma unroll
      for (int j = 0; j < 8; ++j) ss += xv[v][j] * xv[v][j];
    const float rx = rsqrtf(block_sum(ss) / C + eps_n);
#pragma unroll
    for (int v = 0; v < V; ++v) {
      const int c0 = (tid + v * 192) * 8;
      uint4 wu = *reinterpret_cast<const uint4*>(wn + c0), su = *reinterpret_cast<const uint4*>(scale + c0);
      const __nv_bfloat162* w2 = reinterpret_cast<const __nv_bfloat162*>(&wu);
      const __nv_bfloat162* s2 = reinterpret_cast<const __nv_bfloat162*>(&su);
      uint4 outv;
      __nv_bfloat162* o2 = reinterpret_cast<__nv_bfloat162*>(&outv);
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        float2 w = __bfloat1622float2(w2[j]), sc = __bfloat1622float2(s2[j]);
        o2[j] = __floats2bfloat162_rn(xv[v][2 * j] * rx * w.x * sc.x, xv[v][2 * j + 1] * rx * w.y * sc.y);
      }
      reinterpret_cast<uint4*>(mo + row * C)[tid + v * 192] = outv;
    }
  }
}

std::vector<torch::Tensor> res_norm(torch::Tensor x, c10::optional<torch::Tensor> a, c10::optional<torch::Tensor> gate,
                                    c10::optional<torch::Tensor> wa, double eps_a, c10::optional<torch::Tensor> wn,
                                    c10::optional<torch::Tensor> scale, double eps_n) {
  const int C = x.size(-1);
  TORCH_CHECK(C == 4608 && x.is_contiguous() && x.scalar_type() == at::kBFloat16);
  const long rows = x.numel() / C;
  const bool has_res = a.has_value(), has_next = wn.has_value();
  auto xo = has_res ? torch::empty_like(x) : x;
  auto mo = has_next ? torch::empty_like(x) : x;
  auto ptr = [](const c10::optional<torch::Tensor>& t, const torch::Tensor& d) {
    return (const __nv_bfloat16*)(t.has_value() ? t->data_ptr() : d.data_ptr()); };
  auto stream = at::cuda::getCurrentCUDAStream();
#define LAUNCH(R, N) res_norm_kernel<R, N><<<rows, 192, 0, stream>>>((const __nv_bfloat16*)x.data_ptr(), ptr(a, x), ptr(gate, x), \
      ptr(wa, x), ptr(wn, x), ptr(scale, x), (__nv_bfloat16*)xo.data_ptr(), (__nv_bfloat16*)mo.data_ptr(), (float)eps_a, (float)eps_n, C)
  if (has_res && has_next) LAUNCH(true, true);
  else if (has_res) LAUNCH(true, false);
  else LAUNCH(false, true);
  return {xo, mo};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("lt_tile", &lt_tile);
  m.def("lt_linear", &lt_linear);
  m.def("res_norm", &res_norm);
}
'''

_LT = {"mod": None, "tried": False, "error": None}


def _build_lt():
    if _LT["tried"]:
        return _LT["mod"]
    _LT["tried"] = True
    try:
        from ...fast import build_extension
        _LT["mod"] = build_extension("gstar_lt_ext", LT_CUDA_SRC, ".cu", arch="9.0", extra_ldflags=["-lcublasLt"],
                                     extra_cflags=["-O2"], extra_cuda_cflags=["-O2"])
    except Exception as exc:  # pragma: no cover
        _LT["error"] = f"{type(exc).__name__}: {exc}"
        _LT["mod"] = None
        print(f"[gstar.fast] cuBLASLt extension unavailable, using F.linear: {_LT['error'][:500]}", flush=True)
    return _LT["mod"]


def lt_enabled():
    return _fast_on("REFINER_GEMM") and _sm90() and _build_lt() is not None


def fuse4_enabled():
    """The fast DiT block loop with explicit kernels - the gated residual + RMSNorm·modulation of
    the block halves as one vectorized CUDA row kernel (one block per 4,608-wide row, 16-byte loads; ≈ 4 TB/s vs
    inductor's 2.7–3.2 TB/s in context), the post-MLP residual fused with the next block's pre-norm."""
    return _fast_on("REFINER_KERNELS") and _sm90() and _build_lt() is not None


def res_norm(x, a, gate, wa, eps_a, wn, scale, eps_n):
    return _LT["mod"].res_norm(x, a, gate, wa, float(eps_a), wn, scale, float(eps_n))


@torch.library.custom_op("gstar::lt_linear", mutates_args=())
def lt_linear(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """x @ w.T (bf16, no bias) with the large-tile sm90 cuBLASLt kernel; F.linear when none applies."""
    x = x.contiguous()
    K = x.shape[-1]
    M = x.numel() // K
    if _LT["mod"] is None or _LT["mod"].lt_tile(M, w.shape[0], K) < 0:
        return torch.nn.functional.linear(x, w)
    return _LT["mod"].lt_linear(x, w.contiguous())


@lt_linear.register_fake
def _(x, w):
    return x.new_empty(*x.shape[:-1], w.shape[0])


if (_fast_on("REFINER_GEMM") or _fast_on("REFINER_KERNELS")) and _sm90():
    _build_lt()          # at import: the build is part of loading the refiner
