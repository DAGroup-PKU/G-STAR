"""GroupNorm for the refiner's VAE at inference (on by default; opt-out GSTAR_FAST_REFINER_KERNELS=0; other rounding).

ATen's GroupNorm reduces each (sample, group) row with one thread block, so at batch 1 a 2048 x 2048 decoder
activation gets 32 blocks on the whole GPU (≈ 6 ms per call). Here the statistics are fp32 partial sums over many
chunks per group (mean, then the centered sum of squares: two passes, numerically stable), combined in fp64, and the
normalization + affine is applied per channel in one pass with a single rounding to the output dtype.
"""
import torch
from torch import nn

from ...fast import enabled as _fast_on

try:
    import triton
    import triton.language as tl
    _TRITON = True
except Exception:  # pragma: no cover
    _TRITON = False

_CHUNK, _BLOCK = 16384, 2048

if _TRITON:
    @triton.jit
    def _gn_sum_kernel(x_ptr, mean_ptr, out_ptr, group_numel, n_chunks, CHUNK: tl.constexpr, BLOCK: tl.constexpr,
                       CENTERED: tl.constexpr):
        row = tl.program_id(0)
        ch = tl.program_id(1)
        base = x_ptr + row.to(tl.int64) * group_numel
        start = ch.to(tl.int64) * CHUNK
        acc = tl.zeros([BLOCK], dtype=tl.float32)
        if CENTERED:
            mu = tl.load(mean_ptr + row)
        for off in range(0, CHUNK, BLOCK):
            idx = start + off + tl.arange(0, BLOCK)
            m = idx < group_numel
            v = tl.load(base + idx, mask=m, other=0.0).to(tl.float32)
            if CENTERED:
                d = tl.where(m, v - mu, 0.0)
                acc += d * d
            else:
                acc += v
        tl.store(out_ptr + row * n_chunks + ch, tl.sum(acc, axis=0))

    @triton.jit
    def _gn_apply_kernel(x_ptr, y_ptr, mean_ptr, rstd_ptr, w_ptr, b_ptr, HW, C, G, CHUNK: tl.constexpr,
                         BLOCK: tl.constexpr):
        nc = tl.program_id(0)            # n * C + c
        ch = tl.program_id(1)
        c = nc % C
        n = nc // C
        g = c // (C // G)
        mu = tl.load(mean_ptr + n * G + g)
        rs = tl.load(rstd_ptr + n * G + g)
        w = tl.load(w_ptr + c).to(tl.float32)
        b = tl.load(b_ptr + c).to(tl.float32)
        base = nc.to(tl.int64) * HW
        start = ch.to(tl.int64) * CHUNK
        for off in range(0, CHUNK, BLOCK):
            idx = start + off + tl.arange(0, BLOCK)
            m = idx < HW
            v = tl.load(x_ptr + base + idx, mask=m, other=0.0).to(tl.float32)
            tl.store(y_ptr + base + idx, ((v - mu) * rs * w + b).to(y_ptr.dtype.element_ty), mask=m)


def fast_group_norm(x, num_groups, weight, bias, eps):
    N, C = x.shape[:2]
    HW = x.numel() // (N * C)
    gnum = (C // num_groups) * HW
    rows = N * num_groups
    nch = triton.cdiv(gnum, _CHUNK)
    part = torch.empty(rows, nch, device=x.device, dtype=torch.float32)
    _gn_sum_kernel[(rows, nch)](x, part, part, gnum, nch, CHUNK=_CHUNK, BLOCK=_BLOCK, CENTERED=False, num_warps=8)
    mean = part.sum(1, dtype=torch.float64).div_(gnum).float()
    _gn_sum_kernel[(rows, nch)](x, mean, part, gnum, nch, CHUNK=_CHUNK, BLOCK=_BLOCK, CENTERED=True, num_warps=8)
    rstd = torch.rsqrt(part.sum(1, dtype=torch.float64).div_(gnum) + eps).float()
    y = torch.empty_like(x)
    _gn_apply_kernel[(N * C, triton.cdiv(HW, _CHUNK))](x, y, mean, rstd, weight, bias, HW, C, num_groups,
                                                         CHUNK=_CHUNK, BLOCK=_BLOCK, num_warps=8)
    return y


class GroupNorm(nn.GroupNorm):
    """nn.GroupNorm (same parameters and state dict) with the faster inference forward above where it applies."""

    def forward(self, x):
        if (_TRITON and not torch.is_grad_enabled() and x.is_cuda and x.is_contiguous() and x.dim() >= 3
                and self.affine and x.dtype in (torch.bfloat16, torch.float16, torch.float32)
                and x.shape[1] == self.num_channels and self.num_channels % self.num_groups == 0
                and _fast_on("REFINER_KERNELS")):
            return fast_group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
        return super().forward(x)
