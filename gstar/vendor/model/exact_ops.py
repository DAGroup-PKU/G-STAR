"""fp32 round-to-nearest arithmetic without flush-to-zero for the bit-exact Triton kernels.

libdevice's __nv_fmul_rn / __nv_fadd_rn are compiled with FTZ under Triton (denormal results become 0),
while ATen's kernels keep denormals.  These helpers emit the plain IEEE instructions (no .ftz), which are
also never contracted into FMAs.
"""
import triton
import triton.language as tl


@triton.jit
def mul_rn(a, b):
    return tl.inline_asm_elementwise("mul.rn.f32 $0, $1, $2;", "=f,f,f", [a, b], dtype=tl.float32,
                                     is_pure=True, pack=1)


@triton.jit
def add_rn(a, b):
    return tl.inline_asm_elementwise("add.rn.f32 $0, $1, $2;", "=f,f,f", [a, b], dtype=tl.float32,
                                     is_pure=True, pack=1)


@triton.jit
def sub_rn(a, b):
    return tl.inline_asm_elementwise("sub.rn.f32 $0, $1, $2;", "=f,f,f", [a, b], dtype=tl.float32,
                                     is_pure=True, pack=1)
