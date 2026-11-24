"""Switches of the fast paths.

Every optimization of the generation pipeline is on by default whenever its requirements hold (a CUDA device of
compute capability 9.0, the shapes and dtypes it was verified for, an extension that builds); otherwise the code
falls back to the next-best implementation on its own. The language-model optimizations reproduce the reference
arithmetic bit for bit (same codes); the refiner optimizations change only rounding inside the refiner.

Opt-outs (environment variables, read when the respective module is used):

    GSTAR_FAST=0                    everything off: the reference implementation everywhere (also the host-side
                                    changes of the generation loop: no per-step host synchronizations, no text
                                    logits during image generation)
    GSTAR_FAST_LM_GRAPH=0           CUDA graphs of the decode step and of the head's per-position sampling
    GSTAR_FAST_LM_KERNELS=0         fused bit-exact kernels of the decode step and of the head's sampling
                                    (RMSNorm, RoPE, attention operands, CFG/temperature, top-k/top-p, draw)
    GSTAR_FAST_LM_GEMM=0            bit-exact Triton replicas of cuBLAS's decode GEMMs (backbone and head)
    GSTAR_FAST_LM_ATTN=0            bit-exact fused CUDA kernels of the decode attention and the head attention
    GSTAR_FAST_REFINER=0            the refiner from the Hugging Face folder's own code (trust_remote_code)
    GSTAR_FAST_REFINER_ATTN=0       refiner attention: FlashInfer's sm90 kernel / the Triton kernel off (flash-attn)
    GSTAR_FAST_REFINER_GEMM=0       refiner linears on cuBLAS's default kernel instead of the 128x256 cuBLASLt one
    GSTAR_FAST_REFINER_KERNELS=0    refiner q/k RMSNorm + RoPE and residual + RMSNorm kernels off (torch.compile),
                                    ATen's GroupNorm in the refiner's VAE

Header locations for the CUDA extensions that need them (FlashInfer's sm90 attention templates, CUTLASS/CuTe):
an installed ``flashinfer-python`` package, or ``FLASHINFER_INCLUDE_DIR`` / ``CUTLASS_INCLUDE_DIR``.

The CUDA extensions are compiled once per machine into torch's extension directory (``TORCH_EXTENSIONS_DIR``,
default ``~/.cache/torch_extensions``) and loaded from there by later processes (see ``build_extension``).
"""
import functools
import os

_OFF = {"0", "false", "off", "no"}

PARTS = ("LM_GRAPH", "LM_KERNELS", "LM_GEMM", "LM_ATTN", "REFINER", "REFINER_ATTN", "REFINER_GEMM",
         "REFINER_KERNELS")


def _off(name):
    return os.environ.get(name, "").strip().lower() in _OFF


def enabled(part=None):
    """True unless GSTAR_FAST=0 or GSTAR_FAST_<part>=0."""
    if _off("GSTAR_FAST"):
        return False
    return part is None or not _off(f"GSTAR_FAST_{part}")


@functools.lru_cache(maxsize=None)
def sm90():
    """A CUDA device of compute capability 9.0 (the kernels' target) is the current device."""
    try:
        import torch
        return torch.cuda.is_available() and torch.cuda.get_device_capability() == (9, 0)
    except Exception:  # pragma: no cover
        return False


def note(msg):
    print(f"[gstar.fast] {msg}", flush=True)


def build_extension(name, source, suffix, arch=None, **load_kwargs):
    """torch.utils.cpp_extension.load of an extension given as a source string, compiled once per machine.

    The build lives in torch's extension directory (TORCH_EXTENSIONS_DIR, default ~/.cache/torch_extensions), where
    torch serializes concurrent builds of one extension with a file lock. The module name carries a hash of the
    source, the build arguments, the target architecture and the Python / torch / CUDA versions: a changed input is
    built under a new name, an unchanged one is loaded without compiling. If that directory is not writable, the
    extension is built in a fresh temporary directory instead."""
    import hashlib
    import sys
    import tempfile
    import torch
    from torch.utils import cpp_extension
    key = hashlib.sha256(repr((source, sorted(load_kwargs.items()), arch, sys.version_info[:2], torch.__version__,
                               torch.version.cuda)).encode()).hexdigest()[:16]
    full = f"{name}_{key}"
    prev = os.environ.get("TORCH_CUDA_ARCH_LIST")
    if arch is not None:
        os.environ["TORCH_CUDA_ARCH_LIST"] = arch
    try:
        try:
            build_dir = cpp_extension._get_build_directory(full, verbose=False)
            src = os.path.join(build_dir, full + suffix)
            current = None
            if os.path.isfile(src):
                with open(src) as f:
                    current = f.read()
            if current != source:                      # written once, atomically (concurrent processes)
                tmp = f"{src}.{os.getpid()}.tmp"
                with open(tmp, "w") as f:
                    f.write(source)
                os.replace(tmp, src)
            return cpp_extension.load(name=full, sources=[src], build_directory=build_dir, verbose=False,
                                      **load_kwargs)
        except OSError:                                # extension directory not writable
            build_dir = tempfile.mkdtemp(prefix=f"{name}_build_")
            src = os.path.join(build_dir, full + suffix)
            with open(src, "w") as f:
                f.write(source)
            return cpp_extension.load(name=full, sources=[src], build_directory=build_dir, verbose=False,
                                      **load_kwargs)
    finally:
        if arch is not None:
            if prev is None:
                os.environ.pop("TORCH_CUDA_ARCH_LIST", None)
            else:
                os.environ["TORCH_CUDA_ARCH_LIST"] = prev
