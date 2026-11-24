"""Graph launches without CUDAGraph.replay()'s RNG prologue (opt-in: GSTAR_FAST_LM_RAW_REPLAY=1).

torch 2.6's CUDAGraph.replay() runs, for every generator state registered at capture, replay_prologue: two
fill_ kernels (seed / offset tensors) and increase(whole-graph increment) - also for graphs that contain no RNG
consumer (increment 0).  The backbone's 37 graph pieces per decode step contain none, so their replays cost 74
extra eager launches and kernels per step.  replay_raw launches the instantiated graph directly
(cudaGraphLaunch on the current stream) after checking that every captured increment is 0 - exactly what
replay() would do minus the no-op RNG bookkeeping.  The DSpark head's graph (it samples) keeps replay().
The extension is built when this module is imported (part of loading), once per machine.
"""
import os
from ...fast import enabled as _fast_on, sm90 as _sm90

RAW_SRC = r'''
// all regular headers first (normal access), then CUDAGraph.h with its protected members opened
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAGeneratorImpl.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGraphsC10Utils.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/util/flat_hash_map.h>
#include <atomic>
#include <string>
#define protected public
#include <ATen/cuda/CUDAGraph.h>
#undef protected

// Launch a captured torch CUDA graph without CUDAGraph::replay()'s per-generator replay_prologue (two fill_
// kernels for the seed / offset tensors + increase(increment)).  Only for graphs whose captured generator
// states all have a whole-graph increment of 0 (no RNG consumer inside): replay() would not move them either.
int64_t max_rng_increment(py::object graph) {
  auto& g = graph.cast<at::cuda::CUDAGraph&>();
  uint64_t m = 0;
  for (auto& kv : g.captured_generator_states_) m = std::max(m, kv.second);
  return (int64_t)m;
}

void replay_raw(py::object graph) {
  auto& g = graph.cast<at::cuda::CUDAGraph&>();
  TORCH_CHECK(g.has_graph_exec_, "graph not instantiated");
  for (auto& kv : g.captured_generator_states_)
    TORCH_CHECK(kv.second == 0, "graph consumes RNG: use replay()");
  c10::OptionalDeviceGuard device_guard{g.capture_stream_.device()};
  C10_CUDA_CHECK(cudaGraphLaunch(g.graph_exec_, at::cuda::getCurrentCUDAStream()));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("replay_raw", &replay_raw);
  m.def("max_rng_increment", &max_rng_increment);
}
'''

_RAW = {"mod": None, "tried": False}


def _build():
    if _RAW["tried"]:
        return _RAW["mod"]
    _RAW["tried"] = True
    try:
        from torch.utils.cpp_extension import CUDA_HOME
        from ...fast import build_extension
        cuda_home = CUDA_HOME or os.environ.get("CUDA_HOME", "/usr/local/cuda")
        _RAW["mod"] = build_extension("gstar_raw_replay", RAW_SRC, ".cpp", extra_cflags=["-O2"],
                                      extra_include_paths=[os.path.join(cuda_home, "include")],
                                      extra_ldflags=["-L" + os.path.join(cuda_home, "lib64"), "-lcudart"])
    except Exception as exc:  # pragma: no cover
        print(f"[gstar.fast] raw graph launch unavailable, using replay(): {type(exc).__name__}: {str(exc)[:300]}", flush=True)
        _RAW["mod"] = None
    return _RAW["mod"]


def lm_fewer_enabled():
    return os.environ.get("GSTAR_FAST_LM_RAW_REPLAY", "0") == "1" and _sm90() and _build() is not None


def raw_ok(graph):
    m = _RAW["mod"]
    return m is not None and m.max_rng_increment(graph) == 0


def replay_raw(graph):
    _RAW["mod"].replay_raw(graph)


if os.environ.get("GSTAR_FAST_LM_RAW_REPLAY", "0") == "1" and _fast_on("LM_GRAPH"):
    _build()
