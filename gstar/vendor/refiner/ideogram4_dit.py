import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .gradient_checkpoint import gradient_checkpoint_forward
from .attention import attention_forward_varlen, triton_attention_enabled, triton_attention_supported, triton_attention
from .attention import fa3_attention_enabled, fa3_attention_supported, fa3_attention
from .lt_gemm import lt_enabled as _lt_enabled, fuse4_enabled as _fuse4_enabled, res_norm as _res_norm
from .qk_rope_kernel import dit_qkr_enabled, qk_norm_rope
from ...fast import enabled as _fast_on, sm90 as _sm90

LLM_TOKEN_INDICATOR = 3
OUTPUT_IMAGE_INDICATOR = 2
IMAGE_POSITION_OFFSET = 65536
QWEN3_VL_ACTIVATION_LAYERS = (0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 35)

FP8_E4M3_MAX = 448.0
FP8_WEIGHT_DTYPE = torch.float8_e4m3fn
FP8_SCALE_SUFFIX = ".weight_scale"


class Fp8Linear(nn.Module):
    """Linear layer holding an e4m3 float8 weight + per-row float32 scale."""

    weight: torch.Tensor
    weight_scale: torch.Tensor
    bias: torch.Tensor | None

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool,
        compute_dtype: torch.dtype,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.compute_dtype = compute_dtype
        self.register_buffer(
            "weight",
            torch.empty(out_features, in_features, dtype=FP8_WEIGHT_DTYPE),
        )
        self.register_buffer("weight_scale", torch.empty(out_features, dtype=torch.float32))
        if bias:
            self.register_buffer("bias", torch.empty(out_features, dtype=compute_dtype))
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.weight.to(x.dtype) * self.weight_scale.to(x.dtype).unsqueeze(1)
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x, w, bias)


def is_fp8_state_dict(state_dict: dict[str, torch.Tensor]) -> bool:
    return any(k.endswith(FP8_SCALE_SUFFIX) for k in state_dict) or any(
        v.dtype == FP8_WEIGHT_DTYPE for v in state_dict.values()
    )


def swap_linears_to_fp8(
    module: nn.Module,
    state_dict: dict[str, torch.Tensor],
    compute_dtype: torch.dtype,
    *,
    prefix: str = "",
) -> None:
    for name, child in list(module.named_children()):
        child_prefix = f"{prefix}{name}"
        if (
            isinstance(child, nn.Linear) and f"{child_prefix}{FP8_SCALE_SUFFIX}" in state_dict
        ):
            setattr(
                module,
                name,
                Fp8Linear(
                    child.in_features,
                    child.out_features,
                    bias=child.bias is not None,
                    compute_dtype=compute_dtype,
                ),
            )
        else:
            swap_linears_to_fp8(child, state_dict, compute_dtype, prefix=f"{child_prefix}.")


@dataclass
class Ideogram4Config:
    emb_dim: int = 4608
    num_layers: int = 34
    num_heads: int = 18
    intermediate_size: int = 12288
    adanln_dim: int = 512
    in_channels: int = 128
    llm_features_dim: int = 4096 * len(QWEN3_VL_ACTIVATION_LAYERS)
    rope_theta: int = 5_000_000
    mrope_section: tuple[int, ...] = (24, 20, 20)
    norm_eps: float = 1e-5


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return torch.cat((-x2, x1), dim=-1)


def _apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    q_embed = (q * cos) + (_rotate_half(q) * sin)
    k_embed = (k * cos) + (_rotate_half(k) * sin)
    return q_embed, k_embed


class Ideogram4MRoPE(nn.Module):
    inv_freq: torch.Tensor

    def __init__(
        self,
        head_dim: int,
        base: int,
        mrope_section: tuple[int, ...],
    ) -> None:
        super().__init__()
        inv_freq = 1.0 / (
            base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.mrope_section = tuple(mrope_section)
        self.head_dim = head_dim

    @torch.no_grad()
    def forward(self, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        assert position_ids.ndim == 3 and position_ids.shape[-1] == 3
        batch_size, seq_len, _ = position_ids.shape

        pos = position_ids.permute(2, 0, 1).to(dtype=torch.float32)
        inv_freq = self.inv_freq.to(dtype=torch.float32)[None, None, :, None].expand(
            3, batch_size, -1, 1
        ).to(pos.device)
        freqs = inv_freq @ pos.unsqueeze(2)
        freqs = freqs.transpose(2, 3)

        freqs_t = freqs[0].clone()
        for axis, offset in ((1, 1), (2, 2)):
            length = self.mrope_section[axis] * 3
            idx = torch.arange(offset, length, 3, device=freqs_t.device)
            freqs_t[..., idx] = freqs[axis][..., idx]

        emb = torch.cat((freqs_t, freqs_t), dim=-1)
        return emb.cos(), emb.sin()


class Ideogram4RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.rms_norm(x, self.weight.shape, self.weight, self.eps)


def _attention_varlen(q, k, v, cu_seqlens, max_seqlen):
    """Block-diagonal (packed-sequence) flash attention.

    q/k/v: (1, H, S, D) with S = sum of per-sample lengths. `cu_seqlens` marks
    [0, S_0, S_0+S_1, ...] so samples never attend across each other. Returns
    (1, S, H*D). Mirrors krea2_dit.attention_varlen."""
    q_v = q[0].transpose(0, 1).contiguous()  # (S, H, D)
    k_v = k[0].transpose(0, 1).contiguous()
    v_v = v[0].transpose(0, 1).contiguous()
    out = attention_forward_varlen(
        q_v, k_v, v_v,
        cu_seqlens_q=cu_seqlens, cu_seqlens_k=cu_seqlens,
        max_seqlen_q=max_seqlen, max_seqlen_k=max_seqlen,
        is_causal=False,
    )  # (S, H, D)
    return out.reshape(out.shape[0], -1).unsqueeze(0)


class Ideogram4Attention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, eps: float = 1e-5) -> None:
        super().__init__()
        assert hidden_size % num_heads == 0
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        self.qkv = nn.Linear(hidden_size, hidden_size * 3, bias=False)
        self.norm_q = Ideogram4RMSNorm(self.head_dim, eps=eps)
        self.norm_k = Ideogram4RMSNorm(self.head_dim, eps=eps)
        self.o = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        segment_ids: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        cu_seqlens: torch.Tensor = None,
        max_seqlen: int = None,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        qkv = self.qkv(x)
        qkv = qkv.view(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)

        q = self.norm_q(q)
        k = self.norm_k(k)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        q, k = _apply_rotary_pos_emb(q, k, cos, sin)

        if cu_seqlens is not None:
            # Block-diagonal flash-varlen path (packed anyres, O(ΣL) memory).
            # q/k/v are (1, H, S, D); segment_ids not needed — cu_seqlens marks
            # the per-sample boundaries so samples never attend across each other.
            out = _attention_varlen(q, k, v, cu_seqlens, max_seqlen)  # (1, S, H*D)
            return self.o(out)

        attn_mask = (segment_ids.unsqueeze(2) == segment_ids.unsqueeze(1)).unsqueeze(1)

        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        out = out.transpose(1, 2).reshape(batch_size, seq_len, self.hidden_size)
        return self.o(out)


class Ideogram4MLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Ideogram4TransformerBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_heads: int,
        norm_eps: float,
        adanln_dim: int,
    ) -> None:
        super().__init__()
        self.attention = Ideogram4Attention(hidden_size, num_heads, eps=1e-5)
        self.feed_forward = Ideogram4MLP(hidden_size, intermediate_size)

        self.attention_norm1 = Ideogram4RMSNorm(hidden_size, eps=norm_eps)
        self.ffn_norm1 = Ideogram4RMSNorm(hidden_size, eps=norm_eps)
        self.attention_norm2 = Ideogram4RMSNorm(hidden_size, eps=norm_eps)
        self.ffn_norm2 = Ideogram4RMSNorm(hidden_size, eps=norm_eps)

        self.adaln_modulation = nn.Linear(adanln_dim, 4 * hidden_size, bias=True)

    def forward(
        self,
        x: torch.Tensor,
        segment_ids: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        adaln_input: torch.Tensor,
        cu_seqlens: torch.Tensor = None,
        max_seqlen: int = None,
    ) -> torch.Tensor:
        mod = self.adaln_modulation(adaln_input)
        scale_msa, gate_msa, scale_mlp, gate_mlp = mod.chunk(4, dim=-1)
        gate_msa = torch.tanh(gate_msa)
        gate_mlp = torch.tanh(gate_mlp)
        scale_msa = 1.0 + scale_msa
        scale_mlp = 1.0 + scale_mlp

        attn_out = self.attention(
            self.attention_norm1(x) * scale_msa,
            segment_ids=segment_ids,
            cos=cos,
            sin=sin,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )
        x = x + gate_msa * self.attention_norm2(attn_out)
        x = x + gate_mlp * self.ffn_norm2(self.feed_forward(self.ffn_norm1(x) * scale_mlp))
        return x


def _sinusoidal_embedding(
    t: torch.Tensor, dim: int, scale: float = 1e4
) -> torch.Tensor:
    t = t.to(torch.float32)
    half = dim // 2
    freq = math.log(scale) / (half - 1)
    freq = torch.exp(torch.arange(half, dtype=torch.float32, device=t.device) * -freq)
    emb = t.unsqueeze(-1) * freq
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


class Ideogram4EmbedScalar(nn.Module):
    def __init__(self, dim: int, input_range: tuple[float, float]) -> None:
        super().__init__()
        self.dim = dim
        self.range_min, self.range_max = input_range
        assert self.range_max > self.range_min
        self.mlp_in = nn.Linear(dim, dim, bias=True)
        self.mlp_out = nn.Linear(dim, dim, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.float32)
        scaled = 1e4 * (x - self.range_min) / (self.range_max - self.range_min)
        emb = _sinusoidal_embedding(scaled, self.dim)
        emb = emb.to(
            getattr(self.mlp_in, "compute_dtype", None) or getattr(self.mlp_in, "computation_dtype", None) or self.mlp_in.weight.dtype
        )
        emb = F.silu(self.mlp_in(emb))
        return self.mlp_out(emb)


class Ideogram4FinalLayer(nn.Module):
    def __init__(self, hidden_size: int, out_channels: int, adanln_dim: int) -> None:
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, eps=1e-6, elementwise_affine=False)
        self.linear = nn.Linear(hidden_size, out_channels, bias=True)
        self.adaln_modulation = nn.Linear(adanln_dim, hidden_size, bias=True)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        scale = 1.0 + self.adaln_modulation(F.silu(c))
        return self.linear(self.norm_final(x) * scale)


# ---------------------------------------------------------------------------
# Fast single-sample path (on by default; opt-out GSTAR_FAST_REFINER=0; same model
# function, other rounding).  With one packed sample every token shares the flow time t, so the
# per-token adaLN inputs are identical rows: compute the modulation once and
# broadcast it.  The norm / modulation / RoPE / gating / SwiGLU chains around the
# GEMMs are fused by torch.compile (weights are arguments, so one compiled graph
# serves all 34 blocks); attention stays the flash-attn varlen kernel.
# ---------------------------------------------------------------------------
import os as _os


def dit_fast_enabled():
    return _fast_on("REFINER")


def _rms(x, w, eps: float):
    return F.rms_norm(x, w.shape, w, eps)


_LT_ON = _lt_enabled()      # large-tile cuBLASLt kernel for the block linears (bitwise == F.linear)


def _lin(x, w):
    if _LT_ON:
        return torch.ops.gstar.lt_linear(x, w)
    return F.linear(x, w)


def _fast_pre(x, scale_msa, n1_w, qkv_w, nq_w, nk_w, cos, sin, eps: float, qk_eps: float,
              num_heads: int, head_dim: int):
    S = x.shape[1]
    h = _rms(x, n1_w, eps) * scale_msa
    qkv = _lin(h, qkv_w).view(1, S, 3, num_heads, head_dim)
    q, k, v = qkv.unbind(dim=2)                      # (1, S, H, D)
    q = _rms(q, nq_w, qk_eps)
    k = _rms(k, nk_w, qk_eps)
    c = cos.unsqueeze(2)                             # (1, S, 1, D)
    s_ = sin.unsqueeze(2)
    q = (q * c) + (_rotate_half(q) * s_)
    k = (k * c) + (_rotate_half(k) * s_)
    return q[0].contiguous(), k[0].contiguous(), v[0].contiguous()   # (S, H, D)


def _fast_pre_view_v(x, scale_msa, n1_w, qkv_w, nq_w, nk_w, cos, sin, eps: float, qk_eps: float,
                     num_heads: int, head_dim: int):
    """As _fast_pre, but V stays a strided view of the QKV GEMM output (the Triton attention takes strides)."""
    S = x.shape[1]
    h = _rms(x, n1_w, eps) * scale_msa
    qkv = _lin(h, qkv_w).view(1, S, 3, num_heads, head_dim)
    q, k, v = qkv.unbind(dim=2)
    q = _rms(q, nq_w, qk_eps)
    k = _rms(k, nk_w, qk_eps)
    c = cos.unsqueeze(2)
    s_ = sin.unsqueeze(2)
    q = (q * c) + (_rotate_half(q) * s_)
    k = (k * c) + (_rotate_half(k) * s_)
    return q[0].contiguous(), k[0].contiguous(), v[0]


def _fast_post(x, attn, o_w, n2_w, gate_msa, f1_w, scale_mlp, w1, w3, w2, f2_w, gate_mlp, eps: float):
    a = _lin(attn, o_w)
    x = x + gate_msa * _rms(a, n2_w, eps)
    m = _rms(x, f1_w, eps) * scale_mlp
    m = _lin(F.silu(_lin(m, w1)) * _lin(m, w3), w2)
    return x + gate_mlp * _rms(m, f2_w, eps)


def _fast_pre_qkv(x, scale_msa, n1_w, qkv_w, eps: float):
    """The compiled part of the pre half (norm1 * scale, QKV GEMM); q/k norm + RoPE follow in Triton."""
    h = _rms(x, n1_w, eps) * scale_msa
    return _lin(h, qkv_w)


def _pre_qkr(pre_qkv):
    def pre(x, scale_msa, n1_w, qkv_w, nq_w, nk_w, cos, sin, eps: float, qk_eps: float, num_heads: int,
            head_dim: int):
        S = x.shape[1]
        qkv = pre_qkv(x, scale_msa, n1_w, qkv_w, eps)                     # [1, S, 3*H*D]
        q, k = qk_norm_rope(qkv, nq_w, nk_w, cos, sin, qk_eps, num_heads, head_dim)
        v = qkv.view(1, S, 3, num_heads, head_dim)[0, :, 2]              # strided view, as _fast_pre_view_v
        return q, k, v
    return pre


_FAST_FNS = {}


def compile_options():
    """GSTAR_FAST_REFINER_AUTOTUNE=1 (opt-in): inductor max-autotune for the GEMMs of the compiled refiner pieces (Triton templates
    incl. the persistent TMA matmul vs cuBLAS, chosen per shape at the first call)."""
    if _os.environ.get("GSTAR_FAST_REFINER_AUTOTUNE", "0") != "1":
        return None
    return {"max_autotune": True, "max_autotune_gemm_backends": "TRITON,ATEN",
            "triton.enable_persistent_tma_matmul": True}


def _compile(fn):
    opts = compile_options()
    if opts is None:
        return torch.compile(fn, dynamic=False, fullgraph=True)
    return torch.compile(fn, dynamic=False, fullgraph=True, options=opts)


def _fast_fns():
    if not _FAST_FNS:
        _FAST_FNS["pre"] = _compile(_fast_pre)
        _FAST_FNS["post"] = _compile(_fast_post)
    if _fast_on("REFINER") and (triton_attention_enabled() or fa3_attention_enabled()):
        if dit_qkr_enabled():
            if "pre_qkr" not in _FAST_FNS:
                _FAST_FNS["pre_qkr"] = _pre_qkr(_compile(_fast_pre_qkv))
            return _FAST_FNS["pre_qkr"], _FAST_FNS["post"]
        if "pre_v" not in _FAST_FNS:
            _FAST_FNS["pre_v"] = _compile(_fast_pre_view_v)
        return _FAST_FNS["pre_v"], _FAST_FNS["post"]
    return _FAST_FNS["pre"], _FAST_FNS["post"]


class Ideogram4DiT(nn.Module):
    """Ideogram 4 flow-matching transformer."""

    def __init__(self, config: Ideogram4Config | dict = None, **kwargs) -> None:
        super().__init__()
        if config is None:
            config = Ideogram4Config()
        elif isinstance(config, dict):
            config = Ideogram4Config(**config)
        self.config = config
        self.patch_size = 2

        head_dim = config.emb_dim // config.num_heads

        self.input_proj = nn.Linear(config.in_channels, config.emb_dim, bias=True)
        self.llm_cond_norm = Ideogram4RMSNorm(config.llm_features_dim, eps=1e-6)
        self.llm_cond_proj = nn.Linear(config.llm_features_dim, config.emb_dim, bias=True)
        self.t_embedding = Ideogram4EmbedScalar(config.emb_dim, input_range=(0.0, 1.0))
        self.adaln_proj = nn.Linear(config.emb_dim, config.adanln_dim, bias=True)

        self.embed_image_indicator = nn.Embedding(2, config.emb_dim)

        self.rotary_emb = Ideogram4MRoPE(
            head_dim=head_dim,
            base=config.rope_theta,
            mrope_section=config.mrope_section,
        )

        self.layers = nn.ModuleList(
            [
                Ideogram4TransformerBlock(
                    hidden_size=config.emb_dim,
                    intermediate_size=config.intermediate_size,
                    num_heads=config.num_heads,
                    norm_eps=config.norm_eps,
                    adanln_dim=config.adanln_dim,
                )
                for _ in range(config.num_layers)
            ]
        )

        self.final_layer = Ideogram4FinalLayer(
            hidden_size=config.emb_dim,
            out_channels=config.in_channels,
            adanln_dim=config.adanln_dim,
        )

    def load_state_dict(self, state_dict, strict=True, assign=False):
        if is_fp8_state_dict(state_dict):
            swap_linears_to_fp8(self, state_dict, torch.bfloat16)
            return super().load_state_dict(state_dict, strict=False, assign=assign)
        return super().load_state_dict(state_dict, strict=strict, assign=assign)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def forward(
        self,
        *,
        llm_features: torch.Tensor,
        x: torch.Tensor,
        t: torch.Tensor,
        position_ids: torch.Tensor,
        segment_ids: torch.Tensor,
        indicator: torch.Tensor,
        cu_seqlens: torch.Tensor = None,
        max_seqlen: int = None,
        context_precomputed: bool = False,
        inject_fn=None,
        inject_layers=None,
        feature_layers: list[int] | tuple[int, ...] | None = None,
        return_features: bool = False,
        use_gradient_checkpointing: bool = False,
        use_gradient_checkpointing_offload: bool = False,
    ) -> torch.Tensor:
        """Velocity prediction.

        Args:
            llm_features: (B, L, llm_features_dim) Qwen3-VL conditioning features,
                OR (B, L, emb_dim) already-projected context when
                `context_precomputed=True` (in-context conditioning path).
            x: (B, L, in_channels) noise tokens.
            t: (B,) or (B, L) flow-matching time in [0, 1]. For packed multi-sample
                batches pass per-token (B, L) so each sample carries its own time.
            position_ids: (B, L, 3) (t, h, w) positions for MRoPE.
            segment_ids: (B, L) sample id within a packed batch (dense path only).
            indicator: (B, L) per-token role: LLM_TOKEN_INDICATOR or OUTPUT_IMAGE_INDICATOR.
            cu_seqlens/max_seqlen: when given, use the flash block-diagonal varlen
                attention (packed anyres) instead of the dense segment mask.
            inject_fn(point) -> (1, N_img, out_dim): additive residual on the IMAGE
                tokens. point = -2 (before input_proj, dim in_channels), -1 (after
                input_proj, before blocks, dim emb_dim), i>=0 (after block i, dim
                emb_dim). None -> byte-identical to the un-conditioned base.
            context_precomputed: llm_features is already emb_dim (skip norm/proj).

        Returns:
            Normally `(B, L, in_channels)` velocity prediction in float32.
            With `return_features=True`, returns `(velocity, features)`, where
            `features` contains the packed hidden states after the requested
            transformer blocks. This is used by one-step adversarial training
            to turn a frozen DiT into a projected discriminator backbone.
        """
        batch_size, seq_len, in_channels = x.shape
        assert in_channels == self.config.in_channels
        fast = getattr(self, "_fast_image_range", None)
        if (fast is not None and dit_fast_enabled() and batch_size == 1 and cu_seqlens is not None
                and cu_seqlens.numel() == 2 and t.dim() == 2 and not return_features
                and not feature_layers and not use_gradient_checkpointing and not torch.is_grad_enabled()
                and fast[1] == seq_len):
            return self._fast_forward(llm_features=llm_features, x=x, t=t, position_ids=position_ids,
                                      indicator=indicator, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen,
                                      context_precomputed=context_precomputed, inject_fn=inject_fn,
                                      inject_layers=inject_layers, img_range=fast)

        param_dtype = (
            getattr(self.input_proj, "compute_dtype", None) or getattr(self.input_proj, "computation_dtype", None) or self.input_proj.weight.dtype
        )
        x = x.to(param_dtype)
        t = t.to(param_dtype)
        llm_features = llm_features.to(param_dtype)

        indicator = indicator.to(torch.long)
        image_bool = (indicator == OUTPUT_IMAGE_INDICATOR)          # (B, L)
        llm_token_mask = (indicator == LLM_TOKEN_INDICATOR).to(x.dtype).unsqueeze(-1)
        output_image_mask = image_bool.to(x.dtype).unsqueeze(-1)

        do_inject = inject_fn is not None and inject_layers is not None
        feature_layers = tuple(feature_layers or ())
        feature_layer_set = set(feature_layers)
        captured_features = {}

        def _add_img(hh, point):
            """Add inject_fn(point) (1, N_img, D) onto the image-token positions of
            hh (B, L, D) in packed image order (== sequence image order)."""
            resid = inject_fn(point).to(hh.dtype)
            hh = hh.clone()
            hh[image_bool] = hh[image_bool] + resid.reshape(-1, resid.shape[-1])
            return hh

        llm_features = llm_features * llm_token_mask
        x = x * output_image_mask

        if do_inject and -2 in inject_layers:                       # latent-space (pre input_proj)
            x = _add_img(x, -2)
        x = self.input_proj(x) * output_image_mask

        t_cond = self.t_embedding(t)
        if t.dim() == 1:
            t_cond = t_cond.unsqueeze(1)
        adaln_input = F.silu(self.adaln_proj(t_cond))

        if context_precomputed:
            llm_features = llm_features * llm_token_mask            # already emb_dim
        else:
            llm_features = self.llm_cond_norm(llm_features)
            llm_features = self.llm_cond_proj(llm_features) * llm_token_mask

        h = x + llm_features

        image_indicator_embedding = self.embed_image_indicator(image_bool.to(torch.long))
        h = h + image_indicator_embedding

        if do_inject and -1 in inject_layers:                       # DiT input (post input_proj)
            h = _add_img(h, -1)

        cos, sin = self.rotary_emb(position_ids)
        cos = cos.to(h.dtype)
        sin = sin.to(h.dtype)

        for i, layer in enumerate(self.layers):
            h = gradient_checkpoint_forward(
                layer,
                use_gradient_checkpointing=use_gradient_checkpointing,
                use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
                x=h,
                segment_ids=segment_ids,
                cos=cos,
                sin=sin,
                adaln_input=adaln_input,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
            )
            if do_inject and i in inject_layers:
                h = _add_img(h, i)
            if i in feature_layer_set:
                captured_features[i] = h

        out = self.final_layer(h, c=adaln_input)
        out = out.to(torch.float32)
        if return_features:
            return out, [captured_features[i] for i in feature_layers]
        return out

    def _fuse4_layers(self, h, adaln_input, cos, sin, S, do_inject, inject_layers, _add_img):
        """The block loop with explicit kernels (same model function, other rounding):
        QKV / o / w1 / w3 / w2 on the cuBLASLt 128x256 kernel, q/k norm + RoPE (qk_rope_kernel), FA3 attention, the
        gated residual + RMSNorm·modulation of each half as one CUDA row kernel, the post-MLP residual fused with
        the next block's pre-norm (separate where a condition injection follows the block)."""
        lin = torch.ops.gstar.lt_linear
        swiglu = _FAST_FNS.get("swiglu")
        if swiglu is None:
            swiglu = _FAST_FNS["swiglu"] = _compile(lambda g, u: F.silu(g) * u)
        mods = []
        for layer in self.layers:
            mod = layer.adaln_modulation(adaln_input)
            scale_msa, gate_msa, scale_mlp, gate_mlp = mod.chunk(4, dim=-1)
            mods.append(((1.0 + scale_msa).contiguous(), torch.tanh(gate_msa).contiguous(),
                         (1.0 + scale_mlp).contiguous(), torch.tanh(gate_mlp).contiguous()))
        n = len(self.layers)
        l0 = self.layers[0]
        hn = _res_norm(h, None, None, None, 0.0, l0.attention_norm1.weight, mods[0][0], l0.attention_norm1.eps)[1]
        for i, layer in enumerate(self.layers):
            scale_msa, gate_msa, scale_mlp, gate_mlp = mods[i]
            att = layer.attention
            ff = layer.feed_forward
            eps2 = layer.attention_norm2.eps
            qkv = lin(hn, att.qkv.weight)
            q, k = qk_norm_rope(qkv, att.norm_q.weight, att.norm_k.weight, cos, sin, att.norm_q.eps, att.num_heads,
                                att.head_dim)
            v = qkv.view(1, S, 3, att.num_heads, att.head_dim)[0, :, 2]
            o = fa3_attention(q, k, v).reshape(1, S, -1)
            a = lin(o, att.o.weight)
            h, m = _res_norm(h, a, gate_msa, layer.attention_norm2.weight, eps2, layer.ffn_norm1.weight, scale_mlp,
                             eps2)
            s_ = swiglu(lin(m, ff.w1.weight), lin(m, ff.w3.weight))
            m2 = lin(s_, ff.w2.weight)
            nxt = i + 1 < n
            inj = do_inject and i in inject_layers
            if nxt and not inj:
                ln = self.layers[i + 1]
                h, hn = _res_norm(h, m2, gate_mlp, layer.ffn_norm2.weight, eps2, ln.attention_norm1.weight,
                                  mods[i + 1][0], ln.attention_norm1.eps)
            else:
                h = _res_norm(h, m2, gate_mlp, layer.ffn_norm2.weight, eps2, None, None, 0.0)[0]
                if inj:
                    h = _add_img(h, i)
                if nxt:
                    ln = self.layers[i + 1]
                    hn = _res_norm(h, None, None, None, 0.0, ln.attention_norm1.weight, mods[i + 1][0],
                                   ln.attention_norm1.eps)[1]
        return h

    def _fast_forward(self, *, llm_features, x, t, position_ids, indicator, cu_seqlens, max_seqlen,
                      context_precomputed, inject_fn, inject_layers, img_range):
        """Single packed sample: text tokens [0, s), image tokens [s, e) (e == seq_len)."""
        s, e = img_range
        param_dtype = (
            getattr(self.input_proj, "compute_dtype", None) or getattr(self.input_proj, "computation_dtype", None) or self.input_proj.weight.dtype
        )
        x = x.to(param_dtype)
        t = t.to(param_dtype)
        llm_features = llm_features.to(param_dtype)
        indicator = indicator.to(torch.long)
        image_bool = (indicator == OUTPUT_IMAGE_INDICATOR)
        llm_token_mask = (indicator == LLM_TOKEN_INDICATOR).to(x.dtype).unsqueeze(-1)
        output_image_mask = image_bool.to(x.dtype).unsqueeze(-1)
        do_inject = inject_fn is not None and inject_layers is not None

        def _add_img(hh, point):
            resid = inject_fn(point).to(hh.dtype)
            hh = hh.clone()
            hh[:, s:e] = hh[:, s:e] + resid
            return hh

        llm_features = llm_features * llm_token_mask
        x = x * output_image_mask
        if do_inject and -2 in inject_layers:
            x = _add_img(x, -2)
        x = self.input_proj(x) * output_image_mask

        t_cond = self.t_embedding(t[:, :1])                       # (1, 1, emb): t is uniform
        adaln_input = F.silu(self.adaln_proj(t_cond))             # (1, 1, adanln)

        if context_precomputed:
            llm_features = llm_features * llm_token_mask
        else:
            llm_features = self.llm_cond_norm(llm_features)
            llm_features = self.llm_cond_proj(llm_features) * llm_token_mask
        h = x + llm_features
        h = h + self.embed_image_indicator(image_bool.to(torch.long))
        if do_inject and -1 in inject_layers:
            h = _add_img(h, -1)

        cos, sin = self.rotary_emb(position_ids)
        cos = cos.to(h.dtype)
        sin = sin.to(h.dtype)
        pre, post = _fast_fns()
        S = h.shape[1]
        if (_fuse4_enabled() and dit_qkr_enabled() and fa3_attention_enabled() and h.shape[-1] == 4608
                and h.is_contiguous()):
            h = self._fuse4_layers(h, adaln_input, cos, sin, S, do_inject, inject_layers, _add_img)
            out = self.final_layer(h, c=adaln_input)
            return out.to(torch.float32)
        for i, layer in enumerate(self.layers):
            mod = layer.adaln_modulation(adaln_input)
            scale_msa, gate_msa, scale_mlp, gate_mlp = mod.chunk(4, dim=-1)
            gate_msa = torch.tanh(gate_msa)
            gate_mlp = torch.tanh(gate_mlp)
            scale_msa = 1.0 + scale_msa
            scale_mlp = 1.0 + scale_mlp
            att = layer.attention
            q, k, v = pre(h, scale_msa, layer.attention_norm1.weight, att.qkv.weight, att.norm_q.weight,
                          att.norm_k.weight, cos, sin, layer.attention_norm1.eps, att.norm_q.eps,
                          att.num_heads, att.head_dim)
            if fa3_attention_enabled() and fa3_attention_supported(q, k, v):
                o = fa3_attention(q, k, v)                       # FlashInfer sm90 (FA3-derived) kernel
            elif triton_attention_enabled() and triton_attention_supported(q, k, v):
                o = triton_attention(q, k, v)                    # one sequence: S == max_seqlen
            else:
                o = attention_forward_varlen(q, k, v, cu_seqlens_q=cu_seqlens, cu_seqlens_k=cu_seqlens,
                                             max_seqlen_q=max_seqlen, max_seqlen_k=max_seqlen, is_causal=False)
            o = o.reshape(1, S, -1)
            ff = layer.feed_forward
            h = post(h, o, att.o.weight, layer.attention_norm2.weight, gate_msa, layer.ffn_norm1.weight,
                     scale_mlp, ff.w1.weight, ff.w3.weight, ff.w2.weight, layer.ffn_norm2.weight, gate_mlp,
                     layer.attention_norm2.eps)
            if do_inject and i in inject_layers:
                h = _add_img(h, i)

        out = self.final_layer(h, c=adaln_input)
        return out.to(torch.float32)
