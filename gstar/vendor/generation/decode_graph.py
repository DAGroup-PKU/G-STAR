"""Piecewise CUDA-graph decode step for OmniForCausalLM (on by default; opt-out GSTAR_FAST_LM_GRAPH=0).

One decode forward (q_len = 1, the CFG pair as batch rows) is split at the attention core of every
layer:

    G_0   : token embedding + vision bridge + multimodal merge -> x_0, layer 0 pre-attention
    attn_0: eager (length dependent)
    G_i   : layer i-1 post-attention (o_proj, residual, norm, MLP, residual) -> x_i, layer i pre-attention
    ...
    G_36  : layer 35 post-attention -> final norm

pre-attention = input RMSNorm, q/k/v projections, q/k RMSNorm, RoPE, and the KV write into a static
per-layer cache at the step's position (index_copy_ with a device index; the reference concatenates the
same values).  The graphs are independent of the KV length and are captured once per process.  The
attention core between them runs eagerly with exactly the reference call: repeat_kv of the first L
cache rows, then LocalAttention -> F.scaled_dot_product_attention (math backend) with the 4D mask
rows of the reference.  Every value-producing op is the reference op on the same values and shapes,
so the hidden states are bitwise identical to the eager forward.
"""
import os
from ...fast import enabled as _fast_on, sm90 as _sm90

import torch
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask

from ..model.exact_rope import fused_rope, fused_rope_enabled, rope_supported
from ..model.exact_decode_attn import decode_attention, fast_decode_attn_enabled
from ..model.exact_fused2 import add_rmsnorm, add_rmsnorm_supported, lm_fuse2_enabled, qk_rope_rows, silu_mul
from ..model.exact_gemm import lm_tgemm_enabled, lm_tgemm2_enabled, m2_gate_up_silu, m2_linear, m2_linear_splitk, m2_qkv
from ..model.exact_gemm import m2_linear_splitk_f32a
from .graph_replay import lm_fewer_enabled, raw_ok, replay_raw
from ..model.exact_xattn import XATTN_LMAX, XATTN_LMIN, xattn, xattn_enabled
from ..model.exact_decode_attn import (decode_attention_cached, expand_into, lm_lean_enabled, sdpa_math_scale,
                                        step_rows)
from ..model.exact_decode_attn import supported as decode_attn_supported


def llm_graph_enabled():
    return _fast_on("LM_GRAPH")


def runner_supported(model):
    cfg = model.config
    try:
        from ..model.sequence_parallel_utils import get_sequence_parallel_size
        if get_sequence_parallel_size() != 1:
            return False
    except Exception:
        return False
    vc = cfg.visual_config
    return (not model.training
            and not cfg.audio_config.enable
            and vc.enable
            and not bool(getattr(vc, "rar_random_order", False))
            and not cfg.attention_qkv_pack
            and cfg.position_embedding_type.lower() == "rope"
            and not cfg.sparse_attention_layers
            and model.model.visual_bridge_model is not None)


class DecodeGraphRunner:
    """Persistent per model: static buffers (KV capacity >= the generation length) and the captured
    graphs survive across generate() calls; reset() loads one generation's prefill state."""

    @classmethod
    def for_generation(cls, model, prefill_past, prompt_mask, max_length, first_positions):
        runner = getattr(model, "_gstar_decode_runner", None)
        k0 = prefill_past[0][0]
        if runner is None or not runner.compatible(k0, max_length):
            runner = cls(model, k0, max(2048, int(max_length)))
            model._gstar_decode_runner = runner
        runner.reset(prefill_past, prompt_mask, max_length, first_positions)
        return runner

    def __init__(self, model, k0, capacity):
        self.model = model
        m = model.model
        self.layers = list(m.layers)
        self.dev = k0.device
        self.dtype = k0.dtype
        self.B, self.Hkv, _, self.D = k0.shape
        self.cap = int(capacity)
        hidden = m.config.hidden_size
        self.KB = [torch.zeros(self.B, self.Hkv, self.cap, self.D, dtype=self.dtype, device=self.dev)
                   for _ in self.layers]
        self.VB = [torch.zeros_like(kb) for kb in self.KB]
        self.lean = lm_lean_enabled()
        if self.lean:
            hq = m.config.num_attention_heads
            # persistent fp32 math-SDPA operands: k_acc * s and v_acc for every query head
            self.KS = [torch.zeros(self.B, hq, self.cap, self.D, dtype=torch.float32, device=self.dev)
                       for _ in self.layers]
            self.VF = [torch.zeros_like(ks) for ks in self.KS]
            self.kscale = sdpa_math_scale(self.D)
        # fused kernels (need the lean caches): q/k norm + RoPE + cache rows, add+RMSNorm, silu*up
        self.Af32 = torch.zeros(self.B, m.config.num_attention_heads, 1, self.D, dtype=torch.float32, device=self.dev)
        self.fuse2 = self.lean and lm_fuse2_enabled() and self.D == 128 and all(
            l.self_attn.q_norm.variance_epsilon == l.self_attn.k_norm.variance_epsilon for l in self.layers)
        # with the fused kernels: bit-exact Triton decode GEMMs for the MLP (gate+up+SiLU fused, down)
        self.tgemm = self.fuse2 and lm_tgemm_enabled() and all(
            l.mlp.gate_proj.bias is None and l.mlp.up_proj.bias is None and l.mlp.down_proj.bias is None
            for l in self.layers)
        # bit-exact replicas of cuBLAS's split-K q/k/v (one launch) and o_proj decode GEMMs
        self.tgemm2 = self.fuse2 and lm_tgemm2_enabled() and all(
            a.q_proj.bias is None and a.k_proj.bias is None and a.v_proj.bias is None and a.o_proj.bias is None
            for a in (l.self_attn for l in self.layers))
        # opt-in (GSTAR_FAST_LM_RAW_REPLAY=1): graph pieces launched without replay()'s no-op RNG prologue (2 fill_ kernels each), and
        # o_proj reading the fp32 attention output directly (its bf16 rounding on the load)
        self.fewer = self.fuse2 and lm_fewer_enabled()
        self.raw = {}
        # bit-exact fused decode attention from the bf16 caches for XATTN_LMIN <= L <= XATTN_LMAX: the kernels
        # replicate cuBLAS's batched-gemv orders for GQA 4 with B * heads = 64 (cuBLAS picks other gemv kernels,
        # with other orders, for other batch counts)
        hq = m.config.num_attention_heads
        self.xattn = self.fuse2 and xattn_enabled() and hq == 4 * self.Hkv and self.B * hq == 64
        self.xattn_checked = False
        # with the fused attention: the whole decode step as one CUDA graph for L in the XATTN range (every op
        # reads L from device memory), own memory pool
        self.xgraph = self.xattn and _fast_on("LM_GRAPH")
        self.pool_full = None
        if self.xattn:
            self.L_dev = torch.zeros(1, dtype=torch.int32, device=self.dev)
            self.sc_buf = torch.zeros(self.B, m.config.num_attention_heads, 1024, dtype=torch.float32, device=self.dev)
        self.mask_full = torch.zeros(self.B, 1, 1, self.cap, dtype=self.dtype, device=self.dev)
        # static step inputs / cross-piece buffers
        self.ids = torch.zeros(self.B, 1, dtype=torch.long, device=self.dev)
        # the previous position's codes: [B * mms^2, depth] (the spatial merger folds mms^2 positions into one)
        mms = int(getattr(model.config.visual_config, "merger_merge_size", 1) or 1)
        self.vt = torch.zeros(self.B * mms * mms, len(model.config.visual_quantizer_config.codebook_sizes),
                              dtype=torch.long, device=self.dev)
        self.pos = torch.zeros(self.B, 1, dtype=torch.long, device=self.dev)
        self.kvidx = torch.zeros(1, dtype=torch.long, device=self.dev)
        self.A = torch.zeros(self.B, 1, hidden, dtype=self.dtype, device=self.dev)
        self.X0 = torch.zeros(self.B, 1, hidden, dtype=self.dtype, device=self.dev)
        self.Q0 = None
        self.cos, self.sin = self.layers[0].self_attn.rotary_emb.device_tables(self.dev)
        self.graphs = {}
        self.outs = {}
        self.pool = None
        self.stream = None

    def compatible(self, k0, max_length):
        return (k0.device == self.dev and k0.dtype == self.dtype and tuple(k0.shape[:2]) == (self.B, self.Hkv)
                and k0.shape[3] == self.D and int(max_length) <= self.cap)

    def reset(self, prefill_past, prompt_mask, max_length, first_positions):
        P = prefill_past[0][0].shape[2]
        self.Lmax = int(max_length)
        for layer in self.layers:
            if layer.self_attn.rotary_emb.max_seq_len_cached < self.Lmax:
                raise RuntimeError("RoPE table shorter than the generation")
        for i, (k, v) in enumerate(prefill_past):
            self.KB[i][:, :, :P].copy_(k)
            self.VB[i][:, :, :P].copy_(v)
            if self.lean:
                expand_into(self.KB[i], self.KS[i], P, self.kscale)
                expand_into(self.VB[i], self.VF[i], P, None)
        # the reference's 4D mask at every length L is the first L columns of the full-length one
        full2d = torch.cat([prompt_mask, prompt_mask.new_ones((self.B, self.Lmax - prompt_mask.shape[1]))], dim=-1)
        like = torch.empty(self.B, 1, self.A.shape[-1], dtype=self.dtype, device=self.dev)
        self.mask_full[..., :self.Lmax].copy_(
            _prepare_4d_causal_attention_mask(full2d, (self.B, 1), like, self.Lmax - 1))
        self.pos.copy_(first_positions.view(self.B, 1))
        self.kvidx.fill_(P)
        self.L = P                    # cached rows; the next token goes to index L
        self.steps = 0

    # ---------------------------------------------------------------- pieces (reference ops)
    def _embed(self, with_tokens):
        m = self.model.model
        inputs_embeds = m.embed_tokens(self.ids)
        final, _, _ = m.get_visual_embed_given_tokens(
            self.ids, inputs_embeds, self.vt if with_tokens else None, rar_generation_step=None, group_index=None)
        return final

    def _pre(self, i, x):
        from ..model.modeling_omni import apply_rotary_pos_emb
        layer = self.layers[i]
        att = layer.self_attn
        h = layer.input_layernorm(x)
        bsz, q_len = h.shape[:2]
        q = att.q_proj(h)
        k = att.k_proj(h)
        v = att.v_proj(h)
        q = att.q_norm(q.view(bsz, q_len, -1, att.head_dim)).transpose(1, 2)
        k = att.k_norm(k.view(bsz, q_len, -1, att.head_dim)).transpose(1, 2)
        v = v.view(bsz, q_len, -1, att.head_dim).transpose(1, 2)
        if fused_rope_enabled() and rope_supported(q, k, self.cos, self.sin, self.pos):
            q, k = fused_rope(q, k, self.cos, self.sin, self.pos)     # bit-exact single kernel
        else:
            q, k = apply_rotary_pos_emb(q, k, self.cos, self.sin, self.pos)
        if self.lean:
            # q_s, and row L-1 of the fp32 caches (the bf16 caches are only needed for the prefill rows)
            return step_rows(q, k, v, self.KS[i], self.VF[i], self.kvidx)
        self.KB[i].index_copy_(2, self.kvidx, k)
        self.VB[i].index_copy_(2, self.kvidx, v)
        return q

    def _pre_from_h(self, i, h):
        att = self.layers[i].self_attn
        if self.tgemm2:
            q, k, v = m2_qkv(h, att.q_proj.weight, att.k_proj.weight, att.v_proj.weight)
        else:
            q = att.q_proj(h)
            k = att.k_proj(h)
            v = att.v_proj(h)
        if self.xattn:   # also the bf16 cache rows (k_embed, v) read by the fused attention
            return qk_rope_rows(q, k, v, att.q_norm, att.k_norm, self.cos, self.sin, self.pos, self.kvidx,
                                self.KS[i], self.VF[i], self.kscale, self.KB[i], self.VB[i])
        return qk_rope_rows(q, k, v, att.q_norm, att.k_norm, self.cos, self.sin, self.pos, self.kvidx,
                            self.KS[i], self.VF[i], self.kscale)

    def _post_fused(self, i, x):
        """Layer i after attention, up to (residual, mlp_out) - the final residual add is fused with the next norm."""
        layer = self.layers[i]
        att = layer.self_attn
        # the eager attention left matmul(attn, v_acc) in Af32; .to(bf16) + transpose/reshape as the reference
        if self.fewer and self.tgemm2:
            # [B, H, 1, D] fp32 == the transposed [B, 1, H*D] element order (size-1 dim): view, cast on the load
            o = att.resid_dropout(m2_linear_splitk_f32a(self.Af32.view(self.B, 1, -1), att.o_proj.weight))
        else:
            a = self.Af32.to(self.dtype).transpose(1, 2).reshape(self.B, 1, -1)
            o = att.resid_dropout(m2_linear_splitk(a, att.o_proj.weight) if self.tgemm2 else att.o_proj(a))
        residual, hn = add_rmsnorm(x, o, layer.post_attention_layernorm)
        mlp = layer.mlp
        if self.tgemm:
            m = m2_gate_up_silu(hn, mlp.gate_proj.weight, mlp.up_proj.weight, silu_mul)
            return residual, mlp.ffn_dropout(m2_linear(m, mlp.down_proj.weight))
        m = silu_mul(mlp.gate_proj(hn), mlp.up_proj(hn))
        return residual, mlp.ffn_dropout(mlp.down_proj(m))

    def _post(self, i, x):
        layer = self.layers[i]
        att = layer.self_attn
        o = att.resid_dropout(att.o_proj(self.A))
        hs = x + o
        residual = hs
        hs = layer.post_attention_layernorm(hs)
        hs = layer.mlp(hs)
        return residual + hs

    def _attn(self, i, q, L):
        if self.lean:
            out = decode_attention_cached(q, self.KS[i], self.VF[i], L, self.mask_full[..., :L], self.dtype)
            return out.transpose(1, 2).reshape(self.B, 1, -1)
        if fast_decode_attn_enabled() and decode_attn_supported(q, self.KB[i]):
            # bit-exact: the reference math-SDPA tensors built by two Triton kernels (exact_decode_attn)
            out = decode_attention(q, self.KB[i], self.VB[i], L, self.mask_full[..., :L])
            return out.transpose(1, 2).reshape(self.B, 1, -1)
        att = self.layers[i].self_attn
        k = att._repeat_kv(self.KB[i][:, :, :L], q.size(1))
        v = att._repeat_kv(self.VB[i][:, :, :L], q.size(1))
        out = att.attention(q, k, v, attn_mask=self.mask_full[..., :L], use_flash=False)
        return out.reshape(self.B, 1, -1)

    def _full_step(self, with_tokens):
        """The whole fused decode step: embed, 36 x (q/k/v + rope rows, fused exact attention, o_proj / MLP),
        final norm - the same ops as the pieces + the eager attention, in the same order."""
        x = self._embed(with_tokens=with_tokens)
        q = self._pre_from_h(0, self.layers[0].input_layernorm(x))
        n = len(self.layers)
        for i in range(n):
            xattn(q, self.KB[i], self.VB[i], self.mask_full, self.L_dev, self.sc_buf, self.Af32, self.kscale)
            residual, d = self._post_fused(i, x)
            if i + 1 == n:
                return (add_rmsnorm(residual, d, self.model.model.norm)[1],)
            x, h = add_rmsnorm(residual, d, self.layers[i + 1].input_layernorm)
            q = self._pre_from_h(i + 1, h)

    def _piece(self, key):
        if key[0] == "full":
            return self._full_step(with_tokens=(key[1] == 1))
        if self.fuse2:
            return self._piece_fused(key)
        kind, i = key
        if kind == "g0":
            x0 = self._embed(with_tokens=(i == 1))
            q0 = self._pre(0, x0)
            self.X0.copy_(x0)
            if self.Q0 is None:
                self.Q0 = torch.empty_like(q0)
            self.Q0.copy_(q0)
            return (self.X0, self.Q0)
        x_in = self.X0 if i == 1 else self.outs[("g", i - 1)][0]
        x = self._post(i - 1, x_in)
        if i == len(self.layers):
            return (self.model.model.norm(x),)
        return (x, self._pre(i, x))

    def _piece_fused(self, key):
        kind, i = key
        if kind == "g0":
            x0 = self._embed(with_tokens=(i == 1))
            q0 = self._pre_from_h(0, self.layers[0].input_layernorm(x0))
            self.X0.copy_(x0)
            if self.Q0 is None:
                self.Q0 = torch.empty_like(q0)
            self.Q0.copy_(q0)
            return (self.X0, self.Q0)
        x_in = self.X0 if i == 1 else self.outs[("g", i - 1)][0]
        residual, d = self._post_fused(i - 1, x_in)
        if i == len(self.layers):
            return (add_rmsnorm(residual, d, self.model.model.norm)[1],)
        x_i, h_i = add_rmsnorm(residual, d, self.layers[i].input_layernorm)
        return (x_i, self._pre_from_h(i, h_i))

    def _run(self, key):
        g = self.graphs.get(key)
        if g is not None:
            if self.raw.get(key):
                replay_raw(g)
            else:
                g.replay()
            return self.outs[key]
        # first use: run eagerly on the capture stream (lazy library init), capture, then replay.
        # The eager run computes the same values the replay writes (KV rows included).
        if self.stream is None:
            self.stream = torch.cuda.Stream(device=self.dev)
            self.pool = torch.cuda.graph_pool_handle()
        if key[0] == "g0" and self.Q0 is None:
            self._piece(key)           # allocate the static q buffer outside the capture
        self.stream.wait_stream(torch.cuda.current_stream(self.dev))
        with torch.cuda.stream(self.stream):
            self._piece(key)
        torch.cuda.current_stream(self.dev).wait_stream(self.stream)
        g = torch.cuda.CUDAGraph()
        pool = self.pool
        if key[0] == "full":
            if self.pool_full is None:
                self.pool_full = torch.cuda.graph_pool_handle()
            pool = self.pool_full
        with torch.cuda.graph(g, pool=pool, stream=self.stream):
            out = self._piece(key)
        self.graphs[key] = g
        self.outs[key] = out
        self.raw[key] = self.fewer and raw_ok(g)
        g.replay()
        return out

    # ---------------------------------------------------------------- one decode step
    @torch.no_grad()
    def step(self, ids, vision_tokens):
        """ids: [B, 1] token ids of this step; vision_tokens: [B * mms^2, depth] codes of the previous
        position or None (no image-pad in this step). Returns the final-norm hidden state [B, 1, hidden]."""
        L = self.L + 1
        if L > self.Lmax:
            raise RuntimeError("decode past the static cache")
        if self.steps:
            self.pos.add_(1)
            self.kvidx.add_(1)
        self.ids.copy_(ids)
        if vision_tokens is not None:
            self.vt.copy_(vision_tokens)
        _, q = self._run(("g0", 1 if vision_tokens is not None else 0))
        n = len(self.layers)
        use_x = self.xattn and XATTN_LMIN <= L <= XATTN_LMAX
        if use_x:
            self.L_dev.fill_(L)
            if self.xgraph and self.xattn_checked:
                out = self._run(("full", 1 if vision_tokens is not None else 0))
                self.L = L
                self.steps += 1
                return out[0]
        for i in range(n):
            if self.fuse2:
                if use_x and self.xattn_checked:
                    xattn(q, self.KB[i], self.VB[i], self.mask_full, self.L_dev, self.sc_buf, self.Af32, self.kscale)
                    continue_ref = False
                else:
                    continue_ref = True
                if continue_ref:
                    attn = torch.matmul(q, self.KS[i][:, :, :L].transpose(-2, -1))
                    attn.add_(self.mask_full[..., :L])
                    attn = torch.softmax(attn, -1)
                    torch.matmul(attn, self.VF[i][:, :, :L], out=self.Af32)     # same bmm, static output
                    if use_x and not self.xattn_checked:
                        # live check once: the fused kernels must give the same bits as the reference ops
                        ref = self.Af32.clone()
                        xattn(q, self.KB[i], self.VB[i], self.mask_full, self.L_dev, self.sc_buf, self.Af32,
                              self.kscale)
                        good = torch.equal(self.Af32, ref)
                        self.xattn_checked = True
                        if not good:
                            print("[gstar.fast] fused decode attention: live check failed, reference attention kept", flush=True)
                            self.xattn = False
                            use_x = False
                            self.Af32.copy_(ref)
            else:
                self.A.copy_(self._attn(i, q, L))
            out = self._run(("g", i + 1))
            if i + 1 < n:
                q = out[1]
        self.L = L
        self.steps += 1
        return out[0]
