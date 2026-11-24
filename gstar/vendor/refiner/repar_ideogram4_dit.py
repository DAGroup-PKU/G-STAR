"""RepAR-conditioned Ideogram4 DiT.

Wraps a pretrained `Ideogram4DiT` with the same unified *zero-init additive*
conditioning path as Krea2 (`repar_common`), plus an optional in-context adapter.
A frozen RepAR encoder turns the LQ image into `z_q`; `ConditionEncoder` aligns it
to the DiT image-token grid and `InjectionHead`s add zero-init residuals to the
image tokens; `IncontextAdapter` (optional) turns z_q into pseudo-text context
tokens that replace the empty prompt on the single-stream joint-attention route.

Ideogram4 differences from Krea2:
  * the base `Ideogram4DiT.forward` already handles the fully-assembled packed
    sequence (text+image interleaved, MRoPE positions, cu_seqlens varlen) and
    scatters inject residuals to image positions via the `indicator` mask — so
    this wrapper only builds `cond`/`inject_fn`/`context` INSIDE the engine call
    (ZeRO-3-safe) and delegates; the packed *assembly* is param-free and lives in
    the pipeline's `model_fn_repar_ideogram4_packed`.
  * time condition for the injection heads is the DiT's adaLN input (adanln_dim),
    from `base.t_embedding` + `base.adaln_proj`.
  * inject residual dims: emb_dim for -1/i, in_channels for -2 (pre input_proj).

Zero-init injection ⇒ step-0 output is byte-identical to the base model; intended
for FULL finetune (base + condition path all trainable).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .ideogram4_dit import Ideogram4DiT, LLM_TOKEN_INDICATOR, dit_fast_enabled
from .repar_common import ConditionEncoder, InjectionHead, IncontextAdapter, dit_fast2_enabled, _c2, _film_fp32


class RepARIdeogram4DiT(nn.Module):
    def __init__(
        self,
        base: Ideogram4DiT,
        code_dim: int = 1152,
        cond_channels: int = 1152,
        inject_interval: int = 4,
        condition_time_adain: bool = True,
        use_rmsnorm: bool = False,
        resize_mode: str = "bilinear",
        num_resblocks: int = 1,
        inject_latent: bool = False,
        enable_incontext_inject: bool = False,
        incontext_adapter_depth: int = 2,
        incontext_adapter_heads: int = 8,
        incontext_adapter_dim: int = 2048,
    ):
        super().__init__()
        self.base = base
        self.code_dim = code_dim
        self.condition_time_adain = bool(condition_time_adain)
        cfg = base.config
        D = cfg.emb_dim                       # 4608 — post input_proj / post-block residual dim
        time_dim = cfg.adanln_dim             # 512  — adaLN time condition dim
        num_layers = cfg.num_layers

        # inject points: every `inject_interval` blocks (≈ multi-layer ControlNet).
        self.inject_points = [i for i in range(num_layers) if i % inject_interval == 0]
        self.point_to_head = {p: k for k, p in enumerate(self.inject_points)}

        self.cond_encoder = ConditionEncoder(
            code_dim, cond_channels, time_dim=time_dim,
            num_resblocks=num_resblocks, resize_mode=resize_mode)
        self.heads = nn.ModuleList([
            InjectionHead(cond_channels, time_dim, D, use_rmsnorm=use_rmsnorm)
            for _ in self.inject_points
        ])

        # Optional latent-space (-2) injection before input_proj (dim in_channels).
        self.inject_latent = bool(inject_latent)
        if self.inject_latent:
            self.latent_head = InjectionHead(
                cond_channels, time_dim, cfg.in_channels, use_rmsnorm=use_rmsnorm)

        # Optional in-context: z_q -> pseudo-text context tokens (emb_dim).
        self.enable_incontext_inject = bool(enable_incontext_inject)
        if self.enable_incontext_inject:
            self.incontext_adapter = IncontextAdapter(
                code_dim, D, depth=incontext_adapter_depth,
                heads=incontext_adapter_heads, inner_dim=incontext_adapter_dim)

    def __getattr__(self, name):
        """Transparent stand-in for the base Ideogram4DiT (delegates config etc.)."""
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(super().__getattr__("base"), name)

    # ------------------------------------------------------------------
    def _time_cond_per_sample(self, t_samples: torch.Tensor, device, dtype) -> torch.Tensor:
        """Per-sample adaLN time condition (n, adanln_dim). condition_time_adain=
        False -> zeros (static condition). One base.t_embedding + base.adaln_proj
        call over the (n,) batch (deterministic count -> ZeRO-3 safe)."""
        n = t_samples.shape[0]
        if not self.condition_time_adain:
            return torch.zeros(n, self.base.config.adanln_dim, device=device, dtype=dtype)
        tt = self.base.t_embedding(t_samples.to(dtype))            # (n, emb_dim)
        return F.silu(self.base.adaln_proj(tt)).to(dtype)          # (n, adanln_dim)

    # ------------------------------------------------------------------
    def forward(
        self,
        *,
        llm_features: torch.Tensor,           # (1, S, D) full packed context (text positions filled)
        x: torch.Tensor,                      # (1, S, in_channels) packed noise tokens
        t: torch.Tensor,                      # (1, S) per-token time
        position_ids: torch.Tensor,           # (1, S, 3)
        indicator: torch.Tensor,              # (1, S)
        segment_ids: torch.Tensor = None,     # ignored (packed uses cu_seqlens); base-T2I passthrough
        cu_seqlens: torch.Tensor = None,
        max_seqlen: int = None,
        z_q_packed: torch.Tensor = None,      # (1, sum_Lz, code_dim) — None -> base T2I
        z_grid_list: list = None,             # per-sample (zH, zW)
        img_grid_list: list = None,           # per-sample (tH, tW) DiT token grid
        img_len_list: list = None,            # per-sample image-token count (tH*tW)
        t_samples: torch.Tensor = None,       # (n,) per-sample scalar time
        context_precomputed: bool = False,
        inject_fn=None,                       # ignored when z_q given; base-T2I passthrough
        inject_layers=None,
        feature_layers: list[int] | tuple[int, ...] | None = None,
        return_features: bool = False,
        use_gradient_checkpointing: bool = False,
        use_gradient_checkpointing_offload: bool = False,
        z_q_src_override: torch.Tensor = None,
        **kwargs,
    ) -> torch.Tensor:
        base_kw = dict(
            x=x, t=t, position_ids=position_ids, segment_ids=segment_ids,
            indicator=indicator, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen,
            feature_layers=feature_layers, return_features=return_features,
            use_gradient_checkpointing=use_gradient_checkpointing,
            use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
        )
        if z_q_packed is None:                                     # pure base T2I (transparent)
            return self.base.forward(llm_features=llm_features,
                                     context_precomputed=context_precomputed,
                                     inject_fn=inject_fn, inject_layers=inject_layers, **base_kw)

        device = x.device
        dtype = self.cond_encoder.in_proj.weight.dtype     # cond runs in the module (bf16) dtype,
                                                           # NOT x.dtype (base keeps latents fp32)
        z_q_src = z_q_src_override if z_q_src_override is not None else z_q_packed
        z_q_packed = z_q_packed.to(dtype)
        n = len(img_grid_list)
        ce = self.cond_encoder
        z_lens = [int(g[0]) * int(g[1]) for g in z_grid_list]      # zH*zW per sample

        # ── Per-sample time vector (ONE t_embedding + adaln_proj over the (n,) batch)
        c_vec = self._time_cond_per_sample(t_samples, device, dtype)   # (n, adanln_dim)

        # ── Batched cond-encode (ZeRO-3 call-count invariant): norm once, time_mlp
        #    once; per-sample FiLM-apply + interpolate are param-free; refine once. ──
        cache = getattr(self, "_decode_cache", None) if (n == 1 and dit_fast_enabled()) else None
        if cache is not None and "z_all" in cache and cache["z_all_src"] is z_q_src:
            z_all = cache["z_all"]                                 # t-independent: once per decode
        else:
            z_all = ce.norm(z_q_packed)                            # (1, sum_Lz, code_dim)
            if cache is not None:
                cache["z_all"], cache["z_all_src"] = z_all, z_q_src
        s_all, b_all, g_all = ce.time_mlp(c_vec).chunk(3, dim=-1)  # each (n, code_dim)
        cond_interp, off = [], 0
        fast2 = (cache is not None and dit_fast2_enabled() and not torch.is_grad_enabled()
                 and ce.resize_mode != "nearest" and img_grid_list is not None)
        if fast2:
            # bilinear weights sum to one, so the per-channel FiLM commutes with the resize: resize the
            # t-independent z_all once per decode (fp32) and apply the step's FiLM after it (other rounding)
            zH, zW = int(z_grid_list[0][0]), int(z_grid_list[0][1])
            tH, tW = int(img_grid_list[0][0]), int(img_grid_list[0][1])
            if cache.get("interp_src") is not z_q_src:
                h0 = z_all[:, :z_lens[0]].transpose(1, 2).reshape(1, self.code_dim, zH, zW).float()
                if (zH, zW) != (tH, tW):
                    h0 = F.interpolate(h0, size=(tH, tW), mode=ce.resize_mode, align_corners=False)
                cache["interp"] = h0.reshape(1, self.code_dim, tH * tW).transpose(1, 2).contiguous()
                cache["interp_src"] = z_q_src
            cond_interp.append(_c2("film", _film_fp32)(cache["interp"], s_all[0:1].unsqueeze(1),
                                                         b_all[0:1].unsqueeze(1), g_all[0:1].unsqueeze(1)))
        for i in range(n if not fast2 else 0):
            h = z_all[:, off:off + z_lens[i]]
            off += z_lens[i]
            h = ce._film_apply(h, s_all[i:i + 1], b_all[i:i + 1], g_all[i:i + 1])
            zH, zW = int(z_grid_list[i][0]), int(z_grid_list[i][1])
            tH, tW = int(img_grid_list[i][0]), int(img_grid_list[i][1])
            h = h.transpose(1, 2).reshape(1, self.code_dim, zH, zW)
            if (zH, zW) != (tH, tW):
                align = None if ce.resize_mode == "nearest" else False
                h = F.interpolate(h.float(), size=(tH, tW), mode=ce.resize_mode, align_corners=align).to(dtype)
            cond_interp.append(h.reshape(1, self.code_dim, tH * tW).transpose(1, 2))  # (1, N_i, code_dim)
        cond_cat = cond_interp[0] if fast2 else torch.cat(cond_interp, dim=1)   # (1, sum_N_img, code_dim)

        # Per-image-token time condition (drives refine ResBlocks + InjectionHead adaLN).
        fast = (n == 1 and dit_fast_enabled() and not torch.is_grad_enabled()
                and img_grid_list is not None and not return_features)
        if fast and self.condition_time_adain:
            # every image token carries the same time row: (1, adanln_dim) broadcasts in
            # ResBlock / InjectionHead exactly like the expanded (1, N, adanln_dim) tensor
            c_packed = c_vec[0:1]
        elif self.condition_time_adain:
            c_segs = [c_vec[i:i + 1].expand(cond_interp[i].shape[1], -1) for i in range(n)]
            c_packed = torch.cat(c_segs, dim=0).unsqueeze(0)       # (1, sum_N_img, adanln_dim)
        else:
            c_packed = torch.zeros(1, cond_cat.shape[1], self.base.config.adanln_dim, device=device, dtype=dtype)

        cond_packed = ce.refine(cond_cat, c_packed)                # (1, sum_N_img, cond_width)  — ONE call

        def inject_fn(point):
            if point == -2:
                return self.latent_head(cond_packed, c_packed)     # (1, sum_N_img, in_channels)
            return self.heads[self.point_to_head[point]](cond_packed, c_packed)  # (1, sum_N_img, emb_dim)

        # ── In-context: z_q -> context tokens via ONE block-diagonal adapter call,
        #    scattered onto the TEXT positions of llm_features (indicator order ==
        #    adapter output order). Text len == L_z per sample (placeholder sizing). ──
        if self.enable_incontext_inject:
            if cache is not None and "ctx" in cache and cache["ctx_src"] is z_q_src:
                ctx_cat = cache["ctx"]                             # t-independent: once per decode
            else:
                ctx_cat = self.incontext_adapter(z_q_packed, seqlens=z_lens)   # (1, sum_Lz, emb_dim)
                if cache is not None:
                    cache["ctx"], cache["ctx_src"] = ctx_cat, z_q_src
            llm_features = llm_features.clone()
            if fast2:
                # one sample: the text positions are the first L_z tokens (no boolean-mask index_put / sync)
                llm_features[:, :z_lens[0]] = ctx_cat.to(llm_features.dtype)
            else:
                text_bool = (indicator == LLM_TOKEN_INDICATOR)         # (1, S)
                llm_features[text_bool] = ctx_cat.reshape(-1, ctx_cat.shape[-1]).to(llm_features.dtype)
            context_precomputed = True

        layers = set(self.inject_points)
        if self.inject_latent:
            layers.add(-2)
        if fast:
            s0 = int(z_lens[0])
            self.base._fast_image_range = (s0, s0 + int(img_grid_list[0][0]) * int(img_grid_list[0][1]))
        else:
            self.base._fast_image_range = None
        return self.base.forward(
            llm_features=llm_features, context_precomputed=context_precomputed,
            inject_fn=inject_fn, inject_layers=layers, **base_kw)

    # ------------------------------------------------------------------
    def condition_parameters(self):
        """Only the added conditioning params (ConditionEncoder + heads + adapter)."""
        params = list(self.cond_encoder.parameters()) + list(self.heads.parameters())
        if self.inject_latent:
            params += list(self.latent_head.parameters())
        if self.enable_incontext_inject:
            params += list(self.incontext_adapter.parameters())
        return params
