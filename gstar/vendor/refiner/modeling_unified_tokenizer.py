"""Unified RepAR tokenizer + Ideogram4 Refiner —— 单文件封装, AutoModel 可加载。

一个 HF 文件夹即可: model.safetensors(合并权重) + config.json + 本文件 + 若干扁平组件 .py。
    from transformers import AutoModel
    m = AutoModel.from_pretrained("<folder>", trust_remote_code=True).to("cuda").eval()
    encoded      = m.encode(pixel_values, grid_thw)
    pixels       = m.decode_to_pixels(encoded.indices, grid_h, grid_w, target_height, target_width)  # Refiner
    pil_vit      = m.reconstruct(image=img, mode="vit")          # RepAR tokenizer 自身 ViT 解码重建
    pil_refiner  = m.reconstruct(image=img, mode="refiner")      # Ideogram4 Refiner 精修重建 (无 color fix)
    pil_refinercf= m.reconstruct(image=img, mode="refiner_cf")   # Refiner 精修 + wavelet color fix

封装的三部分:
  - repar_tok : RepAR 视觉 tokenizer (encoder + VQ + ViT decoder), 权重 self.repar_tok.*
  - dit       : 微调后的 Ideogram4 Refiner DiT (base + 条件注入), 权重 self.dit.*
  - vae       : Flux2 VAE, 权重 self.vae.*
只依赖本文件夹的扁平组件 .py + 运行环境的 torch/transformers/safetensors/timm/einops。
"""
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import PreTrainedModel

from .ideogram4_dit import (Ideogram4DiT, Ideogram4Config, LLM_TOKEN_INDICATOR,
                            OUTPUT_IMAGE_INDICATOR, IMAGE_POSITION_OFFSET)
from .flux2_vae import Flux2VAE
from .ideogram4_vae import decode as vae_decode
from .repar_ideogram4_dit import RepARIdeogram4DiT
from .flow_match import FlowMatchScheduler
from .modeling_bar import BARTokenizer, EncodeOutput
from .configuration_bar import BARTokenizerConfig
from .configuration_unified_tokenizer import GStarRefinerConfig
try:
    from .wavelet_color_fix import wavelet_color_fix, adain_color_fix
except Exception:
    wavelet_color_fix = adain_color_fix = None

# HF trust_remote_code 的动态加载器只拷贝**本文件直接**的相对 import, 不递归二级依赖。
# 因此把所有扁平组件在顶层强制 import 一遍, 确保它们都被拷进 HF cache (否则运行时 FileNotFound)。
from .repar_common import ConditionEncoder as _fc1  # noqa: F401
from .attention import attention_forward_varlen as _fc2  # noqa: F401
from .gradient_checkpoint import gradient_checkpoint_forward as _fc3  # noqa: F401
from .base_model import BaseModel as _fc4  # noqa: F401
from .res_mod import ResMod as _fc5  # noqa: F401
from .blocks import UnifiedEncoder as _fc6  # noqa: F401
from .quantizer import RVQ as _fc7  # noqa: F401
from .tokenizer import RVQTokenizer as _fc8  # noqa: F401
from .configuration_bar import BARTokenizerConfig as _fc9  # noqa: F401


# ── 内联: 一个样本的 packed 布局 (copy from diffsynth ideogram4.py) ──
def build_ideogram4_sample_layout(text_len, grid_h, grid_w, device):
    img_len = grid_h * grid_w
    L = text_len + img_len
    h_idx = torch.arange(grid_h, device=device).view(-1, 1).expand(grid_h, grid_w).reshape(-1)
    w_idx = torch.arange(grid_w, device=device).view(1, -1).expand(grid_h, grid_w).reshape(-1)
    t_idx = torch.zeros_like(h_idx)
    image_pos = torch.stack([t_idx, h_idx, w_idx], dim=1) + IMAGE_POSITION_OFFSET
    position_ids = torch.zeros(1, L, 3, dtype=torch.long, device=device)
    if text_len > 0:
        tp = torch.arange(text_len, device=device)
        position_ids[0, :text_len] = torch.stack([tp, tp, tp], dim=1)
    position_ids[0, text_len:] = image_pos
    indicator = torch.zeros(1, L, dtype=torch.long, device=device)
    indicator[0, :text_len] = LLM_TOKEN_INDICATOR
    indicator[0, text_len:] = OUTPUT_IMAGE_INDICATOR
    return {"position_ids": position_ids, "indicator": indicator,
            "text_len": text_len, "img_len": img_len, "L": L}


# ── 内联: 多样本一次 packed varlen forward (copy from diffsynth repar_ideogram4.py) ──
def model_fn_repar_ideogram4_packed(dit_callable, patch, latents_list, context_list, t_list,
                                    z_q_list, z_grid_list, height_list, width_list,
                                    context_precomputed=True):
    device = latents_list[0].device
    n = len(latents_list)
    C = latents_list[0].shape[-1]
    D = context_list[0].shape[-1]
    seg_lens, img_slices = [], []
    seqs_x, seqs_pos, seqs_ind, seqs_t = [], [], [], []
    img_grid_list, off = [], 0
    for i in range(n):
        gh = height_list[i] // patch
        gw = width_list[i] // patch
        img_grid_list.append((gh, gw))
        text_len = context_list[i].shape[1]
        lay = build_ideogram4_sample_layout(text_len, gh, gw, device)
        L = lay["L"]
        x_seg = torch.cat([torch.zeros(1, text_len, C, dtype=latents_list[i].dtype, device=device),
                           latents_list[i]], dim=1)
        seqs_x.append(x_seg)
        seqs_pos.append(lay["position_ids"])
        seqs_ind.append(lay["indicator"])
        seqs_t.append(torch.full((1, L), float(t_list[i]), dtype=torch.float32, device=device))
        seg_lens.append(L)
        img_slices.append((off + text_len, off + text_len + lay["img_len"]))
        off += L
    x = torch.cat(seqs_x, dim=1)
    position_ids = torch.cat(seqs_pos, dim=1)
    indicator = torch.cat(seqs_ind, dim=1)
    t_packed = torch.cat(seqs_t, dim=1)
    # per-sample times and cumulative lengths filled on the device: a torch.tensor(list, device=cuda) is a pageable
    # host -> device copy, which waits for the stream and so stalls the host once per denoising step (same values)
    t_samples = torch.empty(n, dtype=torch.float32, device=device)
    for i, v in enumerate(t_list):
        t_samples[i].fill_(float(v))
    ctx_full = torch.zeros(1, x.shape[1], D, dtype=context_list[0].dtype, device=device)
    for i in range(n):
        s = sum(seg_lens[:i]); tl = context_list[i].shape[1]
        ctx_full[0, s:s + tl] = context_list[i][0]
    z_q_packed = torch.cat(z_q_list, dim=1)
    cu = torch.zeros(n + 1, dtype=torch.int32, device=device)
    for i in range(n):
        cu[i + 1].fill_(sum(seg_lens[:i + 1]))
    max_seqlen = int(max(seg_lens))
    extra = {"z_q_src_override": z_q_list[0]} if n == 1 else {}
    out = dit_callable(
        llm_features=ctx_full, x=x, t=t_packed, position_ids=position_ids, indicator=indicator,
        cu_seqlens=cu, max_seqlen=max_seqlen, z_q_packed=z_q_packed, z_grid_list=z_grid_list,
        img_grid_list=img_grid_list, t_samples=t_samples, context_precomputed=context_precomputed,
        **extra)
    return [-out[:, s:e] for (s, e) in img_slices]


class GStarRefiner(PreTrainedModel):
    """RepAR tokenizer + Ideogram4 Refiner 的统一封装。AutoModel 可加载。"""
    config_class = GStarRefinerConfig

    def __init__(self, config: GStarRefinerConfig):
        super().__init__(config)
        self.torch_dtype = torch.bfloat16
        # 1) RepAR tokenizer (encoder + VQ + ViT decoder)
        self.repar_tok = BARTokenizer(BARTokenizerConfig(bar_config=config.bar_config))
        # 2) 微调后的 Ideogram4 Refiner DiT
        base_dit = Ideogram4DiT(Ideogram4Config(**config.dit_config))
        self.dit = RepARIdeogram4DiT(base_dit, **config.refiner_config)
        # 3) Flux2 VAE
        self.vae = Flux2VAE(**config.vae_config)
        self.scheduler = FlowMatchScheduler("Ideogram4")
        # The reference tokenizer signature has no sampling arguments. Keep the
        # signature identical and expose the deterministic Refiner defaults here.
        self.decode_num_inference_steps = int(getattr(config, "decode_num_inference_steps", 52))
        self.decode_seed = int(getattr(config, "decode_seed", 0))
        self.decode_sampler = getattr(config, "decode_sampler", "native")
        self.fixed_sr_scale = int(getattr(config, "fixed_sr_scale", 1))
        if self.fixed_sr_scale < 1:
            raise ValueError(f"fixed_sr_scale must be positive, got {self.fixed_sr_scale}")

    # ---- BaseVisualTokenizer contract -----------------------------------
    # Keep these signatures byte-for-byte aligned with the visual tokenizer interface.
    def encode(self, pixel_values, grid_thw, fake_input: bool = False) -> EncodeOutput:
        dev_type = self.device.type
        with torch.autocast(dev_type, dtype=self.torch_dtype, enabled=(dev_type == "cuda")):
            return self.repar_tok.encode(pixel_values, grid_thw, fake_input=fake_input)

    def decode_to_pixels(self, indices, grid_h, grid_w, target_height, target_width):
        """Decode discrete RepAR tokens through the trained Ideogram4 Refiner.

        The public signature stays identical to export_hf_repo.py, but the
        unified repo's decode semantics are Refiner reconstruction rather than
        the tokenizer's lightweight ViT pixel decoder. ``target_height`` and
        ``target_width`` describe the condition canvas; this fixed-SR export
        returns ``fixed_sr_scale * H/W`` pixels.
        """
        with torch.no_grad():
            z_q = self._indices_to_codebook_quant(indices, grid_h, grid_w)
            return self._decode_refiner_from_zq(
                z_q, (int(grid_h), int(grid_w)),
                int(target_height) * self.fixed_sr_scale,
                int(target_width) * self.fixed_sr_scale,
                self.decode_num_inference_steps, self.decode_seed)

    def _indices_to_codebook_quant(self, indices, grid_h, grid_w):
        """Rebuild the pre-out_proj codebook-space condition used in training."""
        idx = indices.to(device=self.device, dtype=torch.long)
        grid_h, grid_w = int(grid_h), int(grid_w)
        if idx.dim() == 2:
            idx = idx.reshape(1, grid_h * grid_w, idx.shape[-1])
        elif idx.dim() == 3:
            idx = idx.reshape(idx.shape[0], grid_h * grid_w, idx.shape[-1])
        elif idx.dim() == 4:
            if tuple(idx.shape[1:3]) != (grid_h, grid_w):
                raise ValueError(
                    f"indices spatial grid {tuple(idx.shape[1:3])} != {(grid_h, grid_w)}")
            idx = idx.reshape(idx.shape[0], grid_h * grid_w, idx.shape[-1])
        else:
            raise ValueError(f"indices must be 2D/3D/4D, got {tuple(idx.shape)}")

        quant = self.repar_tok.tokenizer.quantize
        if hasattr(quant, "_num_groups") and hasattr(quant, "codebooks"):
            # ProductVQ/MCQ: each index selects one channel-group codebook;
            # concatenate the group embeddings to recover the exact 4096-D
            # `quant_at_depth_with_grad[-1]` consumed by this Refiner.
            if idx.shape[-1] != int(quant._num_groups):
                raise ValueError(
                    f"ProductVQ expects {quant._num_groups} groups, got {idx.shape[-1]}")
            chunks = []
            for group, atom in enumerate(quant.codebooks):
                codebook = atom._get_codebook()
                if bool(getattr(quant, "use_l2_norm", False)):
                    codebook = F.normalize(codebook, dim=-1)
                chunks.append(F.embedding(idx[..., group], codebook))
            z_q = torch.cat(chunks, dim=-1)
        elif hasattr(quant, "_stack_depth_embeds"):
            # RVQ: sum residual-depth embeddings before the decoder out_proj.
            embeds = quant._stack_depth_embeds(idx, to_latent_shape=False)
            z_q = torch.stack(embeds, dim=-2).sum(dim=-2)
        else:
            raise NotImplementedError(
                f"Cannot recover Refiner codebook-space features from {type(quant).__name__} indices")
        return z_q.to(device=self.device, dtype=self.torch_dtype)

    # ---- helpers (脱离 BasePipeline, 自带最小实现) ----
    def _round_to_tokenizer_multiple(self, x):
        # RepAR encoder 的 Qwen3ViT patchify 要求 H,W 能被
        # patch_size × spatial_merge_size 整除 (MCQ g16: 16×2=32)。
        inner = self.repar_tok.tokenizer
        patch = int(getattr(inner, "patch_size", 16) or 16)
        merge = int(getattr(inner, "spatial_merge_size", 1) or 1)
        m = patch * merge
        return (x + m - 1) // m * m

    def _preprocess(self, pil):
        # fp32 运算后再 cast (与训练/推理 to_tensor 完全一致, 保证与原 pipeline 数值对齐)
        a = np.asarray(pil.convert("RGB")).astype(np.float32) / 255.0
        t = torch.from_numpy(a).permute(2, 0, 1) * 2.0 - 1.0       # (3,H,W) fp32 in [-1,1]
        return t.to(device=self.device, dtype=self.torch_dtype)

    @staticmethod
    def _to_pil(t):
        if t.dim() == 4:
            t = t[0]
        # BasePipeline.vae_output_to_image casts directly to uint8 (truncate,
        # not round). Keep this identical so exported/original pixels match.
        a = ((t - (-1)) * (255 / (1 - (-1)))).clip(0, 255).to(torch.uint8)
        return Image.fromarray(a.permute(1, 2, 0).cpu().numpy())

    def _noise(self, shape, seed):
        g = None if seed is None else torch.Generator("cpu").manual_seed(int(seed))
        return torch.randn(shape, generator=g, device="cpu", dtype=torch.float32).to(
            device=self.device, dtype=self.torch_dtype)

    # ---- RepAR encode → 连续 z_q ----
    @torch.no_grad()
    def encode_zq(self, x_hq, sr_scale=1):
        x = x_hq.to(self.device, self.torch_dtype)
        if x.dim() == 3:
            x = x.unsqueeze(0)
        sr = int(sr_scale or 1)
        if sr > 1:
            H, W = x.shape[-2], x.shape[-1]
            x = F.interpolate(x, size=(H // sr, W // sr), mode="bicubic", align_corners=False, antialias=True)
        Hl, Wl = x.shape[-2], x.shape[-1]
        hp, wp = Hl // 16, Wl // 16
        grid = torch.tensor([[1, hp, wp]], dtype=torch.long, device=x.device)
        # Match HFVisualTokenizerAdaptor used by training/original inference:
        # autocast is required by the tokenizer, but must not leak into the
        # Refiner DiT/VAE path (that changes Ideogram4 numerics substantially).
        dev_type = self.device.type
        with torch.autocast(dev_type, dtype=self.torch_dtype, enabled=(dev_type == "cuda")):
            out = self.repar_tok.encode([x[i] for i in range(x.shape[0])], grid)
        inner = self.repar_tok.tokenizer
        ms = int(getattr(inner, "spatial_merge_size", None) or 1)
        if bool(getattr(inner, "merger_after_quant", False)) and ms > 1:
            zH, zW = hp // ms, wp // ms
        else:
            zH, zW = hp, wp
        # Rebuild from the discrete ids even for image reconstruction. The
        # tokenizer's STE `codebook_quant` is mathematically the same codebook
        # value, but bf16 cancellation in z + (quant-z).detach() leaves tiny
        # numerical differences that a long diffusion trajectory can amplify.
        # Using the identical ids->embedding path here guarantees parity with
        # public decode_to_pixels(), which is the AR deployment path.
        z = self._indices_to_codebook_quant(out.indices, zH, zW)
        return z, (zH, zW)

    def _empty_ctx(self, z_q):
        return z_q.new_zeros(1, z_q.shape[1], self.dit.base.config.emb_dim)

    def _decode_refiner_from_zq(self, z_q, z_grid, height, width,
                                num_inference_steps=None, seed=None):
        """Run packed Ideogram4 denoising directly from discrete-token z_q.

        Samplers (config.decode_sampler): "native" - the flow-matching scheduler's steps; "one_step" - one
        evaluation on the one_step_schedule_steps schedule, latents = latent - sigma * pred; "dmad4" - four such
        evaluations with re-noise (1 - next_sigma) * clean + next_sigma * randn from a CPU generator per sample."""
        height, width = int(height), int(width)
        steps = self.decode_num_inference_steps if num_inference_steps is None else int(num_inference_steps)
        seed = self.decode_seed if seed is None else int(seed)
        required = {"one_step": 1, "dmad4": 4}.get(self.decode_sampler)
        if required is not None and steps != required:
            raise ValueError(f"{self.decode_sampler} export requires {required} evaluations, got {steps}")
        patch = self.dit.base.patch_size * 8
        if height % patch or width % patch:
            raise ValueError(f"target size {width}x{height} must be divisible by {patch}")
        if z_q.dim() == 2:
            z_q = z_q.unsqueeze(0)
        expected = int(z_grid[0]) * int(z_grid[1])
        if z_q.shape[1] != expected:
            raise ValueError(f"z_q length {z_q.shape[1]} != grid product {expected}")

        batch = z_q.shape[0]
        z_q_list = [z_q[i:i + 1] for i in range(batch)]
        z_grid_list = [tuple(z_grid)] * batch
        ctx_list = [self._empty_ctx(z) for z in z_q_list]
        latent_h, latent_w = height // patch, width // patch
        channels = self.dit.base.config.in_channels
        latents = [self._noise((1, latent_h * latent_w, channels), int(seed) + i)
                   for i in range(batch)]
        # Re-noise uses an independent reproducible CPU stream per sample.
        # Consume the initial draw so subsequent draws are fresh, not reused.
        generators = [torch.Generator("cpu").manual_seed(seed + i) for i in range(batch)]
        if self.decode_sampler == "dmad4":
            for generator, latent in zip(generators, latents):
                torch.randn(latent.shape, generator=generator, dtype=torch.float32)
        schedule_steps = (int(getattr(self.config, "one_step_schedule_steps", 52))
                          if self.decode_sampler == "one_step" else steps)
        self.scheduler.set_timesteps(
            schedule_steps, denoising_strength=1.0,
            image_resolution=(height, width))
        # t-independent condition work (z_q norm, in-context adapter) is computed once per
        # decode and reused by its steps (fast path); never across decodes.
        self.dit._decode_cache = {}
        try:
            for step in range(steps):
                timestep = self.scheduler.timesteps[step]
                preds = model_fn_repar_ideogram4_packed(
                    self.dit, patch, latents, ctx_list,
                    [float(timestep)] * batch, z_q_list, z_grid_list,
                    [height] * batch, [width] * batch,
                    context_precomputed=True)
                if self.decode_sampler == "native":
                    latents = [self.scheduler.step(pred, timestep, latent)
                               for pred, latent in zip(preds, latents)]
                else:
                    # Identical clean prediction and re-noise rule to the trainer.
                    sigma = float(self.scheduler.sigmas[step])
                    latents = [latent - sigma * pred for pred, latent in zip(preds, latents)]
                    if self.decode_sampler == "dmad4" and step + 1 < steps:
                        next_sigma = float(self.scheduler.sigmas[step + 1])
                        latents = [(1.0 - next_sigma) * clean + next_sigma * torch.randn(
                            clean.shape, generator=generator, dtype=torch.float32).to(clean)
                            for clean, generator in zip(latents, generators)]
        finally:
            self.dit._decode_cache = None
        images = [vae_decode(self.vae, latent, height, width, self.torch_dtype)
                  for latent in latents]
        return torch.cat(images, dim=0)

    # ---- ViT 解码重建 (tokenizer 自身) ----
    @torch.no_grad()
    def reconstruct_vit(self, x_hq):
        x = x_hq.to(self.device, self.torch_dtype)
        if x.dim() == 3:
            x = x.unsqueeze(0)
        H, W = int(self._round_to_tokenizer_multiple(x.shape[-2])), int(self._round_to_tokenizer_multiple(x.shape[-1]))
        if (H, W) != (x.shape[-2], x.shape[-1]):
            x = F.interpolate(x, size=(H, W), mode="bicubic", align_corners=False, antialias=True)
        x_lq = x
        hp, wp = x_lq.shape[-2] // 16, x_lq.shape[-1] // 16
        grid = torch.tensor([[1, hp, wp]], dtype=torch.long, device=x.device)
        dev_type = self.device.type
        with torch.autocast(dev_type, dtype=self.torch_dtype, enabled=(dev_type == "cuda")):
            out = self.repar_tok.encode([x_lq[i] for i in range(x_lq.shape[0])], grid)
            og = out.out_grid_thw[0].tolist()
            gh, gw = int(og[1]), int(og[2])
            idx = out.indices
            if idx.dim() == 2:
                idx = idx.unsqueeze(0)
            idx4 = idx.reshape(1, gh, gw, idx.shape[-1]).to(x.device)
            rec = self.repar_tok.decode_to_pixels(idx4, gh, gw, H, W)
        if rec is None:
            return None
        return self._to_pil(rec[0] if rec.dim() == 4 else rec)

    # ---- Refiner 精修重建 ----
    @torch.no_grad()
    def _resolve_sr_scale(self, sr_scale):
        scale = self.fixed_sr_scale if sr_scale is None else int(sr_scale)
        if scale != self.fixed_sr_scale:
            raise ValueError(
                f"This HF export is fixed at {self.fixed_sr_scale}x SR; got sr_scale={scale}.")
        return scale

    @torch.no_grad()
    def reconstruct_refiner(self, x_hq, sr_scale=None, num_inference_steps=None, seed=None):
        x = x_hq.to(self.device, self.torch_dtype)
        if x.dim() == 3:
            x = x.unsqueeze(0)
        H, W = int(self._round_to_tokenizer_multiple(x.shape[-2])), int(self._round_to_tokenizer_multiple(x.shape[-1]))
        if (H, W) != (x.shape[-2], x.shape[-1]):
            x = F.interpolate(x, size=(H, W), mode="bicubic", align_corners=False, antialias=True)
        scale = self._resolve_sr_scale(sr_scale)
        # x itself is the deployment condition. Never downsample it again.
        z_q, z_grid = self.encode_zq(x, sr_scale=1)
        image = self._decode_refiner_from_zq(
            z_q, z_grid, H * scale, W * scale, num_inference_steps, seed)
        return self._to_pil(image if torch.is_tensor(image) else image)

    # ---- 统一入口: 选输出模式 ----
    @torch.no_grad()
    def reconstruct(self, image=None, source_tensor=None, mode="refiner_cf",
                    sr_scale=None, num_inference_steps=None, seed=None):
        """mode: 'vit' | 'refiner' | 'refiner_cf'(refiner + wavelet color fix)。
        image=PIL 或 source_tensor=(3,H,W)/(1,3,H,W) in [-1,1]。"""
        if source_tensor is not None:
            x = source_tensor
        elif image is not None:
            x = self._preprocess(image)
        else:
            raise ValueError("Provide image=PIL or source_tensor.")
        if mode == "vit":
            return self.reconstruct_vit(x)
        rec = self.reconstruct_refiner(x, sr_scale=sr_scale,
                                       num_inference_steps=num_inference_steps, seed=seed)
        if mode == "refiner":
            return rec
        if mode == "refiner_cf":
            if wavelet_color_fix is None:
                return rec
            ref = self.reconstruct_vit(x)      # 色源 = 原生分辨率 ViT 重建 (与 z_q 一致)
            if ref is not None:
                rec = wavelet_color_fix(rec, ref.resize(rec.size))
            return rec
        raise ValueError(f"unknown mode {mode!r}; use vit / refiner / refiner_cf")
