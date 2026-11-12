import os

import torch
import torch.utils.checkpoint
from torch import nn

from .common import FlashVarLenAttention, RMSNorm
from .spatial_merger import SpatialUnmerger


class CasualDepthTransformerLayer(nn.Module):
    def __init__(self, config, depth):
        super().__init__()
        self.config = config.visual_config.image_head_config
        self.depth = depth
        self.llm_hidden_size = config.hidden_size
        self.num_heads = self.config.image_head_transformer_dims // 128
        self.transformer_ffn_scale = self.config.image_head_transformer_ffn_scale
        self.transformer_dims = self.config.image_head_transformer_dims

        assert self.transformer_dims % 128 == 0
        assert self.transformer_dims % depth == 0

        self.self_attention = FlashVarLenAttention(self.transformer_dims, self.num_heads, causal=True)
        self.layernorm1 = RMSNorm(self.transformer_dims)
        self.layernorm2 = RMSNorm(self.transformer_dims)
        if self.transformer_ffn_scale <= 0:
            self.linear1 = nn.Linear(self.transformer_dims * self.depth, 2 * self.transformer_dims)
            self.linear2 = nn.Linear(2 * self.transformer_dims * self.depth, self.transformer_dims)
        else:
            self.linear1 = nn.Linear(
                self.transformer_dims,
                self.transformer_ffn_scale * self.transformer_dims
            )
            self.linear2 = nn.Linear(
                self.transformer_ffn_scale * self.transformer_dims,
                self.transformer_dims
            )

    def forward(self, x):
        bsz = x.shape[0]
        res = x
        x = self.layernorm1(x)

        seqlens = self.depth * torch.ones((bsz,), dtype=torch.int32, device=x.device)
        _x = self.self_attention(x.view(-1, self.transformer_dims), seqlens)
        _x = _x.view(bsz, self.depth, self.transformer_dims).contiguous()

        _res = _x + res
        res = self.layernorm2(_res)

        if self.transformer_ffn_scale <= 0:
            x = torch.einsum('bld,tld->blt', res, torch.reshape(self.linear1.weight, (2 * self.llm_hidden_size, -1, self.llm_hidden_size)))
            x = torch.nn.functional.gelu(x)
            x = torch.einsum('blt,dlt->bld', x, torch.reshape(self.linear2.weight, (self.llm_hidden_size, -1, 2 * self.llm_hidden_size)))
        else:
            if self.depth > 1:
                x = torch.einsum('bld,tld->blt',
                    res,
                    torch.reshape(self.linear1.weight,
                        (self.transformer_ffn_scale * self.transformer_dims // self.depth,
                        self.depth,
                        self.transformer_dims)
                    ))
            else:
                x = self.linear1(res)
            x = torch.nn.functional.gelu(x)
            if self.depth > 1:
                x = torch.einsum('blt,dlt->bld',
                    x,
                    torch.reshape(self.linear2.weight,
                        (self.transformer_dims,
                        self.depth,
                        self.transformer_ffn_scale * self.transformer_dims // self.depth)
                    ))
            else:
                x = self.linear2(x)
        return _res + x


class BlockARDepthTransformerLayer(nn.Module):
    """Block-wise AR depth transformer 层 (block_ar 模式, 不依赖 flash).

    序列长度 seq_len = ffn_depth * block_size (= depth * mms_unit). 注意力是 block-causal:
    同一 depth 组内的 block_size 个空间位置 *双向* attend, depth 组之间 causal——
    位置 i (=d*block_size+s) attend 到 j 当且仅当 group(j) <= group(i), group = idx//block_size.
    FFN 按 RVQ-level (ffn_depth) 做 8 路分解, block_size 个空间位置在同 level 内共享权重.
    对称由 OmniImageHead 的可学习 pos_embed 打破, 这里不兼职.
    """

    def __init__(self, config, ffn_depth, block_size):
        super().__init__()
        self.config = config.visual_config.image_head_config
        self.ffn_depth = ffn_depth
        self.block_size = block_size
        self.seq_len = ffn_depth * block_size
        self.transformer_ffn_scale = self.config.image_head_transformer_ffn_scale
        self.transformer_dims = self.config.image_head_transformer_dims
        self.head_dim = 128
        self.num_heads = self.transformer_dims // self.head_dim

        assert self.transformer_dims % self.head_dim == 0
        assert self.transformer_ffn_scale > 0, "block_ar 模式要求 image_head_transformer_ffn_scale > 0"
        assert (self.transformer_ffn_scale * self.transformer_dims) % self.ffn_depth == 0

        self.q_proj = nn.Linear(self.transformer_dims, self.transformer_dims, bias=True)
        self.k_proj = nn.Linear(self.transformer_dims, self.transformer_dims, bias=False)
        self.v_proj = nn.Linear(self.transformer_dims, self.transformer_dims, bias=True)
        self.out_proj = nn.Linear(self.transformer_dims, self.transformer_dims, bias=True)
        self.layernorm1 = RMSNorm(self.transformer_dims)
        self.layernorm2 = RMSNorm(self.transformer_dims)
        self.linear1 = nn.Linear(self.transformer_dims, self.transformer_ffn_scale * self.transformer_dims)
        self.linear2 = nn.Linear(self.transformer_ffn_scale * self.transformer_dims, self.transformer_dims)

        groups = torch.arange(self.seq_len) // self.block_size
        # mask[i, j] = True 表示 query i 参与 key j 的注意力; 块内双向 + 跨块 causal.
        attn_mask = groups.unsqueeze(1) >= groups.unsqueeze(0)
        self.register_buffer("attn_mask", attn_mask, persistent=False)

    def forward(self, x):
        # x: [N, seq_len, tdims]
        n = x.shape[0]
        res = x
        x = self.layernorm1(x)

        q = self.q_proj(x).view(n, self.seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(n, self.seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(n, self.seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        attn = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=self.attn_mask)
        attn = attn.transpose(1, 2).reshape(n, self.seq_len, self.transformer_dims)
        attn = self.out_proj(attn)

        _res = attn + res
        res = self.layernorm2(_res)

        # 8 路 per-RVQ-level FFN: 把 [N, seq_len, dims] 视作 [N, ffn_depth, block_size, dims],
        # level d 用 w1[:, d, :] / w2[:, d, :], block_size 个空间位置共享.
        t8 = self.transformer_ffn_scale * self.transformer_dims // self.ffn_depth
        res4 = res.view(n, self.ffn_depth, self.block_size, self.transformer_dims)
        w1 = self.linear1.weight.view(t8, self.ffn_depth, self.transformer_dims)
        h = torch.einsum('ndsh,tdh->ndst', res4, w1)
        h = torch.nn.functional.gelu(h)
        w2 = self.linear2.weight.view(self.transformer_dims, self.ffn_depth, t8)
        o = torch.einsum('ndst,hdt->ndsh', h, w2)
        o = o.reshape(n, self.seq_len, self.transformer_dims)
        return _res + o


class SpatialCausalDepthTransformerLayer(nn.Module):
    """纯 causal depth transformer 层 (spatial_causal 模式).

    序列长度 seq_len = block_size * ffn_depth (= mms_unit * depth), spatial-major 排列:
    [s0l0, s0l1, .., s0l7, s1l0, ..], pos = s*depth + l. 注意力是完整下三角 causal——
    位置 i attend 到所有 j <= i, 因此 4 个空间位置被强加一个 AR 顺序 (与 block_ar 的
    "空间双向无序" 本质不同). FFN 仍按 RVQ-level (ffn_depth) 做 8 路分解, block_size
    个空间位置在同 level 内共享权重 (按 l = pos % ffn_depth 选权重).
    teacher-forcing 的加性输入由 OmniImageHead._run_spatial_causal 构造.
    """

    def __init__(self, config, ffn_depth, block_size):
        super().__init__()
        self.config = config.visual_config.image_head_config
        self.ffn_depth = ffn_depth
        self.block_size = block_size
        self.seq_len = block_size * ffn_depth
        self.transformer_ffn_scale = self.config.image_head_transformer_ffn_scale
        self.transformer_dims = self.config.image_head_transformer_dims
        self.head_dim = 128
        self.num_heads = self.transformer_dims // self.head_dim

        assert self.transformer_dims % self.head_dim == 0
        assert self.transformer_ffn_scale > 0, "spatial_causal 模式要求 image_head_transformer_ffn_scale > 0"
        assert (self.transformer_ffn_scale * self.transformer_dims) % self.ffn_depth == 0

        self.q_proj = nn.Linear(self.transformer_dims, self.transformer_dims, bias=True)
        self.k_proj = nn.Linear(self.transformer_dims, self.transformer_dims, bias=False)
        self.v_proj = nn.Linear(self.transformer_dims, self.transformer_dims, bias=True)
        self.out_proj = nn.Linear(self.transformer_dims, self.transformer_dims, bias=True)
        self.layernorm1 = RMSNorm(self.transformer_dims)
        self.layernorm2 = RMSNorm(self.transformer_dims)
        self.linear1 = nn.Linear(self.transformer_dims, self.transformer_ffn_scale * self.transformer_dims)
        self.linear2 = nn.Linear(self.transformer_ffn_scale * self.transformer_dims, self.transformer_dims)

    def forward(self, x):
        # x: [N, seq_len, tdims], seq_len = block_size * ffn_depth (spatial-major)
        n = x.shape[0]
        res = x
        x = self.layernorm1(x)

        q = self.q_proj(x).view(n, self.seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(n, self.seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(n, self.seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        attn = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
        attn = attn.transpose(1, 2).reshape(n, self.seq_len, self.transformer_dims)
        attn = self.out_proj(attn)

        _res = attn + res
        res = self.layernorm2(_res)

        # 8 路 per-RVQ-level FFN: [N, seq_len, dims] 视作 [N, block_size, ffn_depth, dims]
        # (spatial-major, 内层是 level), level l 用 w1[:, l, :] / w2[:, l, :], block_size 个空间共享.
        t8 = self.transformer_ffn_scale * self.transformer_dims // self.ffn_depth
        res4 = res.view(n, self.block_size, self.ffn_depth, self.transformer_dims)
        w1 = self.linear1.weight.view(t8, self.ffn_depth, self.transformer_dims)
        h = torch.einsum('nslh,tlh->nslt', res4, w1)
        h = torch.nn.functional.gelu(h)
        w2 = self.linear2.weight.view(self.transformer_dims, self.ffn_depth, t8)
        o = torch.einsum('nslt,hlt->nslh', h, w2)
        o = o.reshape(n, self.seq_len, self.transformer_dims)
        return _res + o


def _ada_modulate(x, shift, scale):
    """DiT AdaLN 调制. x: [n, depth, dims]; shift/scale: [n, dims] (逐行, 沿 depth 广播)."""
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class MaskDepthTransformerLayer(nn.Module):
    """全双向 + AdaLN 调制的 depth transformer 层 (mask / MaskGit 模式).

    序列长度 seq_len = depth (每个 depth-level 一个位置). 注意力是 *完整双向*——
    位置 i attend 到所有 j (无 causal mask), 因为 MaskGit 是 absorbing-state 扩散,
    未遮位置提供双向上下文帮助补全被遮位置, 不存在 AR 顺序. FFN 按 depth (每 level)
    做逐路分解. level 间对称由 OmniImageHead 的可学习 pos_embed 打破.

    条件 (LLM hidden x) 不再加进 token slot, 而是像 DiT 一样通过 AdaLN 调制注入:
    每层用条件 c 出 6 组 (shift/scale/gate) x (attn, ffn), modulate(norm(x)) + gate*sub(x).
    """

    def __init__(self, config, depth, stable_attention=False):
        super().__init__()
        self.config = config.visual_config.image_head_config
        self.depth = depth
        self.seq_len = depth
        self.transformer_ffn_scale = self.config.image_head_transformer_ffn_scale
        self.transformer_dims = self.config.image_head_transformer_dims
        self.head_dim = 128
        self.num_heads = self.transformer_dims // self.head_dim
        self.stable_attention = bool(stable_attention)

        assert self.transformer_dims % self.head_dim == 0
        assert self.transformer_ffn_scale > 0, "mask 模式要求 image_head_transformer_ffn_scale > 0"
        assert (self.transformer_ffn_scale * self.transformer_dims) % self.depth == 0

        self.q_proj = nn.Linear(self.transformer_dims, self.transformer_dims, bias=True)
        self.k_proj = nn.Linear(self.transformer_dims, self.transformer_dims, bias=False)
        self.v_proj = nn.Linear(self.transformer_dims, self.transformer_dims, bias=True)
        self.out_proj = nn.Linear(self.transformer_dims, self.transformer_dims, bias=True)
        self.layernorm1 = RMSNorm(self.transformer_dims)
        self.layernorm2 = RMSNorm(self.transformer_dims)
        self.linear1 = nn.Linear(self.transformer_dims, self.transformer_ffn_scale * self.transformer_dims)
        self.linear2 = nn.Linear(self.transformer_ffn_scale * self.transformer_dims, self.transformer_dims)
        # AdaLN: 条件 c -> (shift,scale,gate) x (attn, ffn) = 6 组.
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(self.transformer_dims, 6 * self.transformer_dims),
        )

    def forward(self, x, c):
        # x: [N, depth, tdims]; c: [N, tdims] (条件 = 投影后的 LLM hidden)
        n = x.shape[0]
        shift_a, scale_a, gate_a, shift_m, scale_m, gate_m = self.adaLN_modulation(c).chunk(6, dim=-1)

        h = _ada_modulate(self.layernorm1(x), shift_a, scale_a)
        q = self.q_proj(h).view(n, self.seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(h).view(n, self.seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(h).view(n, self.seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        # 全双向: 无 attn_mask / 无 is_causal. DSpark 的 all-MASK
        # backbone 在 H100 BF16 + 大 flattened batch 下需要 FP32 attention
        # math；其他 MaskGiT 路径保持原有 SDPA 吞吐。
        if self.stable_attention and q.dtype in (torch.float16, torch.bfloat16):
            with torch.autocast(device_type=q.device.type, enabled=False):
                scores = torch.matmul(q.float(), k.float().transpose(-2, -1))
                scores = scores * (self.head_dim ** -0.5)
                attn = torch.matmul(torch.softmax(scores, dim=-1), v.float()).to(q.dtype)
        else:
            attn = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        attn = attn.transpose(1, 2).reshape(n, self.seq_len, self.transformer_dims)
        attn = self.out_proj(attn)
        x = x + gate_a.unsqueeze(1) * attn

        # 逐 level FFN: [N, depth, dims], level l 用 w1[:, l, :] / w2[:, l, :].
        h = _ada_modulate(self.layernorm2(x), shift_m, scale_m)
        t8 = self.transformer_ffn_scale * self.transformer_dims // self.depth
        w1 = self.linear1.weight.view(t8, self.depth, self.transformer_dims)
        ff = torch.einsum('nlh,tlh->nlt', h, w1)
        ff = torch.nn.functional.gelu(ff)
        w2 = self.linear2.weight.view(self.transformer_dims, self.depth, t8)
        ff = torch.einsum('nlt,hlt->nlh', ff, w2)
        return x + gate_m.unsqueeze(1) * ff


class _GroupedCausalTransformerLayer(nn.Module):
    """Small causal layer used by the explicit grouped AR/DSpark heads.

    The sequence axis is the configured group axis (for f64g32 it is the
    four raster-ordered 2x2 patch offsets, each followed by its eight PQ
    groups).  ``forward_step`` is numerically the same layer with a tiny KV
    cache, so teacher-forced training is parallel while generation is truly
    autoregressive without re-running the prefix.
    """

    def __init__(self, dims, num_heads, ffn_scale=4, stable_attention=False):
        super().__init__()
        assert dims % num_heads == 0
        self.dims = int(dims)
        self.num_heads = int(num_heads)
        self.head_dim = self.dims // self.num_heads
        self.stable_attention = bool(stable_attention)
        self.q_proj = nn.Linear(self.dims, self.dims, bias=True)
        self.k_proj = nn.Linear(self.dims, self.dims, bias=False)
        self.v_proj = nn.Linear(self.dims, self.dims, bias=True)
        self.out_proj = nn.Linear(self.dims, self.dims, bias=True)
        self.norm1 = RMSNorm(self.dims)
        self.norm2 = RMSNorm(self.dims)
        inner = int(ffn_scale) * self.dims
        self.fc1 = nn.Linear(self.dims, inner)
        self.fc2 = nn.Linear(inner, self.dims)

    def _qkv(self, x):
        bsz, seqlen = x.shape[:2]
        shape = (bsz, seqlen, self.num_heads, self.head_dim)
        q = self.q_proj(x).view(shape).transpose(1, 2)
        k = self.k_proj(x).view(shape).transpose(1, 2)
        v = self.v_proj(x).view(shape).transpose(1, 2)
        return q, k, v

    def _ffn(self, x):
        return self.fc2(torch.nn.functional.gelu(self.fc1(x)))

    def forward(self, x):
        residual = x
        q, k, v = self._qkv(self.norm1(x))
        if self.stable_attention and q.dtype in (torch.float16, torch.bfloat16):
            # DSpark's group axis is only 8, but its flattened image-token batch
            # is large. H100 BF16 attention backward is non-finite for this
            # shape. Explicitly disable the outer autocast for the tiny matrix.
            with torch.autocast(device_type=q.device.type, enabled=False):
                scores = torch.matmul(q.float(), k.float().transpose(-2, -1))
                scores = scores * (self.head_dim ** -0.5)
                causal_mask = torch.ones(
                    x.shape[1], x.shape[1], dtype=torch.bool, device=x.device,
                ).triu_(1)
                scores.masked_fill_(causal_mask, float("-inf"))
                attn = torch.matmul(torch.softmax(scores, dim=-1), v.float()).to(q.dtype)
        else:
            attn = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
        attn = attn.transpose(1, 2).reshape(x.shape[0], x.shape[1], self.dims)
        x = residual + self.out_proj(attn)
        return x + self._ffn(self.norm2(x))

    def forward_step(self, x, cache=None):
        """x: [B,1,D]; cache: (K,V) with shape [B,H,T,head_dim]."""
        residual = x
        q, k_new, v_new = self._qkv(self.norm1(x))
        if cache is None:
            k, v = k_new, v_new
        else:
            k = torch.cat([cache[0], k_new], dim=2)
            v = torch.cat([cache[1], v_new], dim=2)
        # The sole query is the newest position, so every cached key is legal.
        if self.stable_attention and q.dtype in (torch.float16, torch.bfloat16):
            with torch.autocast(device_type=q.device.type, enabled=False):
                scores = torch.matmul(q.float(), k.float().transpose(-2, -1))
                scores = scores * (self.head_dim ** -0.5)
                attn = torch.matmul(torch.softmax(scores, dim=-1), v.float()).to(q.dtype)
        else:
            attn = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        attn = attn.transpose(1, 2).reshape(x.shape[0], 1, self.dims)
        x = residual + self.out_proj(attn)
        x = x + self._ffn(self.norm2(x))
        return x, (k, v)


class _ExplicitGroupedHeadBase(nn.Module):
    """Common f{compression}g{groups} layout and sampling utilities.

    The tokenizer still produces ``M`` spatial positions with ``D`` codebook
    groups.  ``merger_merge_size`` folds M=mms^2 positions into one LLM step;
    the head sees the reversible spatial-major sequence ``G=M*D``.
    """

    def __init__(self, config, causal_layers, stable_attention=False):
        super().__init__()
        self.root_config = config
        self.config = config.visual_config.image_head_config
        self.llm_hidden_size = int(config.hidden_size)
        self.transformer_dims = int(self.config.image_head_transformer_dims)
        self.codebook_sizes = list(config.visual_quantizer_config.codebook_sizes)
        self.depth = len(self.codebook_sizes)
        self.mms = int(getattr(config.visual_config, "merger_merge_size", 1) or 1)
        self.mms_unit = self.mms * self.mms
        self.group_count = self.mms_unit * self.depth
        configured_patch = int(getattr(self.config, "image_head_spatial_group_size", self.mms))
        configured_groups = int(getattr(self.config, "image_head_group_count", self.group_count))
        assert configured_patch == self.mms, (
            f"image_head_spatial_group_size={configured_patch} must equal "
            f"merger_merge_size={self.mms}"
        )
        assert configured_groups == self.group_count, (
            f"image_head_group_count={configured_groups}, expected "
            f"mms^2*depth={self.mms_unit}*{self.depth}={self.group_count}"
        )
        base_compression = int(getattr(self.config, "image_head_base_compression", 0) or 0)
        compression = int(getattr(self.config, "image_head_compression", 0) or 0)
        if base_compression and compression:
            assert compression == base_compression * self.mms, (
                f"compression={compression}, expected base_compression({base_compression}) "
                f"* spatial_group_size({self.mms})"
            )

        rank = int(getattr(self.config, "image_head_token_embed_dims", 256))
        self.token_embed_dims = rank
        self.token_emb_layers = nn.ModuleList([
            nn.Embedding(vq_size + 1, rank) for vq_size in self.codebook_sizes
        ])
        self.bos_embed = nn.Embedding(1, rank)
        self.cond_proj = nn.Linear(self.llm_hidden_size, self.transformer_dims, bias=True)
        self.input_proj = nn.Linear(self.transformer_dims + rank, self.transformer_dims, bias=True)
        self.pos_embed = nn.Embedding(self.group_count, self.transformer_dims)
        head_dim = int(getattr(self.config, "image_head_causal_head_dim", 128))
        assert self.transformer_dims % head_dim == 0
        num_heads = self.transformer_dims // head_dim
        ffn_scale = int(getattr(self.config, "image_head_causal_ffn_scale", 4))
        self.causal_layers = nn.ModuleList([
            _GroupedCausalTransformerLayer(
                self.transformer_dims, num_heads, ffn_scale,
                stable_attention=stable_attention,
            )
            for _ in range(int(causal_layers))
        ])
        self.headnorm = RMSNorm(self.transformer_dims)
        self.heads = nn.ModuleList([
            nn.Linear(self.transformer_dims, vq_size + 1) for vq_size in self.codebook_sizes
        ])
        self.allow_eos = bool(getattr(self.config, "image_head_allow_eos", False))
        # Chunk only the outer row/LLM-position dimension.  Each row retains
        # all mms^2*depth groups, so f64g32 causal semantics are unchanged.
        self.image_head_num_chunks = int(getattr(
            self.config, "image_head_num_chunks", 1
        ))
        self.gradient_checkpointing = True

    def _flatten_tokens(self, visual_tokens, n):
        assert visual_tokens.shape == (n * self.mms_unit, self.depth), (
            f"visual_tokens={tuple(visual_tokens.shape)}, expected "
            f"({n}*{self.mms_unit}, {self.depth})"
        )
        return visual_tokens.view(n, self.mms_unit, self.depth).reshape(n, self.group_count)

    def _shifted_embeddings(self, flat_tokens):
        n = flat_tokens.shape[0]
        shifted = [self.bos_embed.weight[0].view(1, -1).expand(n, -1)]
        for slot in range(1, self.group_count):
            prev_level = (slot - 1) % self.depth
            shifted.append(self.token_emb_layers[prev_level](flat_tokens[:, slot - 1]))
        return torch.stack(shifted, dim=1)

    def _pack_legacy_logits(self, hidden_states):
        n = hidden_states.shape[0]
        result = []
        for level, head in enumerate(self.heads):
            slots = hidden_states[:, level::self.depth, :]
            result.append(head(slots).reshape(n * self.mms_unit, -1))
        return result

    def _causal_forward(self, base_hidden, flat_tokens, use_checkpoint=None):
        if use_checkpoint is None:
            use_checkpoint = self.gradient_checkpointing and self.training
        shifted = self._shifted_embeddings(flat_tokens)
        hidden_states = self.input_proj(torch.cat([base_hidden, shifted], dim=-1))
        for layer in self.causal_layers:
            if use_checkpoint:
                hidden_states = torch.utils.checkpoint.checkpoint(
                    layer, hidden_states, use_reentrant=False,
                )
            else:
                hidden_states = layer(hidden_states)
        return self.headnorm(hidden_states)

    def _forward_logits_tuple(self, x, flat_tokens, use_inner_checkpoint):
        base_hidden = self._base_hidden(
            x, use_checkpoint=use_inner_checkpoint
        )
        hidden = self._causal_forward(
            base_hidden, flat_tokens,
            use_checkpoint=use_inner_checkpoint,
        )
        return tuple(self._pack_legacy_logits(hidden))

    def _forward_grouped(self, x, visual_tokens):
        flat = self._flatten_tokens(visual_tokens, x.shape[0])
        chunked = (
            self.training
            and self.image_head_num_chunks > 1
            and x.shape[0] >= self.image_head_num_chunks
        )
        if not chunked:
            return list(self._forward_logits_tuple(
                x, flat,
                self.gradient_checkpointing and self.training,
            ))

        x_chunks = torch.chunk(x, self.image_head_num_chunks, dim=0)
        token_chunks = torch.chunk(flat, self.image_head_num_chunks, dim=0)
        per_chunk_logits = []
        for chunk_x, chunk_tokens in zip(x_chunks, token_chunks):
            if self.gradient_checkpointing:
                # One outer checkpoint keeps only the chunk input.  Disable
                # nested per-layer checkpoints during recomputation.
                def forward_chunk(local_x, local_tokens):
                    return self._forward_logits_tuple(
                        local_x, local_tokens, False
                    )
                chunk_logits = torch.utils.checkpoint.checkpoint(
                    forward_chunk,
                    chunk_x,
                    chunk_tokens,
                    use_reentrant=False,
                )
            else:
                chunk_logits = self._forward_logits_tuple(
                    chunk_x, chunk_tokens, False
                )
            per_chunk_logits.append(chunk_logits)

        return [
            torch.cat([chunk[level] for chunk in per_chunk_logits], dim=0)
            for level in range(self.depth)
        ]

    def _sample_scores(self, logits, level, *, cfg_scale, do_sample,
                       logits_processor, vision_sequences, image_cfg_scale=None):
        rows = logits.shape[0]
        branches = 3 if image_cfg_scale is not None else (2 if cfg_scale != 1.0 else 1)
        if rows % branches:
            raise ValueError("CFG rows must be divisible by the number of branches")
        half = rows // branches
        scores = logits[:, :self.codebook_sizes[level]].float()
        if image_cfg_scale is not None:
            full, image, empty = scores.chunk(3, dim=0)
            scores = image + cfg_scale * (full - image) + (image_cfg_scale - 1.0) * (image - empty)
        elif branches == 2:
            cond, uncond = scores[:half], scores[half:]
            scores = uncond + cfg_scale * (cond - uncond)
        if logits_processor is not None:
            if vision_sequences is None or vision_sequences.numel() == 0:
                history = torch.zeros(half, 0, dtype=torch.long, device=scores.device)
            else:
                history = vision_sequences[:half, :, level]
            scores = logits_processor(history, scores)
        if do_sample:
            probs = torch.softmax(scores, dim=-1)
            sampled = torch.multinomial(probs, num_samples=1).squeeze(1)
        else:
            sampled = scores.argmax(dim=-1)
        if branches > 1:
            sampled = sampled.repeat((branches,))
        return sampled

    def _generate_causal(self, x, base_hidden, *, cfg_scale=1.0,
                         do_sample=True, logits_processor=None,
                         vision_sequences=None, image_cfg_scale=None, **_kwargs):
        rows = x.shape[0]
        tokens = torch.zeros(rows, self.group_count, dtype=torch.long, device=x.device)
        caches = [None] * len(self.causal_layers)
        prev_emb = self.bos_embed.weight[0].view(1, -1).expand(rows, -1)
        for slot in range(self.group_count):
            step = self.input_proj(torch.cat([base_hidden[:, slot, :], prev_emb], dim=-1)).unsqueeze(1)
            for layer_idx, layer in enumerate(self.causal_layers):
                step, caches[layer_idx] = layer.forward_step(step, caches[layer_idx])
            step = self.headnorm(step[:, 0, :])
            level = slot % self.depth
            logits = self.heads[level](step)
            sampled = self._sample_scores(
                logits, level, cfg_scale=cfg_scale, do_sample=do_sample,
                logits_processor=logits_processor, vision_sequences=vision_sequences,
                image_cfg_scale=image_cfg_scale,
            )
            tokens[:, slot] = sampled
            prev_emb = self.token_emb_layers[level](sampled)
        return tokens.view(rows, self.mms_unit, self.depth).reshape(
            rows * self.mms_unit, self.depth
        )


class ARHead(_ExplicitGroupedHeadBase):
    """DSpark-style lightweight group AR directly conditioned on LLM hidden."""

    def __init__(self, config):
        layers = int(getattr(
            config.visual_config.image_head_config,
            "image_head_ar_layers",
            config.visual_config.image_head_config.image_head_transformer_layers,
        ))
        super().__init__(config, causal_layers=layers)

    def _base_hidden(self, x, use_checkpoint=None):
        base = self.cond_proj(x).unsqueeze(1).expand(-1, self.group_count, -1)
        return base + self.pos_embed.weight.unsqueeze(0)

    def forward(self, x, visual_tokens):
        return self._forward_grouped(x, visual_tokens)

    @torch.no_grad()
    def sample(self, x, **kwargs):
        return self._generate_causal(x, self._base_hidden(x), **kwargs)


class DSparkAttnHead(_ExplicitGroupedHeadBase):
    """All-MASK parallel backbone followed by a cached causal correction head.

    The all-MASK rule is intentional: allowing ground-truth unmasked groups into
    the bidirectional backbone would leak future targets around the causal layer.
    """

    def __init__(self, config):
        ihc = config.visual_config.image_head_config
        causal_layers = int(getattr(ihc, "image_head_dspark_attn_layers", 1))
        super().__init__(config, causal_layers=causal_layers, stable_attention=True)
        mask_layers = int(getattr(
            ihc, "image_head_mask_backbone_layers",
            max(int(ihc.image_head_transformer_layers) - causal_layers, 1),
        ))
        self.mask_embed = nn.Embedding(1, self.llm_hidden_size)
        self.mask_input_proj = nn.Linear(self.llm_hidden_size, self.transformer_dims, bias=True)
        self.mask_input_norm = RMSNorm(self.transformer_dims)
        self.mask_cond_proj = nn.Linear(self.llm_hidden_size, self.transformer_dims, bias=True)
        self.mask_layers = nn.ModuleList([
            MaskDepthTransformerLayer(
                config, self.group_count, stable_attention=True,
            ) for _ in range(mask_layers)
        ])

    def _base_hidden(self, x, use_checkpoint=None):
        if use_checkpoint is None:
            use_checkpoint = self.gradient_checkpointing and self.training
        rows = x.shape[0]
        mask = self.mask_embed.weight[0].view(1, 1, -1).expand(
            rows, self.group_count, -1
        )
        hidden = self.mask_input_proj(mask) + self.pos_embed.weight.unsqueeze(0)
        hidden = self.mask_input_norm(hidden)
        cond = self.mask_cond_proj(x)
        for layer in self.mask_layers:
            if use_checkpoint:
                hidden = torch.utils.checkpoint.checkpoint(
                    layer, hidden, cond, use_reentrant=False,
                )
            else:
                hidden = layer(hidden, cond)
        return hidden

    def forward(self, x, visual_tokens):
        return self._forward_grouped(x, visual_tokens)

    @torch.no_grad()
    def sample(self, x, **kwargs):
        return self._generate_causal(x, self._base_hidden(x), **kwargs)


class MaskGiTHead(nn.Module):
    """Explicit grouped MaskGiT head supporting g8 and f64g32 alike."""

    def __init__(self, config):
        super().__init__()
        self.root_config = config
        self.config = config.visual_config.image_head_config
        self.llm_hidden_size = int(config.hidden_size)
        self.transformer_dims = int(self.config.image_head_transformer_dims)
        self.codebook_sizes = list(config.visual_quantizer_config.codebook_sizes)
        self.depth = len(self.codebook_sizes)
        self.mms = int(getattr(config.visual_config, "merger_merge_size", 1) or 1)
        self.mms_unit = self.mms * self.mms
        self.group_count = self.mms_unit * self.depth
        configured_patch = int(getattr(self.config, "image_head_spatial_group_size", self.mms))
        configured_groups = int(getattr(self.config, "image_head_group_count", self.group_count))
        assert configured_patch == self.mms
        assert configured_groups == self.group_count
        base_compression = int(getattr(self.config, "image_head_base_compression", 0) or 0)
        compression = int(getattr(self.config, "image_head_compression", 0) or 0)
        if base_compression and compression:
            assert compression == base_compression * self.mms

        self.mask_ratio_dist = str(getattr(self.config, "image_head_mask_ratio_dist", "uniform_count"))
        self.mask_logitnorm_std = float(getattr(self.config, "image_head_mask_logitnorm_std", 0.8))
        self.mask_temperature = float(getattr(self.config, "image_head_mask_temperature", 1.0))
        default_schedule = [self.group_count]
        self.mask_schedule = list(getattr(self.config, "image_head_mask_schedule", default_schedule))
        assert all(v > 0 for v in self.mask_schedule)
        assert sum(self.mask_schedule) == self.group_count
        assert all(self.mask_schedule[i] <= self.mask_schedule[i + 1]
                   for i in range(len(self.mask_schedule) - 1))
        self.mask_refine_eta = float(getattr(self.config, "image_head_mask_refine_eta", 0.0))
        self.mask_refine_perturb_k = int(getattr(self.config, "image_head_mask_refine_perturb_k", 8))
        self.mask_refine_loss_weight = float(getattr(self.config, "image_head_mask_refine_loss_weight", 1.0))
        gamma = getattr(self.config, "image_head_mask_refine_gamma", None)
        self.mask_refine_gamma = None if gamma is None else float(gamma)

        self.mask_embed = nn.Embedding(1, self.llm_hidden_size)
        self.token_emb_layers = nn.ModuleList([
            nn.Embedding(vq_size + 1, self.llm_hidden_size)
            for vq_size in self.codebook_sizes
        ])
        self.input_proj = nn.Linear(self.llm_hidden_size, self.transformer_dims, bias=True)
        self.input_norm = RMSNorm(self.transformer_dims)
        self.cond_proj = nn.Linear(self.llm_hidden_size, self.transformer_dims, bias=True)
        self.pos_embed = nn.Embedding(self.group_count, self.transformer_dims)
        self.layers = nn.ModuleList([
            MaskDepthTransformerLayer(config, self.group_count)
            for _ in range(int(self.config.image_head_transformer_layers))
        ])
        self.final_norm = RMSNorm(self.transformer_dims)
        self.final_adaLN = nn.Sequential(
            nn.SiLU(), nn.Linear(self.transformer_dims, 2 * self.transformer_dims)
        )
        self.heads = nn.ModuleList([
            nn.Linear(self.transformer_dims, vq_size + 1) for vq_size in self.codebook_sizes
        ])
        self.gradient_checkpointing = True
        self._last_train_mask = None
        self._last_perturb_mask = None
        self._neighbor_table = None
        self._warned_no_neighbor = False

    def set_neighbor_table(self, tables):
        self._neighbor_table = None if tables is None else [t.to(torch.long) for t in tables]

    def _flatten_tokens(self, visual_tokens, n):
        assert visual_tokens.shape == (n * self.mms_unit, self.depth)
        return visual_tokens.view(n, self.mms_unit, self.depth).reshape(n, self.group_count)

    def _legacy_mask(self, masks):
        return masks.view(masks.shape[0], self.mms_unit, self.depth).reshape(
            masks.shape[0] * self.mms_unit, self.depth
        )

    def _sample_masks(self, flat_tokens):
        rows = flat_tokens.shape[0]
        if os.environ.get("OMNI_MASK_FORCE_ALL_MASKED", "0") == "1":
            num_masked = torch.full(
                (rows,), self.group_count, device=flat_tokens.device, dtype=torch.long
            )
        elif self.mask_ratio_dist == "logit_normal":
            z = torch.randn(rows, device=flat_tokens.device) * self.mask_logitnorm_std
            ratio = (1.0 - torch.sigmoid(z)).clamp(
                min=1.0 / self.group_count, max=1.0
            )
            num_masked = (ratio * self.group_count).round().long().clamp(
                min=1, max=self.group_count
            )
        else:
            num_masked = torch.randint(
                1, self.group_count + 1, (rows,), device=flat_tokens.device
            )
        ranks = torch.rand(rows, self.group_count, device=flat_tokens.device).argsort(1).argsort(1)
        masks = ranks < num_masked.unsqueeze(1)
        eos = flat_tokens[:, 0] == self.codebook_sizes[0]
        return masks & (~eos).unsqueeze(1)

    def _apply_refine_perturb(self, flat_tokens, masks):
        if not (self.training and self.mask_refine_eta > 0.0):
            self._last_perturb_mask = None
            return flat_tokens
        if self._neighbor_table is None:
            if not self._warned_no_neighbor:
                print("[image_head][grouped-maskgit] eta>0 but tokenizer has no neighbor table; disabled")
                self._warned_no_neighbor = True
            self._last_perturb_mask = None
            return flat_tokens
        eos = flat_tokens[:, 0] == self.codebook_sizes[0]
        eligible = (~masks) & (~eos).unsqueeze(1)
        perturb = eligible & (
            torch.rand_like(flat_tokens, dtype=torch.float32) < self.mask_refine_eta
        )
        result = flat_tokens.clone()
        for slot in range(self.group_count):
            pos = perturb[:, slot]
            if not bool(pos.any()):
                continue
            level = slot % self.depth
            table = self._neighbor_table[level]
            if table.device != flat_tokens.device:
                table = table.to(flat_tokens.device)
                self._neighbor_table[level] = table
            ids = flat_tokens[pos, slot]
            pick = torch.randint(0, table.shape[1], (ids.shape[0],), device=ids.device)
            result[pos, slot] = table[ids, pick]
        self._last_perturb_mask = self._legacy_mask(perturb)
        return result

    def _embed_slots(self, flat_tokens, masks):
        rows = flat_tokens.shape[0]
        mask_vec = self.mask_embed.weight[0].view(1, -1).expand(rows, -1)
        slots = []
        for slot in range(self.group_count):
            level = slot % self.depth
            token_emb = self.token_emb_layers[level](flat_tokens[:, slot])
            slots.append(torch.where(masks[:, slot].unsqueeze(1), mask_vec, token_emb))
        return torch.stack(slots, dim=1)

    def _transform(self, token_slots, cond):
        hidden = self.input_proj(token_slots) + self.pos_embed.weight.unsqueeze(0)
        hidden = self.input_norm(hidden)
        c = self.cond_proj(cond)
        for layer in self.layers:
            if self.gradient_checkpointing and self.training:
                hidden = torch.utils.checkpoint.checkpoint(
                    layer, hidden, c, use_reentrant=False,
                )
            else:
                hidden = layer(hidden, c)
        shift, scale = self.final_adaLN(c).chunk(2, dim=-1)
        return _ada_modulate(self.final_norm(hidden), shift, scale)

    def _pack_legacy_logits(self, hidden):
        rows = hidden.shape[0]
        return [
            self.heads[level](hidden[:, level::self.depth, :]).reshape(
                rows * self.mms_unit, -1
            )
            for level in range(self.depth)
        ]

    def forward(self, x, visual_tokens):
        flat = self._flatten_tokens(visual_tokens, x.shape[0])
        masks = self._sample_masks(flat)
        self._last_train_mask = self._legacy_mask(masks)
        embed_tokens = self._apply_refine_perturb(flat, masks)
        hidden = self._transform(self._embed_slots(embed_tokens, masks), x)
        return self._pack_legacy_logits(hidden)

    @torch.no_grad()
    def sample(self, x, *, cfg_scale=1.0, do_sample=True, temperature=None,
               logits_processor=None, vision_sequences=None, refine_gamma=None,
               **_kwargs):
        rows = x.shape[0]
        use_cfg = cfg_scale != 1.0
        half = rows // 2 if use_cfg else rows
        temperature = self.mask_temperature if temperature is None else float(temperature)
        gamma = self.mask_refine_gamma if refine_gamma is None else float(refine_gamma)
        schedule = [self.group_count] if os.environ.get("OMNI_MASK_ONE_STEP", "0") == "1" \
            else self.mask_schedule
        tokens = torch.zeros(rows, self.group_count, dtype=torch.long, device=x.device)
        decided = torch.zeros(rows, self.group_count, dtype=torch.bool, device=x.device)
        for step_idx, n_new in enumerate(schedule):
            hidden = self._transform(self._embed_slots(tokens, ~decided), x)
            sampled_all = torch.zeros(half, self.group_count, dtype=torch.long, device=x.device)
            confidence = torch.full(
                (half, self.group_count), float("-inf"), device=x.device
            )
            clean_logprob = torch.full_like(confidence, float("-inf"))
            annealed_temp = temperature * (1.0 - step_idx / max(len(schedule), 1))
            for slot in range(self.group_count):
                level = slot % self.depth
                logits = self.heads[level](hidden[:, slot, :])[:, :self.codebook_sizes[level]].float()
                if use_cfg:
                    cond, uncond = logits[:half], logits[half:]
                    logits = uncond + cfg_scale * (cond - uncond)
                if logits_processor is not None:
                    if vision_sequences is None or vision_sequences.numel() == 0:
                        history = torch.zeros(half, 0, dtype=torch.long, device=x.device)
                    else:
                        history = vision_sequences[:half, :, level]
                    logits = logits_processor(history, logits)
                if do_sample:
                    sampled = torch.multinomial(torch.softmax(logits, -1), 1).squeeze(1)
                else:
                    sampled = logits.argmax(-1)
                logprob = torch.log_softmax(logits, -1).gather(1, sampled[:, None]).squeeze(1)
                gumbel = -torch.log(-torch.log(
                    torch.rand(half, device=x.device).clamp_min(1e-20)
                ).clamp_min(1e-20))
                sampled_all[:, slot] = sampled
                clean_logprob[:, slot] = logprob
                confidence[:, slot] = torch.where(
                    ~decided[:half, slot], logprob + annealed_temp * gumbel,
                    torch.full_like(logprob, float("-inf")),
                )
            topk = torch.topk(confidence, k=n_new, dim=1).indices
            newly = torch.zeros_like(decided[:half])
            newly.scatter_(1, topk, True)
            newly &= ~decided[:half]
            tokens[:half] = torch.where(newly, sampled_all, tokens[:half])
            # Keep the pre-commit state for optional self-refine.  A plain
            # slice is a view and would be mutated by the in-place OR below.
            was_decided = decided[:half].clone()
            decided[:half] |= newly
            changed = newly
            if gamma is not None:
                can_refine = (was_decided & (~newly)
                              & (sampled_all != tokens[:half])
                              & (clean_logprob.exp() > gamma))
                tokens[:half] = torch.where(can_refine, sampled_all, tokens[:half])
                changed |= can_refine
            if use_cfg:
                tokens[half:] = torch.where(changed, sampled_all, tokens[half:])
                decided[half:] = decided[:half]
        return tokens.view(rows, self.mms_unit, self.depth).reshape(
            rows * self.mms_unit, self.depth
        )


def build_explicit_grouped_head(config):
    head_type = str(config.visual_config.image_head_config.image_head_type).lower()
    if head_type == "ar":
        return ARHead(config)
    if head_type == "maskgit":
        return MaskGiTHead(config)
    if head_type == "dspark_attn":
        return DSparkAttnHead(config)
    raise ValueError(f"unsupported image_head_type={head_type!r}")



class OmniImageHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.llm_hidden_size = config.hidden_size
        self.codebook_sizes = config.visual_quantizer_config.codebook_sizes
        self.config = config.visual_config.image_head_config
        self.transformer_ffn_scale = self.config.image_head_transformer_ffn_scale
        self.transformer_dims = self.config.image_head_transformer_dims
        num_transformer_layers = self.config.image_head_transformer_layers
        explicit_head_type = getattr(self.config, "image_head_type", None)
        self.explicit_head_type = None if explicit_head_type is None else str(explicit_head_type).lower()
        if self.explicit_head_type is not None:
            self.grouped_head = build_explicit_grouped_head(config)
            self.mask = self.explicit_head_type == "maskgit"
            self.block_ar = False
            self.spatial_causal = False
            self.pq = bool(getattr(self.config, "image_head_pq", False))
            self.gradient_checkpointing = True
            self.mask_refine_loss_weight = float(getattr(
                self.grouped_head, "mask_refine_loss_weight", 1.0
            ))
            if not self.config.enable:
                for param in self.parameters():
                    param.requires_grad = False
            return
        # 沿 batch 维 chunk 串行跑 depth transformer 以省反向显存峰值; 默认 1 = 旧行为.
        # 严格行级独立: cumsum 沿 depth=1, FlashVarLenAttention seqlens=depth*ones(bsz),
        # FFN/RMSNorm/heads 都 per-row, 沿 dim-0 切块数学等价.
        self.image_head_num_chunks = int(getattr(self.config, "image_head_num_chunks", 1))

        unmerge_size = int(getattr(config.visual_config, "merger_merge_size", 1))
        self.unmerge_size = unmerge_size
        self.mms_unit = unmerge_size * unmerge_size
        self.depth = len(self.codebook_sizes)
        # block_ar 模式: 把 mms_unit*depth 个 token 铺平成一个 block-causal 序列
        # (见 BlockARDepthTransformerLayer); 不再用 SpatialUnmerger, 改"复制 mms_unit 遍 + pos_embed".
        self.block_ar = bool(getattr(self.config, "image_head_block_ar", False))
        # spatial_causal 模式: 把 mms_unit*depth 个 token 铺平成 spatial-major 的纯 causal 序列
        # (见 SpatialCausalDepthTransformerLayer); 与 block_ar 共用"复制 mms_unit 遍 + pos_embed".
        self.spatial_causal = bool(getattr(self.config, "image_head_spatial_causal", False))
        # pq 模式: tokenizer 是 Product Quantization (UniTok 式通道切分), 8 个 group 并行独立
        # (无残差顺序)。深度头仍是 per-position 因果深度 AR, 但 teacher-forcing 用"逐个 append
        # group embedding"(concat 语义) 而非 cumsum(残差重建)。且 head 自带反馈嵌入
        # self.pq_emb_layers (llm_hidden 宽), 不复用 bridge 的 embedding_layers ——因为 PQ 的
        # concat bridge 把 per-group embedding 压到 hidden//depth 宽, 无法直接与 x concat。
        # 只在默认路径实现, 与 block_ar/spatial_causal 互斥。
        self.pq = bool(getattr(self.config, "image_head_pq", False))
        assert sum([self.block_ar, self.spatial_causal, self.pq]) <= 1, "block_ar / spatial_causal / pq 互斥"
        # mask 模式 (MaskGit / absorbing-state): 8 个 depth-level 当无序集合, 训练随机遮一部分,
        # 双向头一次预测全部 level; 推理按 schedule 少步迭代 unmask. 与 block_ar/spatial_causal
        # 互斥, 但可与 pq 共存——pq 只驱动输入 bridge 的 concat 语义, 头计算由 mask 接管.
        self.mask = bool(getattr(self.config, "image_head_mask", False))
        assert not (self.mask and (self.block_ar or self.spatial_causal)), "mask 与 block_ar/spatial_causal 互斥"
        if self.mask:
            # 训练遮罩数分布. "uniform_count": num_masked ~ 离散均匀{1..depth} (每档等概率,
            # 全遮占 ~1/depth, 保证推理 step-0 的全遮起始态训得到); "logit_normal": 旧 SD3/MBM
            # 式 logit-normal timestep -> mask_ratio (中心 0.5, 高遮档罕见).
            self.mask_ratio_dist = str(getattr(self.config, "image_head_mask_ratio_dist", "uniform_count"))
            self.mask_logitnorm_std = float(getattr(self.config, "image_head_mask_logitnorm_std", 0.8))
            self.mask_temperature = float(getattr(self.config, "image_head_mask_temperature", 1.0))
            self.mask_schedule = list(getattr(self.config, "image_head_mask_schedule", [1, 2, 2, 3]))
            assert all(s > 0 for s in self.mask_schedule), "mask_schedule 各项须 > 0"
            assert sum(self.mask_schedule) == self.depth, \
                f"mask_schedule 之和 ({sum(self.mask_schedule)}) 必须 == depth ({self.depth})"
            assert all(self.mask_schedule[i] <= self.mask_schedule[i + 1] for i in range(len(self.mask_schedule) - 1)), \
                "mask_schedule 须非递减 (前期少放、后期多放)"
            # self-refine: 训练时以 eta 概率把"未 mask"的 level 换成近邻 token (top-k 里均匀采),
            # target 仍是干净 GT -> 模型学会纠正"已定但略错"的槽; 推理时对已定 level 若重新
            # 预测出概率 > gamma 的不同 token 则覆盖 (见 sample)。代码兜底默认为"关闭"
            # (eta=0 / gamma=None), 老 ckpt/JSON 无这些 key 时行为不变; 启用值只写在 base JSON。
            self.mask_refine_eta = float(getattr(self.config, "image_head_mask_refine_eta", 0.0))
            self.mask_refine_perturb_k = int(getattr(self.config, "image_head_mask_refine_perturb_k", 8))
            self.mask_refine_loss_weight = float(getattr(self.config, "image_head_mask_refine_loss_weight", 1.0))
            _refine_gamma = getattr(self.config, "image_head_mask_refine_gamma", None)
            self.mask_refine_gamma = None if _refine_gamma is None else float(_refine_gamma)

        if self.transformer_ffn_scale > 0 and not self.mask:
            self.hidden_norm = RMSNorm(self.llm_hidden_size)
            self.hidden_proj = nn.Linear(self.llm_hidden_size, self.transformer_dims, bias=False)

        if self.mask:
            assert self.transformer_ffn_scale > 0, "mask 模式要求 image_head_transformer_ffn_scale > 0"
            self.transformer_layers = nn.ModuleList([
                MaskDepthTransformerLayer(config, self.depth)
                for _ in range(num_transformer_layers)
            ])
            # 区分 depth 个 level 的可学习位置编码 (打破 level 对称). 同上用 Embedding 保证初始化.
            self.pos_embed = nn.Embedding(self.depth, self.transformer_dims)
            # 单个可学习 MASK token (absorbing 状态), llm_hidden 宽, 与 GT embedding 同空间.
            self.mask_embed = nn.Embedding(1, self.llm_hidden_size)
            # 头自有的逐 level 反馈嵌入 (llm_hidden 宽, vq+1 含 EOS 槽), 不复用 bridge 的
            # embedding_layers——PQ 下后者只有 hidden//depth 宽.
            self.mask_emb_layers = nn.ModuleList([
                nn.Embedding(vq_size + 1, self.llm_hidden_size) for vq_size in self.codebook_sizes
            ])
            # slot 特征 (level embedding/MASK + pos_embed) 投到 tdims; 条件 x 单独投成 AdaLN 输入.
            # 条件不再加进 slot——DiT 风格: x 只通过每层 AdaLN 的 shift/scale/gate 注入 (无时间步).
            self.input_proj = nn.Linear(self.llm_hidden_size, self.transformer_dims, bias=True)
            self.input_norm = RMSNorm(self.transformer_dims)
            self.cond_proj = nn.Linear(self.llm_hidden_size, self.transformer_dims, bias=True)
            # 出 head 前再来一次 AdaLN (DiT FinalLayer): scale/shift.
            self.final_norm = RMSNorm(self.transformer_dims)
            self.final_adaLN = nn.Sequential(
                nn.SiLU(), nn.Linear(self.transformer_dims, 2 * self.transformer_dims)
            )
            # 训练随机遮罩缓存 (供 modeling_omni 做 masked=1.0/unmasked=0.1 加权 loss).
            self._last_train_mask = None
            # self-refine 训练扰动位置缓存 [N,depth] bool (被换成近邻的 未mask 槽), 供 loss 第三档权重.
            self._last_perturb_mask = None
            # tokenizer 提供的 per-level top-k 近邻表 list[[V_l,k] long]; 由训练入口 init 时挂上,
            # 只训练用, 不入 state_dict. None = 未挂 -> 扰动自动关闭.
            self._neighbor_table = None
            self._warned_no_neighbor = False
            self.spatial_unmerger = None
        elif self.block_ar or self.spatial_causal:
            assert self.transformer_ffn_scale > 0, "block_ar / spatial_causal 模式要求 image_head_transformer_ffn_scale > 0"
            self.depth_total = self.mms_unit * self.depth
            layer_cls = SpatialCausalDepthTransformerLayer if self.spatial_causal else BlockARDepthTransformerLayer
            self.transformer_layers = nn.ModuleList([
                layer_cls(config, self.depth, self.mms_unit)
                for _ in range(num_transformer_layers)
            ])
            # 可学习位置编码: 区分 depth_total 个 (depth, spatial) 位置, 打破空间对称.
            # 用 nn.Embedding 而非裸 nn.Parameter: from_pretrained 的 _init_weights 只遍历 module,
            # 裸 Parameter 不会被初始化 (meta device 下变 NaN); Embedding 会被正确 normal_ 初始化.
            self.pos_embed = nn.Embedding(self.depth_total, self.transformer_dims)
            self.spatial_unmerger = None
        else:
            self.transformer_layers = nn.ModuleList([
                CasualDepthTransformerLayer(config, len(self.codebook_sizes))
                for _ in range(num_transformer_layers)
            ])
            if unmerge_size > 1:
                unmerger_inter = int(getattr(
                    self.config,
                    "unmerger_intermediate_size",
                    self.llm_hidden_size * unmerge_size * unmerge_size,
                ))
                unmerger_act = getattr(self.config, "unmerger_hidden_act", "gelu")
                self.spatial_unmerger = SpatialUnmerger(
                    hidden_size=self.llm_hidden_size,
                    intermediate_size=unmerger_inter,
                    merge_size=unmerge_size,
                    hidden_act=unmerger_act,
                    out_dim=self.llm_hidden_size,
                )
            else:
                self.spatial_unmerger = None

        self.headnorm = RMSNorm(self.transformer_dims)
        self.heads = nn.ModuleList([
            nn.Linear(self.transformer_dims, vq_size + 1) for vq_size in self.codebook_sizes
        ])

        # PQ 深度头自有反馈嵌入 (UniTok AR_head.codebooks 式): depth-1 个 (位置 0 = x 无反馈),
        # 宽度 = llm_hidden_size, 与 bridge 的窄 embedding 解耦。teacher-forcing 时 append 而非 cumsum。
        if self.pq and not self.mask:
            self.pq_emb_layers = nn.ModuleList([
                nn.Embedding(vq_size + 1, self.llm_hidden_size)
                for vq_size in self.codebook_sizes[:-1]
            ])

        self.gradient_checkpointing = True
        if not self.config.enable:
            for param in self.parameters():
                param.requires_grad = False

    def _mask_transformer(self, tok, cond_x, use_per_layer_ckpt):
        """mask 模式共用的主干. tok: [n, depth, llm_hidden] (level embedding 或 MASK token);
        cond_x: [n, llm_hidden] (LLM 条件, 通过 AdaLN 注入, 不加进 slot).
        input_proj -> +pos_embed -> input_norm -> AdaLN 双向层(条件 c) -> FinalLayer AdaLN -> per-level heads.
        """
        hidden_states = self.input_proj(tok)  # [n, depth, tdims]
        hidden_states = hidden_states + self.pos_embed.weight.unsqueeze(0)
        hidden_states = self.input_norm(hidden_states)
        c = self.cond_proj(cond_x)  # [n, tdims]
        for tlayer in self.transformer_layers:
            if use_per_layer_ckpt:
                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)
                    return custom_forward
                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(tlayer), hidden_states, c,
                )
            else:
                hidden_states = tlayer(hidden_states, c)
        shift, scale = self.final_adaLN(c).chunk(2, dim=-1)
        hidden_states = _ada_modulate(self.final_norm(hidden_states), shift, scale)
        logits = [self.heads[l](hidden_states[:, l]) for l in range(self.depth)]
        # ckpt(use_reentrant=False) 要求返回 tuple/Tensor 序列, 不允许 list.
        return tuple(logits)

    def set_neighbor_table(self, tables):
        """挂上 tokenizer 提供的 per-level top-k 近邻表 (list, 第 l 项 [V_l, k])。

        由训练入口在 init 时调用一次 (tokenizer 与 model 同时在场那一刻)。只训练用、
        不入 state_dict; 存普通属性引用 (long)。tables=None 或不调用 -> 扰动关闭。
        """
        if self.explicit_head_type is not None:
            setter = getattr(self.grouped_head, "set_neighbor_table", None)
            if setter is not None:
                setter(tables)
            return
        if tables is None:
            self._neighbor_table = None
            return
        self._neighbor_table = [t.to(torch.long) for t in tables]

    def _apply_refine_perturb(self, visual_tokens, masks):
        """self-refine 训练扰动: 以 eta 概率把"未 mask & 非 EOS"的 level 换成 top-k 近邻里的一个。

        返回用于 *embedding* 的 (可能被扰动的) indices 拷贝; 同时把扰动位置存到 _last_perturb_mask
        供 loss 第三档加权。CE target 仍是干净 visual_tokens (在 modeling_omni), 这里只动输入 embedding。
        必须在 checkpoint 区域之外调用 (同 _sample_masks, 否则 backward 重算会重采、输入/target 错位)。
        """
        eta = self.mask_refine_eta
        if not (self.training and eta > 0.0):
            self._last_perturb_mask = None
            return visual_tokens
        if self._neighbor_table is None:
            if not self._warned_no_neighbor:
                print("[image_head][self-refine] eta>0 但未挂近邻表 (set_neighbor_table 未调用/"
                      "OMNI_RVQ_TOKENIZER_PATH 未设或 tokenizer 无 codebook_neighbors), 扰动关闭。")
                self._warned_no_neighbor = True
            self._last_perturb_mask = None
            return visual_tokens
        D = self.depth
        device = visual_tokens.device
        # 只扰动 "未遮 & 非 EOS 行" 的槽; EOS 行填的是 codebook_size, 越界近邻表.
        is_eos = visual_tokens[:, 0] == self.codebook_sizes[0]          # [N]
        elig = (~masks) & (~is_eos).unsqueeze(1)                        # [N, D]
        draw = torch.rand(visual_tokens.shape[0], D, device=device) < eta
        perturb_pos = elig & draw                                      # [N, D] bool
        vt = visual_tokens.clone()
        for l in range(D):
            pos_l = perturb_pos[:, l]
            if not bool(torch.any(pos_l)):
                continue
            tbl = self._neighbor_table[l]
            if tbl.device != device:
                tbl = tbl.to(device)
                self._neighbor_table[l] = tbl
            idx = visual_tokens[pos_l, l]                              # [m] 当前 token (有效 index)
            pick = torch.randint(0, tbl.shape[1], (idx.shape[0],), device=device)
            vt[pos_l, l] = tbl[idx, pick].to(vt.dtype)                 # 近邻 (必 != 自身)
        self._last_perturb_mask = perturb_pos
        return vt

    def _sample_masks(self, vt):
        """按配置分布逐行采训练遮罩. vt: [n, depth] GT indices -> masks [n, depth] bool (True=遮).
        在 checkpoint 区域之外调用一次 (否则 backward 重跑会重采), 结果同时喂 _run_mask 和加权 loss.
        """
        D = self.depth
        n = vt.shape[0]
        device = vt.device
        # 极端测试 case (OMNI_MASK_FORCE_ALL_MASKED=1): 每行全部 depth-level 都遮
        # (mask_ratio=1.0), 即训练永远从"全 MASK"起始态学一次预测全部 level。
        # EOS 行仍强制不遮 (下方逻辑), 与常规路径一致。
        if os.environ.get("OMNI_MASK_FORCE_ALL_MASKED", "0") == "1":
            num_masked = torch.full((n,), D, device=device, dtype=torch.long)
        elif self.mask_ratio_dist == "logit_normal":
            z = torch.randn(n, device=device, dtype=torch.float32) * self.mask_logitnorm_std
            t = torch.sigmoid(z)
            mask_ratio = (1.0 - t).clamp(min=1.0 / D, max=1.0)
            num_masked = (mask_ratio * D).round().clamp(min=1, max=D).long()
        else:
            # uniform_count: num_masked ~ 离散均匀{1..D}, 每档等概率 (全遮占 ~1/D).
            num_masked = torch.randint(1, D + 1, (n,), device=device, dtype=torch.long)
        rand = torch.rand(n, D, device=device)
        ranks = rand.argsort(dim=1).argsort(dim=1)  # 每行随机名次
        masks = ranks < num_masked.unsqueeze(1)  # [n, D] bool
        # EOS 行 (所有 level == codebook_size, image_end 填充) 强制不遮.
        is_eos = vt[:, 0] == self.codebook_sizes[0]
        if is_eos.any():
            masks = masks & (~is_eos).unsqueeze(1)
        return masks

    def _run_mask(self, x_chunk, vt_chunk, masks_chunk, use_per_layer_ckpt):
        """mask / MaskGit 训练单 chunk. x_chunk: [n, llm_hidden] (条件); vt_chunk: [n, depth] (GT);
        masks_chunk: [n, depth] bool (外部预采, 见 _sample_masks). 返回 depth 个 [n, V_l+1].
        被遮 level 换成可学习 MASK token, 未遮用 GT 的 level embedding; 条件 x 只走 AdaLN, 不加进 slot.
        """
        D = self.depth
        H = self.llm_hidden_size
        gt = torch.stack([self.mask_emb_layers[l](vt_chunk[:, l]) for l in range(D)], dim=1)  # [n, D, H]
        mask_vec = self.mask_embed.weight[0].to(gt.dtype).view(1, 1, H)
        tok = torch.where(masks_chunk.unsqueeze(-1), mask_vec, gt)  # [n, D, H]
        return self._mask_transformer(tok, x_chunk, use_per_layer_ckpt)

    def _forward_mask(self, x, visual_tokens):
        """mask 模式训练前向. x: [N, llm_hidden]; visual_tokens: [N, depth].
        先在 ckpt 外采一次遮罩(存 self._last_train_mask 供加权 loss), 再按 image_head_num_chunks
        沿 dim-0 分块跑 (与默认路径一致). 返回 depth 个 [N, V_l+1] logits list.
        """
        n = x.shape[0]
        masks = self._sample_masks(visual_tokens)  # [N, D]
        self._last_train_mask = masks
        # self-refine: 未遮槽以 eta 概率换成近邻 token (只作 embedding 输入, target 仍干净).
        vt_embed = self._apply_refine_perturb(visual_tokens, masks)
        chunked = self.training and self.image_head_num_chunks > 1 and n >= self.image_head_num_chunks
        if not chunked:
            use_inner_ckpt = bool(self.gradient_checkpointing and self.training)
            return list(self._run_mask(x, vt_embed, masks, use_inner_ckpt))
        x_chunks = torch.chunk(x, self.image_head_num_chunks, dim=0)
        vt_chunks = torch.chunk(vt_embed, self.image_head_num_chunks, dim=0)
        m_chunks = torch.chunk(masks, self.image_head_num_chunks, dim=0)
        per_chunk_logits = []
        for cx, cvt, cm in zip(x_chunks, vt_chunks, m_chunks):
            cl = torch.utils.checkpoint.checkpoint(
                self._run_mask, cx, cvt, cm, False, use_reentrant=False,
            )
            per_chunk_logits.append(cl)
        return [torch.cat([pc[h] for pc in per_chunk_logits], dim=0) for h in range(self.depth)]

    @torch.no_grad()
    def sample(self, x, visual_emb_layers, *, cfg_scale=1.0, mms_unit=1,
               batch_size=None, do_sample=True, temperature=None,
               logits_processor=None, vision_sequences=None, refine_gamma=None,
               image_cfg_scale=None):
        """MaskGit 少步迭代采样. x: [R, llm_hidden] (R = batch_size*mms_unit,
        cfg 时前半 cond 后半 uncond). 返回 tokens [R, depth] (indices, 与 pipeline
        的 next_token_ids 契约一致). 忽略传入的 bridge visual_emb_layers.

        refine_gamma: self-refine 推理门槛; None 回退到 self.mask_refine_gamma。非 None 时,
        每步 commit 后允许"已定 level 若重新预测出概率 > gamma 的不同 token"被覆盖 (纠错)。
        """
        if self.explicit_head_type is not None:
            self.grouped_head.gradient_checkpointing = self.gradient_checkpointing
            return self.grouped_head.sample(
                x, cfg_scale=cfg_scale, mms_unit=mms_unit,
                batch_size=batch_size, do_sample=do_sample,
                temperature=temperature, logits_processor=logits_processor,
                vision_sequences=vision_sequences, refine_gamma=refine_gamma,
                **({"image_cfg_scale": image_cfg_scale} if image_cfg_scale is not None else {}),
            )
        if image_cfg_scale is not None:
            raise ValueError("Image CFG requires an explicit AR or DSpark head")
        D = self.depth
        H = self.llm_hidden_size
        R = x.shape[0]
        device = x.device
        temperature = self.mask_temperature if temperature is None else float(temperature)
        refine_gamma = self.mask_refine_gamma if refine_gamma is None else float(refine_gamma)
        # 极端测试 case (OMNI_MASK_ONE_STEP=1): 一步把全部 depth-level 预测出来
        # (schedule=[depth]), 对应"全 mask 起始 -> 单次前向 commit 所有 level"。
        schedule = [D] if os.environ.get("OMNI_MASK_ONE_STEP", "0") == "1" else self.mask_schedule
        num_steps = len(schedule)
        use_cfg = cfg_scale != 1.0
        half = R // 2 if use_cfg else R  # 需要真正决策的行数 (cond 半)
        _mms = int(mms_unit) if mms_unit else 1

        tokens = torch.zeros(R, D, dtype=torch.long, device=device)
        decided = torch.zeros(R, D, dtype=torch.bool, device=device)
        mask_vec = self.mask_embed.weight[0].view(1, 1, H)

        for step, n_new in enumerate(schedule):
            # 拼 token slot: 已定 level 用其 token 的 level embedding, 未定用 MASK token.
            # 条件 x 不加进 slot——走 _mask_transformer 内的 AdaLN.
            gt = torch.stack([self.mask_emb_layers[l](tokens[:, l]) for l in range(D)], dim=1)  # [R, D, H]
            tok = torch.where(decided.unsqueeze(-1), gt, mask_vec.to(gt.dtype))
            logits = self._mask_transformer(tok, x, use_per_layer_ckpt=False)  # depth 个 [R, V_l+1]

            annealed_temp = temperature * (1.0 - step / max(num_steps, 1))
            # 逐 level 先处理成 [half, V_l] (去 EOS 槽 + CFG + logits_processor), 再算采样与置信度.
            step_sampled = torch.zeros(half, D, dtype=torch.long, device=device)
            step_conf = torch.full((half, D), float("-inf"), device=device)
            # 干净 logprob (不含 gumbel) 供 self-refine 的 prob>gamma 判定; 所有 level 都记 (含已定).
            step_logprob = torch.full((half, D), float("-inf"), device=device)
            for l in range(D):
                vq = self.codebook_sizes[l]
                lg = logits[l][:, :vq].float()  # 去掉 EOS 槽 (c2i 定长, EOS 不参与)
                if use_cfg:
                    cond_lg, uncond_lg = lg[:half], lg[half:]
                    lg = uncond_lg + cfg_scale * (cond_lg - uncond_lg)  # [half, vq]
                if logits_processor is not None:
                    hist = vision_sequences[:half, :, l] if (
                        vision_sequences is not None and vision_sequences.numel() > 0) else \
                        torch.zeros(half, 0, dtype=torch.long, device=device)
                    if _mms > 1:
                        hist = hist.repeat_interleave(_mms, dim=0)
                    lg = logits_processor(hist, lg)
                if do_sample:
                    probs = torch.softmax(lg, dim=-1)
                    sampled = torch.multinomial(probs, num_samples=1).squeeze(1)  # [half]
                else:
                    sampled = lg.argmax(dim=-1)
                logprob = torch.log_softmax(lg, dim=-1).gather(1, sampled.unsqueeze(1)).squeeze(1)
                # MaskGit 置信度: log-prob + 退火 gumbel 噪声 (打散并鼓励早期多样).
                gumbel = -torch.log(-torch.log(torch.rand(half, device=device).clamp_min(1e-20)).clamp_min(1e-20))
                conf = logprob + annealed_temp * gumbel
                undecided_l = ~decided[:half, l]
                step_sampled[:, l] = sampled
                step_logprob[:, l] = logprob
                step_conf[:, l] = torch.where(undecided_l, conf, torch.full_like(conf, float("-inf")))

            # 每行在"未定 level"里挑置信度最高的 n_new 个确定下来.
            topk = torch.topk(step_conf, k=n_new, dim=1).indices  # [half, n_new]
            newly = torch.zeros(half, D, dtype=torch.bool, device=device)
            newly.scatter_(1, topk, True)
            newly = newly & (~decided[:half])  # 只在未定位置生效
            tokens[:half] = torch.where(newly, step_sampled, tokens[:half])
            decided[:half] = decided[:half] | newly
            # self-refine: 对 *上一步之前* 已定的 level, 若这一步重新预测出概率 > gamma 的不同 token,
            # 覆盖之 (纠错)。decided 不变 (仍是已定); newly 与 can_refine 互斥 (~newly & 之前已定)。
            changed = newly
            if refine_gamma is not None:
                refine_prob = step_logprob.exp()  # [half, D] 干净概率
                can_refine = (decided[:half] & (~newly)
                              & (step_sampled != tokens[:half])
                              & (refine_prob > refine_gamma))
                tokens[:half] = torch.where(can_refine, step_sampled, tokens[:half])
                changed = newly | can_refine
            if use_cfg:
                # 镜像到 uncond 半: newly 决定 + refine 覆盖 都要同步取值, 否则 cond/uncond 错位.
                tokens[half:] = torch.where(changed, step_sampled, tokens[half:])
                decided[half:] = decided[:half]

        return tokens

    def _run_block_ar(self, x_chunk, vt_chunk, visual_emb_layers, use_per_layer_ckpt):
        """block_ar 模式单 chunk 逻辑.
        x_chunk: [n, llm_hidden] (n = N_lvl2 chunk); vt_chunk: [n*mms_unit, depth] (spatial-major).
        返回 depth 个 [n*mms_unit, V_i+1], 行序 region*mms_unit+s.
        """
        M = self.mms_unit
        D = self.depth
        H = self.llm_hidden_size
        n = x_chunk.shape[0]
        vt2 = vt_chunk.view(n, M, D)
        # level 0..D-2 的 GT embedding -> [n, D-1, M, H], 沿 depth 累加做 teacher forcing.
        gt = torch.stack([visual_emb_layers[k](vt2[:, :, k]) for k in range(D - 1)], dim=1)
        gt_cum = torch.cumsum(gt, dim=1)  # [n, D-1, M, H]
        zeros = torch.zeros(n, 1, M, H, device=gt_cum.device, dtype=gt_cum.dtype)
        gt_part = torch.concat([zeros, gt_cum], dim=1)  # [n, D, M, H]; depth-0 槽无 GT
        seq = x_chunk.view(n, 1, 1, H) + gt_part  # x 复制到 D*M = "复制 mms_unit 遍"
        seq = seq.reshape(n, D * M, H)  # depth-major: [d0s0,d0s1,..,d7s3]
        assert seq.size(1) == self.depth_total

        hidden_states = self.hidden_norm(seq)
        hidden_states = self.hidden_proj(hidden_states)  # [n, depth_total, tdims]
        hidden_states = hidden_states + self.pos_embed.weight.unsqueeze(0)

        for tlayer in self.transformer_layers:
            if use_per_layer_ckpt:
                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)
                    return custom_forward
                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(tlayer), hidden_states,
                )
            else:
                hidden_states = tlayer(hidden_states)
        hidden_states = self.headnorm(hidden_states)
        out = hidden_states.view(n, D, M, self.transformer_dims)
        logits = [self.heads[i](out[:, i]).reshape(n * M, -1) for i in range(D)]
        # ckpt(use_reentrant=False) 要求返回 tuple/Tensor 序列, 不允许 list.
        return tuple(logits)

    def _run_spatial_causal(self, x_chunk, vt_chunk, visual_emb_layers, use_per_layer_ckpt):
        """spatial_causal 模式单 chunk 逻辑 (纯 causal, spatial-major).
        x_chunk: [n, llm_hidden]; vt_chunk: [n*mms_unit, depth] (spatial-major).
        teacher forcing: 预测 X(s,l) 的加性输入 =
          l==0: h + R(s-1)  (上一空间组的完整 RVQ 重建; s==0 时为 0)
          l>=1: h + sum_{k<l} X(s,k)  (组内局部部分重建, 不含组边界项)
        返回 depth 个 [n*mms_unit, V_l+1], 行序 region*mms_unit+s (与 block_ar 一致).
        """
        M = self.mms_unit
        D = self.depth
        H = self.llm_hidden_size
        n = x_chunk.shape[0]
        vt2 = vt_chunk.view(n, M, D)
        # 全部 D 级 GT embedding (含最后一级, 组边界 full-recon 需要) -> [n, M, D, H].
        gt = torch.stack([visual_emb_layers[l](vt2[:, :, l]) for l in range(D)], dim=2)
        cum = torch.cumsum(gt, dim=2)  # inclusive: cum[..,l] = sum_{k<=l}
        partial = cum - gt  # exclusive 组内部分和: partial[..,l] = sum_{k<l}, partial[..,0]=0
        full_recon = cum[:, :, -1, :]  # [n, M, H] 每个空间组的完整重建 R(s)
        # 组边界 base: boundary[:, s] = R(s-1), boundary[:, 0] = 0.
        boundary = torch.zeros_like(full_recon)
        boundary[:, 1:, :] = full_recon[:, :-1, :]
        # level-0 槽用组边界项, level 1..D-1 用组内局部部分和.
        additive = torch.cat([boundary.unsqueeze(2), partial[:, :, 1:, :]], dim=2)  # [n,M,D,H]
        seq = x_chunk.view(n, 1, 1, H) + additive
        seq = seq.reshape(n, M * D, H)  # spatial-major: [s0l0,..,s0l7,s1l0,..]
        assert seq.size(1) == self.depth_total

        hidden_states = self.hidden_norm(seq)
        hidden_states = self.hidden_proj(hidden_states)  # [n, depth_total, tdims]
        hidden_states = hidden_states + self.pos_embed.weight.unsqueeze(0)

        for tlayer in self.transformer_layers:
            if use_per_layer_ckpt:
                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)
                    return custom_forward
                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(tlayer), hidden_states,
                )
            else:
                hidden_states = tlayer(hidden_states)
        hidden_states = self.headnorm(hidden_states)
        out = hidden_states.view(n, M, D, self.transformer_dims)  # spatial-major
        # head[l] 预测 level l; 行序 region*M+s (与 block_ar 一致).
        logits = [self.heads[l](out[:, :, l]).reshape(n * M, -1) for l in range(D)]
        # ckpt(use_reentrant=False) 要求返回 tuple/Tensor 序列, 不允许 list.
        return tuple(logits)

    def _run_after_unmerger(self, x_chunk, vt_chunk, visual_emb_layers, use_per_layer_ckpt):
        """单个 chunk 上的全部 image_head 逻辑 (在 spatial_unmerger 之后).
        被外层 chunked-ckpt 包住时, use_per_layer_ckpt 应为 False (避免嵌套 ckpt).
        """
        if self.block_ar:
            return self._run_block_ar(x_chunk, vt_chunk, visual_emb_layers, use_per_layer_ckpt)
        if self.spatial_causal:
            return self._run_spatial_causal(x_chunk, vt_chunk, visual_emb_layers, use_per_layer_ckpt)
        if len(self.codebook_sizes) == 1:
            # A single codebook has no preceding groups for teacher forcing.
            depth_visual_embed = x_chunk.new_empty((x_chunk.shape[0], 0, self.llm_hidden_size))
        elif self.pq:
            # PQ: 8 个 group 并行独立, 无残差顺序。teacher-forcing 逐个 append 各 group 的反馈
            # embedding (不 cumsum), 用 head 自有 pq_emb_layers (非 bridge 的窄 embedding)。
            # 位置 0 = x → 预测 group 0; 位置 i = e(gt_{i-1}) → 预测 group i, 因果 attend 到 x + 前 i-1 个 group。
            depth_visual_embed = torch.stack([
                self.pq_emb_layers[i](vt_chunk[..., i])
                for i in range(len(self.codebook_sizes) - 1)
            ], dim=1)
        else:
            depth_visual_embed = torch.stack([
                visual_emb_layers[i](vt_chunk[..., i])
                for i, vq_size in enumerate(self.codebook_sizes[:-1])
            ], dim=1)
            depth_visual_embed = torch.cumsum(depth_visual_embed, dim=1)
        hidden_states = torch.concat(
            [x_chunk.reshape(-1, 1, self.llm_hidden_size), depth_visual_embed], dim=1
        )
        assert hidden_states.size(1) == len(self.codebook_sizes)

        if self.transformer_ffn_scale > 0:
            hidden_states = self.hidden_norm(hidden_states)
            hidden_states = self.hidden_proj(hidden_states)

        for i, tlayer in enumerate(self.transformer_layers):
            if use_per_layer_ckpt:
                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)
                    return custom_forward
                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(tlayer), hidden_states,
                )
            else:
                hidden_states = tlayer(hidden_states)
        hidden_states = self.headnorm(hidden_states)
        logits = [head(hidden_states[:, i]) for i, head in enumerate(self.heads)]
        # ckpt(use_reentrant=False) 要求返回 tuple/Tensor 序列, 不允许 list.
        return tuple(logits)

    def forward(self, x, visual_tokens, visual_emb_layers):
        """x: [N_lvl2, llm_hidden] when SpatialUnmerger/block_ar 启用 else [N_raw, llm_hidden].
        visual_tokens: [N_raw, depth]，调用方需保证 block-major 顺序与 unmerger 输出一致。
        """
        if self.explicit_head_type is not None:
            self.grouped_head.gradient_checkpointing = self.gradient_checkpointing
            logits = self.grouped_head(x, visual_tokens)
            self._last_train_mask = getattr(self.grouped_head, "_last_train_mask", None)
            self._last_perturb_mask = getattr(self.grouped_head, "_last_perturb_mask", None)
            return logits
        if self.mask:
            # mask / MaskGit: x=[N,H] 是条件, 走 AdaLN 注入; 自带遮罩采样与分块 (见 _forward_mask).
            return self._forward_mask(x, visual_tokens)
        if self.block_ar or self.spatial_causal:
            # block_ar / spatial_causal: x=[N_lvl2,H] 直接进, 内部复制 mms_unit 遍 + 加 pos_embed.
            n_lvl2 = x.shape[0]
            chunked = self.training and self.image_head_num_chunks > 1 and n_lvl2 >= self.image_head_num_chunks
            if not chunked:
                use_inner_ckpt = bool(self.gradient_checkpointing and self.training)
                return list(self._run_after_unmerger(x, visual_tokens, visual_emb_layers, use_inner_ckpt))
            # 先 reshape 再 chunk: 沿 N_lvl2 切, 避免把同一 region 的 mms_unit 个空间位置切断.
            vt = visual_tokens.view(n_lvl2, self.mms_unit, self.depth)
            x_chunks = torch.chunk(x, self.image_head_num_chunks, dim=0)
            vt_chunks = torch.chunk(vt, self.image_head_num_chunks, dim=0)
            per_chunk_logits = []
            for cx, cvt in zip(x_chunks, vt_chunks):
                cvt2 = cvt.reshape(-1, self.depth)  # [cn*mms_unit, depth]
                cl = torch.utils.checkpoint.checkpoint(
                    self._run_after_unmerger,
                    cx, cvt2, visual_emb_layers, False,
                    use_reentrant=False,
                )
                per_chunk_logits.append(cl)
            n_heads = len(self.heads)
            logits = [torch.cat([per_chunk_logits[c][h] for c in range(len(per_chunk_logits))], dim=0)
                      for h in range(n_heads)]
            return logits

        if self.spatial_unmerger is not None:
            x = self.spatial_unmerger(x)

        # x 此时形状 [N_raw, llm_hidden], visual_tokens [N_raw, depth]; N_raw 与 chunk 维对齐.
        chunked = self.training and self.image_head_num_chunks > 1 and x.shape[0] >= self.image_head_num_chunks
        if not chunked:
            # 旧路径: 单次 _run_after_unmerger, per-layer ckpt 由 self.gradient_checkpointing 控制.
            use_inner_ckpt = bool(self.gradient_checkpointing and self.training)
            return list(self._run_after_unmerger(x, visual_tokens, visual_emb_layers, use_inner_ckpt))

        # chunked 路径: 沿 dim-0 切块, 每块外层一次 ckpt 包住整个后续计算; 关掉内层 per-layer ckpt 避免嵌套.
        x_chunks = torch.chunk(x, self.image_head_num_chunks, dim=0)
        vt_chunks = torch.chunk(visual_tokens, self.image_head_num_chunks, dim=0)
        per_chunk_logits = []  # 每项是 tuple(Tensor) 长度 = len(self.heads)
        for cx, cvt in zip(x_chunks, vt_chunks):
            cl = torch.utils.checkpoint.checkpoint(
                self._run_after_unmerger,
                cx, cvt, visual_emb_layers, False,
                use_reentrant=False,
            )
            per_chunk_logits.append(cl)
        # 沿 dim=0 cat 回 [N_raw, V_i+1] x len(heads).
        n_heads = len(self.heads)
        logits = [torch.cat([per_chunk_logits[c][h] for c in range(len(per_chunk_logits))], dim=0)
                  for h in range(n_heads)]
        return logits
