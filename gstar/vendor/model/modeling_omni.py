"""PyTorch OmniForCausalLM - qwen3d5vit_qwen3_1d7b_pretrain

2 种 visual_model_type 路由：
  A. qwen3_5     : Qwen3_5VisualEncoder + Qwen3_5VisualBridge（连续）
  B. qwen3_5_vq  : Qwen3_5VisualEncoder + Qwen3_5VisualBridge(frozen) + RVQ
                   -> visual_bridge_model=VisualEmbeddingBridge（离散）
"""
import os
import time
import json
import math
import numpy as np
from typing import List, Optional, Tuple, Union, Any
from threading import Thread
from dataclasses import dataclass

import torch
import torch.distributed
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.nn import functional as F
import torch.distributed as dist
from transformers import PreTrainedModel
from transformers.generation import GenerationMixin
from transformers.activations import ACT2FN
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast, ModelOutput
from transformers.generation.utils import GenerationConfig
from transformers.utils import logging
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2RMSNorm
from .matcha_components import (
    SinusoidalPosEmb,
    Block1D,
    ResnetBlock1D,
    Downsample1D,
    TimestepEmbedding,
    Upsample1D,
)
from .matcha_transformer import BasicTransformerBlock
from .flow_matching import ConditionalDecoder, ConditionalCFM

from .configuration_omni import OmniConfig
from .audio_modeling_omni import (
    RMSNorm,
    OmniAudioEncoder,
    OmniAudioDecoder,
    OmniAudioVQBridgeTokenizer,
    OmniAudioFlowMatchingDecoder,
)
from .image_head import OmniImageHead
from .embedding_bridge import VisualEmbeddingBridge
from .common import FlashVarLenAttention
from .processor_omni import OmniMMProcessor
# 下面两个是 sidecar 占位 import：trust_remote_code 只拷贝 modeling_omni.py 的「顶层」
# relative import（不递归），故被 image_head/embedding_bridge/audio_modeling_omni 二级
# 依赖的 spatial_merger / vector_quantize 必须在此直接 import，否则加载时 FileNotFound。
from .spatial_merger import SpatialMerger, SpatialUnmerger, block_major_reorder_per_image  # noqa: F401
from .vector_quantize import VectorQuantize  # noqa: F401

# support model path contain point(.)
try:
    # step1: copy relative imports to transformers_modules
    from .generation_utils import build_chat_input, TextIterStreamer
    from .sequence_parallel_utils import (
        create_attention_layer,
        get_sequence_parallel_size,
        get_sequence_parallel_chunk,
    )
except ModuleNotFoundError:
    # step2: direct import from transformers_modules
    try:  # bypass check_imports failure
        import sys
        sys.path.append(os.path.dirname(__file__))
        from generation_utils import build_chat_input, TextIterStreamer
        from sequence_parallel_utils import (
            create_attention_layer,
            get_sequence_parallel_size,
            get_sequence_parallel_chunk,
        )
    except Exception:
        raise

logger = logging.get_logger(__name__)


def _resolve_dropout_p(config, attr_name, env_name):
    """LlamaGen-style AR dropout probability, env-override-first.

    merged_weights/config.json is shared across jobs that reuse the same
    (tokenizer, backbone), so we can't bake a non-zero dropout into it without
    affecting the baseline. The regularized imagenet256_c2i job therefore sets
    OMNI_{RESID,FFN,TOKEN}_DROPOUT in its launcher; everything else leaves the
    env unset and the config field at its 0.0 default -> no dropout.
    """
    env_val = os.environ.get(env_name)
    if env_val is not None and env_val != "":
        try:
            return float(env_val)
        except ValueError:
            pass
    return float(getattr(config, attr_name, 0.0) or 0.0)


def get_slopes(n):
    def get_slopes_power_of_2(n):
        start = (2 ** (-2 ** -(math.log2(n) - 3)))
        ratio = start
        return [start * ratio ** i for i in range(n)]

    if math.log2(n).is_integer():
        return get_slopes_power_of_2(
            n)  # In the paper, we only train models that have 2^a heads for some a. This function has
    else:  # some good properties that only occur when the input is a power of 2. To maintain that even
        closest_power_of_2 = 2 ** math.floor(
            math.log2(n))  # when the number of heads is not a power of 2, we use this workaround.
        return get_slopes_power_of_2(closest_power_of_2) + get_slopes(2 * closest_power_of_2)[0::2][
                                                           :n - closest_power_of_2]


class RotaryEmbedding(torch.nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=5e6, device=None):
        super().__init__()
        # 修复RePE初始化精度问题 https://zhuanlan.zhihu.com/p/678963442
        # DeepSpeed 会 Hack torch.arange 强制在 GPU 上运行，这里使用原生的 torch.arange
        try:
            import deepspeed
            self.arange = deepspeed.runtime.zero.partition_parameters._orig_torch_arange
        except:
            self.arange = torch.arange

        self.inv_freq = 1.0 / (base ** (self.arange(0, dim, 2).float().to(device) / dim))
        self.max_seq_len_cached = max_position_embeddings
        t = self.arange(self.max_seq_len_cached, device=self.inv_freq.device, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.cos_cached = emb.cos()[None, None, :, :].to(torch.float32)
        self.sin_cached = emb.sin()[None, None, :, :].to(torch.float32)

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        # This `if` block is unlikely to be run after we build sin/cos in `__init__`. Keep the logic here just in case.
        if seq_len > self.max_seq_len_cached:
            self.max_seq_len_cached = seq_len
            t = self.arange(self.max_seq_len_cached, device=self.inv_freq.device, dtype=torch.float32)
            freqs = torch.outer(t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1)
            self.cos_cached = emb.cos()[None, None, :, :].to(torch.float32).to(x.device)
            self.sin_cached = emb.sin()[None, None, :, :].to(torch.float32).to(x.device)
        return (
            self.cos_cached[:, :, :seq_len, ...].to(torch.float32).to(x.device),
            self.sin_cached[:, :, :seq_len, ...].to(torch.float32).to(x.device),
        )


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos_, sin_, position_ids):
    cos = cos_.squeeze(1).squeeze(0)  # [seq_len, dim]
    sin = sin_.squeeze(1).squeeze(0)  # [seq_len, dim]
    cos = cos[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    sin = sin[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    q_embed = (q.float() * cos) + (rotate_half(q.float()) * sin)
    k_embed = (k.float() * cos) + (rotate_half(k.float()) * sin)
    return q_embed.to(q.dtype), k_embed.to(k.dtype)


class MLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        ffn_dropout_p: float = 0.0,
    ):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.act_fn = ACT2FN[hidden_act]
        self.ffn_dropout = nn.Dropout(ffn_dropout_p) if ffn_dropout_p > 0 else nn.Identity()

    def forward(self, x):
        return self.ffn_dropout(self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x)))

# Copied from transformers.models.llama.modeling_llama.repeat_kv
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class Attention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""
    def __init__(self, config: OmniConfig, is_sparse=False):
        super().__init__()
        self.config = config
        self.position_embedding_type = config.position_embedding_type.lower()
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.hidden_size = config.num_attention_heads * self.head_dim
        self.hidden_kv_size = self.num_kv_heads * self.head_dim

        if is_sparse:
            self.num_heads = config.sparse_attention_heads
            assert self.num_kv_heads == config.num_attention_heads
            self.W_pack = nn.Linear(self.hidden_size, 3 * self.num_heads * self.head_dim, bias=config.attention_qkv_bias)
            self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=config.attention_qkv_bias)
        else:
            self.num_heads = config.num_attention_heads
            if self.config.attention_qkv_pack:
                self.W_pack = nn.Linear(config.hidden_size, self.hidden_size + self.hidden_kv_size * 2, bias=config.attention_qkv_bias)
            else:
                self.q_proj = nn.Linear(config.hidden_size, self.hidden_size, bias=config.attention_qkv_bias)
                self.k_proj = nn.Linear(config.hidden_size, self.hidden_kv_size, bias=config.attention_qkv_bias)
                self.v_proj = nn.Linear(config.hidden_size, self.hidden_kv_size, bias=config.attention_qkv_bias)
                # Qwen3 加了qk_norm
                self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)  # unlike olmo, only on the head dim!
                self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)  # thus post q_norm does not need reshape

            self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=config.attention_qkv_bias)

        if self.position_embedding_type == 'rope':
            self.rotary_emb = RotaryEmbedding(
                dim=self.head_dim,
                max_position_embeddings=config.max_position_embeddings,
                base=config.get_rotary_base()
            )
        elif self.position_embedding_type == 'alibi':
            self.alibi_slopes = get_slopes(self.num_heads)
        self.attention = create_attention_layer(self.hidden_size, self.num_heads, self.head_dim)
        # LlamaGen-style residual dropout on attention output. Default 0.0
        # (nn.Identity) keeps all non-c2i jobs unchanged.
        _resid_p = _resolve_dropout_p(config, 'resid_dropout_p', 'OMNI_RESID_DROPOUT')
        self.resid_dropout = nn.Dropout(_resid_p) if _resid_p > 0 else nn.Identity()

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def _repeat_kv(self, hidden_states: torch.Tensor, num_heads: int) -> torch.Tensor:
        assert hidden_states.size(1) <= num_heads and num_heads % hidden_states.size(1) == 0
        return repeat_kv(hidden_states, num_heads // hidden_states.size(1))

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        seqlens: Optional[torch.IntTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        attention_metadata=None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len = hidden_states.shape[:2]

        if self.config.attention_qkv_pack:
            proj = self.W_pack(hidden_states)
            query_states, key_states, value_states = proj.split([self.hidden_size, self.hidden_kv_size, self.hidden_kv_size], dim=-1)
        else:
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

        # (B, S, hidden_size) -> (B, num_heads, S, head_size)
        query_states = self.q_norm(query_states.view(bsz, q_len, -1, self.head_dim)).transpose(1, 2)
        # (B, S, hidden_size) -> (B, num_kv_heads, S, head_size)
        key_states = self.k_norm(key_states.view(bsz, q_len, -1, self.head_dim)).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]
        if self.position_embedding_type == 'rope':
            max_position = (attention_metadata[0] if attention_metadata is not None else
                            (position_ids.max().item()+1 if position_ids is not None else kv_seq_len * get_sequence_parallel_size()))
            cos, sin = self.rotary_emb(value_states, seq_len=max_position)
            query_states, key_states = apply_rotary_pos_emb(
                query_states, key_states, cos, sin,
                get_sequence_parallel_chunk(position_ids)
            )

        if past_key_value is not None:
            # reuse k, v, self_attention
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
        past_key_value = (key_states, value_states) if use_cache else None

        # repeat k/v heads if n_kv_heads < n_heads
        key_states = self._repeat_kv(key_states, query_states.size(1))
        value_states = self._repeat_kv(value_states, query_states.size(1))

        if seqlens is not None:
            seqlens = seqlens.to(dtype=torch.int32)
            max_seqlen = (attention_metadata[1] if attention_metadata is not None else
                          (seqlens[1:] - seqlens[:-1]).max().item())
            if self.position_embedding_type == 'alibi':
                alibi_slopes = torch.tensor(self.alibi_slopes, dtype=torch.float32).to(query_states.device)
            else:
                alibi_slopes = None
            attn_output = self.attention(
                query_states, key_states, value_states, seqlens, seqlens,
                max_seqlen, max_seqlen, causal=True, alibi_slopes=alibi_slopes, use_flash=True)
        else:
            attn_output = self.attention(
                query_states, key_states, value_states, attn_mask=attention_mask, use_flash=False)

        attn_output = attn_output.reshape(bsz, q_len, -1)
        attn_output = self.o_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        return attn_output, None, past_key_value


class DecoderLayer(nn.Module):
    def __init__(self, config: OmniConfig, is_sparse=False):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = Attention(config=config, is_sparse=is_sparse)
        self.mlp = MLP(
            hidden_size=self.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            ffn_dropout_p=_resolve_dropout_p(config, 'ffn_dropout_p', 'OMNI_FFN_DROPOUT'),
        )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        seqlens: Optional[torch.IntTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        group_index=None,
        attention_metadata=None,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            seqlens=seqlens,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            attention_metadata=attention_metadata,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


class OmniPreTrainedModel(PreTrainedModel):
    config_class = OmniConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["DecoderLayer"]
    _keys_to_ignore_on_load_unexpected = [r"decoder\.version"]
    _supports_flash_attn_2 = True

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear) or isinstance(module, nn.Conv1d) or isinstance(module, nn.ConvTranspose1d):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm) or isinstance(module, nn.GroupNorm):
            module.weight.data.fill_(1.0)
            module.bias.data.zero_()
        elif isinstance(module, RMSNorm) or isinstance(module, Qwen2RMSNorm):
            module.weight.data.fill_(1.0)
        elif isinstance(module, nn.Parameter):
            module.data.fill_(0.0)

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, OmniModel):
            module.gradient_checkpointing = value

@dataclass
class OmniModelOutputWithPast(BaseModelOutputWithPast):
    audio_encoder_ret: Optional[Any] = None
    audio_decoder_ret: Optional[Any] = None
    visual_indices:    Optional[Any] = None

class OmniModel(OmniPreTrainedModel):
    def __init__(self, config: OmniConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        special_token_ids = getattr(config, "multimodal_special_token_list", []) or []
        self._trainable_embed_token_ids = sorted({
            int(token_id) for token_id in special_token_ids if 0 <= int(token_id) < self.vocab_size
        })

        if config.visual_config.enable or (config.video_config.enable and not config.visual_config.enable):
            _vmt = getattr(config.visual_config, "visual_model_type", "qwen3_5")
            # 视觉 tokenizer 已解耦为独立的 trust_remote_code 模型：LLM 不再内置
            # encoder/quantizer。训练/推理时由外部 tokenizer.encode 产出 indices，经
            # vision_tokens 喂入；LLM 这边只保留 indices→embedding 的 VisualEmbeddingBridge。
            # 几何 (codebook_sizes/depth/dim) 仍读自 config.visual_quantizer_config，
            # 使本模型可独立 from_pretrained（含旧 merged ckpt）建出 bridge/head。
            if _vmt in ("qwen3_5_vq", "qwen2_5_vq"):
                if getattr(config, 'enable_visual_embedding_bridge', True):
                    self.visual_bridge_model = VisualEmbeddingBridge(config)
                else:
                    self.visual_bridge_model = None
            else:
                raise ValueError(
                    f"visual_model_type={_vmt!r} 的连续视觉编码路径已移除；视觉 tokenizer 现为"
                    f"独立模型，请用 'qwen3_5_vq'/'qwen2_5_vq' 离散路径并通过 vision_tokens 喂入。"
                )

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        # LlamaGen-style token dropout applied to the (text+visual) input
        # embeddings. Default 0.0 (nn.Identity) -> no change for non-c2i jobs.
        _tok_p = _resolve_dropout_p(config, 'token_dropout_p', 'OMNI_TOKEN_DROPOUT')
        self.tok_dropout = nn.Dropout(_tok_p) if _tok_p > 0 else nn.Identity()
        self.layers = nn.ModuleList([
            DecoderLayer(config, is_sparse=layer_idx in config.sparse_attention_layers)
            for layer_idx in range(config.num_hidden_layers)
        ])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        if config.audio_config.enable:
            self.audio_embed_layers = nn.ModuleList([
                nn.Embedding(codedim + 1, config.hidden_size)
                    for i, codedim in enumerate(config.audio_config.vq_config.codebook_sizes)
            ])
        self._rar_random_ratio = 0.0

        self.gradient_checkpointing = True
        # Initialize weights and apply final processing
        self.post_init()

    def _set_gradient_checkpointing(self, value=False):
        self.gradient_checkpointing = value

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    @property
    def main_device(self):
        return self.embed_tokens.weight.device

    @torch.no_grad()
    def get_multimodal_mask(self, input_ids, pad_token_id, special_token_list):
        '''
        获取任意模态的特殊mask，包含以下
        1. pad mask 表示文本中图像/语音/视频模态提前留出的token位置
        2. special token mask 特殊token 例如对理解模型<start> <end> 不需要next token prediction
        3. embedding mask / lm_head mask 标记出特殊token在embedding中的mask
        '''
        pad_mask = torch.eq(input_ids, pad_token_id)
        sp_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        lm_head_mask = torch.zeros([self.config.vocab_size, 1], dtype=torch.bool)
        for sp_id in special_token_list:
            sp_mask = torch.logical_or(sp_mask, torch.eq(input_ids, sp_id))
            lm_head_mask[sp_id, 0] = True
        return pad_mask, sp_mask, lm_head_mask

    def get_multimodal_embed(
            self, 
            input_ids,
            text_embedding,  # 1. self.embed_tokens(input_ids) 2. 其他模态结果
            multimodal_embed,
            pad_token_id,
            fake_input,
            group_index=None,  # 某种模态的编号 for MoE
        ):
        pad_mask, sp_mask, _ = self.get_multimodal_mask(input_ids, pad_token_id, self.config.multimodal_special_token_list)
        if not self.training:  # 推理支持auto map 把多模态模块输出和input_ids 统一到一个device
            multimodal_embed = multimodal_embed.to(input_ids.device)
        if not fake_input:  # 检查多模态token 和 pad mask数量一致 （不正确的截断会导致该问题）
            assert pad_mask.sum() == multimodal_embed.shape[0], f"{pad_mask.sum()} vs {multimodal_embed.shape[0]}"
        else:
            assert pad_mask.sum() <= 0  # 0 vs 1

        # 合并 当前模态embeddings 和text embeddings
        input_ids = torch.where(pad_mask, torch.cumsum(pad_mask.view(-1).to(input_ids), dim=0).view(input_ids.shape)-1, input_ids)
        if self.config.train_multimodal_special_tokens_only and self.training:  
            # 仅special token传梯度到embedding weight, 保证LLM部分不变
            # 注意: 多种模态之间special token list应该共享，否则会有部分被stop gradient
            sp_mask = sp_mask.unsqueeze(-1).to(text_embedding)
            text_embedding = (1 - sp_mask) * text_embedding.detach() + sp_mask * text_embedding 
        text_embedding = (1 - pad_mask.to(text_embedding)).unsqueeze(-1) * text_embedding  # pad token位置填0 (不传梯度)
        multimodal_embedding = torch.embedding(multimodal_embed, input_ids * pad_mask)  # 非 pad token 位置填idx=0位置结果
        multimodal_embedding = pad_mask.to(multimodal_embedding).unsqueeze(-1) * multimodal_embedding  # 非pad token 位置填0

        final_embedding = multimodal_embedding.to(text_embedding) + text_embedding

        if group_index is None:
            group_index = pad_mask.to(torch.int32)
        else:
            current_index = torch.max(group_index) + 1
            group_index += pad_mask.to(torch.int32) * current_index  # 假设模态无重叠

        return final_embedding, group_index  # group_index 不传None 防止MoE部分参数无梯度

    def set_rar_random_ratio(self, new_ratio):
        """Set the per-image probability of using a random generation order."""
        new_ratio = float(new_ratio)
        if not 0.0 <= new_ratio <= 1.0:
            raise ValueError(f"RAR random ratio must be in [0, 1], got {new_ratio}")
        self._rar_random_ratio = new_ratio

    def _prepare_rar_inputs(
            self, input_ids, vision_tokens, rar_generation_step=None):
        """Permute full training images and build source/next-target positions.

        Training batches contain packed fixed-resolution images, so the fast
        path reshapes all image-pad tokens as [num_images, spatial_tokens].
        Generation is raster-only and may contain a partial image.  With a KV
        cache, ``prepare_inputs_for_generation`` slices away the original
        image-start token, so the incremental path uses the explicit raster
        step supplied by the generation loop.
        """
        bridge = self.visual_bridge_model
        if bridge is None or not getattr(bridge, "rar_random_order", False):
            return vision_tokens, None, None, None

        num_spatial = bridge.rar_num_spatial_tokens
        pad_id = self.config.visual_config.image_pad_token_id
        start_id = self.config.visual_config.image_start_token_id
        flat_pad_positions = torch.nonzero(
            input_ids.reshape(-1).eq(pad_id), as_tuple=False).flatten()
        flat_start_positions = torch.nonzero(
            input_ids.reshape(-1).eq(start_id), as_tuple=False).flatten()

        if self.training and vision_tokens is not None and vision_tokens.numel() > 0:
            num_pads = int(flat_pad_positions.numel())
            if num_pads != int(vision_tokens.shape[0]):
                raise RuntimeError(
                    f"RAR vision-token mismatch: pads={num_pads}, "
                    f"vision_tokens={vision_tokens.shape[0]}; sequence parallel "
                    "is not supported by the packed RAR path")
            if num_pads % num_spatial != 0:
                raise RuntimeError(
                    f"RAR expects {num_spatial} spatial tokens/image, got "
                    f"{num_pads} pad tokens")
            num_images = num_pads // num_spatial
            if int(flat_start_positions.numel()) != num_images:
                raise RuntimeError(
                    f"RAR expects one image_start per image: starts="
                    f"{flat_start_positions.numel()}, images={num_images}")

            orders = torch.arange(num_spatial, device=vision_tokens.device).expand(
                num_images, -1).clone()
            if self._rar_random_ratio > 0.0 and num_images > 0:
                use_random = torch.rand(num_images, device=vision_tokens.device) \
                    < self._rar_random_ratio
                random_orders = torch.rand(
                    num_images, num_spatial, device=vision_tokens.device
                ).argsort(dim=-1)
                orders = torch.where(use_random.unsqueeze(1), random_orders, orders)

            depth = vision_tokens.shape[-1]
            vt = vision_tokens.view(num_images, num_spatial, depth)
            vt = torch.gather(
                vt, 1, orders.unsqueeze(-1).expand(-1, -1, depth)
            ).reshape(-1, depth)
            source_ids = orders.reshape(-1)

            # condition/image_start predicts order[:, 0]; token j predicts
            # order[:, j+1].  The last token predicts EOS and gets no target
            # position, matching the official RAR target-aware embedding.
            pad_by_image = flat_pad_positions.view(num_images, num_spatial)
            target_positions = torch.cat([
                flat_start_positions,
                pad_by_image[:, :-1].reshape(-1),
            ])
            target_ids = torch.cat([
                orders[:, 0],
                orders[:, 1:].reshape(-1),
            ])
            return vt, source_ids, target_positions, target_ids

        # Raster generation, including the first step where vision_tokens is
        # None and only image_start is present.  Parse each image independently
        # because its generated prefix can be shorter than num_spatial.
        if self.training:
            return vision_tokens, None, None, None

        # Incremental KV-cache generation sees only the newly appended
        # image-pad token; the image-start and prior pads live in the cache.
        # ``vision_tokens`` is exactly the code sampled at step-1.  Give it the
        # matching source position and make the current hidden state predict
        # raster position ``step``.  Without this branch the span parser sees
        # no image-start, consumes zero codes, and every RAR eval fails after
        # its first generated token.
        if (rar_generation_step is not None
                and int(rar_generation_step) > 0
                and int(flat_start_positions.numel()) == 0
                and vision_tokens is not None):
            step = int(rar_generation_step)
            if step >= num_spatial:
                raise RuntimeError(
                    f"RAR generation step {step} exceeds {num_spatial} positions")
            num_pads = int(flat_pad_positions.numel())
            if num_pads != int(vision_tokens.shape[0]):
                raise RuntimeError(
                    f"RAR incremental generation mismatch: pads={num_pads}, "
                    f"vision_tokens={vision_tokens.shape[0]}, step={step}")
            pads_per_row = input_ids.eq(pad_id).sum(dim=1)
            if not bool(torch.all(pads_per_row.eq(1))):
                raise RuntimeError(
                    "RAR incremental generation expects one image-pad per row, "
                    f"got {pads_per_row.tolist()} at step {step}")
            source_ids = torch.full(
                (num_pads,), step - 1, device=input_ids.device, dtype=torch.long)
            target_ids = torch.full(
                (num_pads,), step, device=input_ids.device, dtype=torch.long)
            return vision_tokens, source_ids, flat_pad_positions, target_ids

        # FID generation has one equally long image prefix per batch row.  Keep
        # that hot path entirely on-device: the Python fallback below would
        # otherwise introduce a CUDA synchronization for every generated token.
        num_images = int(flat_start_positions.numel())
        num_pads = int(flat_pad_positions.numel())
        if num_images and num_pads % num_images == 0:
            count = num_pads // num_images
            if count > num_spatial:
                raise RuntimeError(
                    f"RAR generation has {count}>{num_spatial} image tokens")
            if vision_tokens is not None and int(vision_tokens.shape[0]) != num_pads:
                raise RuntimeError(
                    f"RAR generation vision-token mismatch: pads={num_pads}, "
                    f"vision_tokens={vision_tokens.shape[0]}")
            pad_by_image = flat_pad_positions.view(num_images, count) \
                if count else None
            uniform_contiguous = count == 0 or torch.equal(
                pad_by_image[:, 0], flat_start_positions + 1)
            if uniform_contiguous:
                source_ids = torch.arange(
                    count, device=input_ids.device, dtype=torch.long
                ).expand(num_images, -1).reshape(-1) if count else None
                if count < num_spatial:
                    predicted_pad_positions = flat_pad_positions
                    predicted_pad_ids = torch.arange(
                        1, count + 1, device=input_ids.device, dtype=torch.long
                    ).expand(num_images, -1).reshape(-1)
                else:
                    predicted_pad_positions = pad_by_image[:, :-1].reshape(-1)
                    predicted_pad_ids = torch.arange(
                        1, num_spatial, device=input_ids.device, dtype=torch.long
                    ).expand(num_images, -1).reshape(-1)
                target_positions = torch.cat([
                    flat_start_positions, predicted_pad_positions])
                target_ids = torch.cat([
                    torch.zeros(num_images, device=input_ids.device, dtype=torch.long),
                    predicted_pad_ids,
                ])
                return vision_tokens, source_ids, target_positions, target_ids

        ordered_chunks, source_chunks = [], []
        target_positions, target_ids = [], []
        offset = 0
        batch, seq_len = input_ids.shape
        for batch_idx in range(batch):
            starts = torch.nonzero(
                input_ids[batch_idx].eq(start_id), as_tuple=False
            ).flatten().tolist()
            for start in starts:
                pad_positions = []
                pos = start + 1
                while pos < seq_len and int(input_ids[batch_idx, pos]) == pad_id:
                    pad_positions.append(pos)
                    pos += 1
                count = len(pad_positions)
                if count > num_spatial:
                    raise RuntimeError(
                        f"RAR generation has {count}>{num_spatial} image tokens")
                if count:
                    if vision_tokens is None or offset + count > vision_tokens.shape[0]:
                        raise RuntimeError("RAR generation vision-token prefix is incomplete")
                    ordered_chunks.append(vision_tokens[offset:offset + count])
                    source_chunks.append(torch.arange(
                        count, device=vision_tokens.device, dtype=torch.long))
                    offset += count
                if count < num_spatial:
                    flat_start = batch_idx * seq_len + start
                    target_positions.append(flat_start)
                    target_ids.append(0)
                    for token_idx, pad_pos in enumerate(pad_positions):
                        target_positions.append(batch_idx * seq_len + pad_pos)
                        target_ids.append(token_idx + 1)
                else:
                    target_positions.append(batch_idx * seq_len + start)
                    target_ids.append(0)
                    for token_idx, pad_pos in enumerate(pad_positions[:-1]):
                        target_positions.append(batch_idx * seq_len + pad_pos)
                        target_ids.append(token_idx + 1)
        if vision_tokens is not None and offset != int(vision_tokens.shape[0]):
            raise RuntimeError(
                f"RAR consumed {offset} generation tokens, got {vision_tokens.shape[0]}")
        device = input_ids.device
        source_ids = torch.cat(source_chunks) if source_chunks else None
        target_positions = torch.tensor(target_positions, device=device, dtype=torch.long) \
            if target_positions else None
        target_ids = torch.tensor(target_ids, device=device, dtype=torch.long) \
            if target_ids else None
        ordered = torch.cat(ordered_chunks) if ordered_chunks else vision_tokens
        return ordered, source_ids, target_positions, target_ids

    def _add_rar_target_positions(self, embeddings, positions, target_ids):
        if positions is None:
            return embeddings
        target = self.visual_bridge_model.rar_target_pos_embed(target_ids).to(embeddings)
        shape = embeddings.shape
        flat = embeddings.reshape(-1, shape[-1]).index_add(0, positions, target)
        return flat.view(shape)

    def get_visual_embed_given_tokens(
            self,
            input_ids,
            text_embedding,  # 1. self.embed_tokens(input_ids) 2. 其他模态结果
            vision_tokens=None,
            rar_generation_step=None,
            group_index=None,   # 某种模态的编号
        ):
        """消费外部已编码好的离散 indices（vision_tokens）。视觉 tokenizer 已解耦，
        encode + block-major 重排都在 LLM 外完成（cached 路径由离线脚本 reorder 后写盘，
        on-the-fly 路径由训练 glue reorder），此处只做 indices→embedding。

        vision_tokens 为 None：无图，跑一遍 fake bridge forward 让 visual_bridge_model
        参数留在 DeepSpeed zero3 计算图内（替代原 get_visual_embed 的 fake 分支）。

        返回的 vision_tokens 同时作为 image_head 的 CE 目标（已是 block-major 顺序，
        与 image_head 输入一一对应）。
        """
        assert self.visual_bridge_model is not None, \
            "VisualEmbeddingBridge is disabled (enable_visual_embedding_bridge=false). Cannot use vision_tokens without it."

        vision_tokens, rar_source_ids, rar_target_positions, rar_target_ids = \
            self._prepare_rar_inputs(
                input_ids, vision_tokens,
                rar_generation_step=rar_generation_step)

        if vision_tokens is None or vision_tokens.numel() == 0:
            # zero3 占位：无真实图像，喂最小 fake indices 保持 bridge 参数有梯度。
            # SpatialMerger 启用时，一个 LLM-side fake token 需要 mms² 个
            # level-1 indices；仍然只产出一个 bridge embedding。
            num_levels = len(self.config.visual_quantizer_config.codebook_sizes)
            mms = int(getattr(self.config.visual_config, "merger_merge_size", 1) or 1)
            fake_indices = torch.zeros((mms * mms, num_levels), dtype=torch.long, device=input_ids.device)
            vision_embed = self.visual_bridge_model(fake_indices)
            final_embedding, group_index = self.get_multimodal_embed(
                input_ids, text_embedding, vision_embed,
                self.config.visual_config.image_pad_token_id, True, group_index=group_index)
            final_embedding = self._add_rar_target_positions(
                final_embedding, rar_target_positions, rar_target_ids)
            return final_embedding, group_index, None

        vision_embed = self.visual_bridge_model(vision_tokens)
        if rar_source_ids is not None:
            vision_embed = vision_embed + self.visual_bridge_model.rar_source_pos_embed(
                rar_source_ids).to(vision_embed)
        # 目前还不支持video
        final_embedding, group_index = self.get_multimodal_embed(
            input_ids, text_embedding, vision_embed,
            self.config.visual_config.image_pad_token_id, False, group_index=group_index)
        final_embedding = self._add_rar_target_positions(
            final_embedding, rar_target_positions, rar_target_ids)
        return final_embedding, group_index, vision_tokens

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        seqlens: Optional[torch.IntTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        audios_tokens: Union[List, torch.Tensor, None] = None, # 音频token bs*seqlen*vq_num
        audiotext_ids: Union[List, torch.LongTensor, None] = None,
        vision_tokens: Union[List, torch.Tensor, None] = None, # 视觉token bs*seqlen*vq_num
        rar_generation_step: Optional[int] = None,
        images: Union[List, torch.Tensor, None] = None,
        patch_nums: Optional[torch.Tensor] = None, 
        images_grid: Union[List, torch.Tensor, None] = None,
        videos: Union[List, torch.Tensor, None] = None,
        videos_patch_nums: Optional[torch.Tensor] = None, 
        videos_grid: Union[List, torch.Tensor, None] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        batch_attention_metadata=None,
    ) -> Union[Tuple, OmniModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = True if (return_dict is not None or self.training) else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        seq_length_with_past = seq_length
        past_key_values_length = 0

        if past_key_values is not None:
            past_key_values_length = past_key_values[0][0].shape[2]
            seq_length_with_past = seq_length_with_past + past_key_values_length

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        else:
            position_ids = position_ids.view(-1, seq_length).long()

        group_index = None
        if inputs_embeds is None:
            sp_input_ids = get_sequence_parallel_chunk(input_ids)
            inputs_embeds = self.embed_tokens(sp_input_ids)
            if self.config.audio_config.enable:
                if audios_tokens is None or len(audios_tokens) <= 0 :
                    audios_tokens = torch.zeros(5, len(self.config.audio_config.vq_config.codebook_sizes), dtype=torch.int32, device=input_ids.device)  # a fake input
                    fake_input = True
                else:
                    fake_input = False
                for i, audio_emb_layer in enumerate(self.audio_embed_layers):
                    if i==0:
                        audio_embs = audio_emb_layer(audios_tokens[..., i]) 
                    else:
                        audio_embs += audio_emb_layer(audios_tokens[..., i]) 
                inputs_embeds, group_index = self.get_multimodal_embed(sp_input_ids, inputs_embeds, audio_embs, self.config.audio_config.audio_pad_token_id, fake_input, group_index=group_index)
                if audiotext_ids is not None:
                    audiotext_embs = self.embed_tokens(audiotext_ids)
                else:
                    audiotext_ids = torch.zeros(5, dtype=sp_input_ids.dtype, device=sp_input_ids.device)
                    audiotext_embs = self.embed_tokens(audiotext_ids) # a fake input for zero3
                # audiotext_ids中操作audiotext_embedding, pad_token位置置0
                audiotext_embs = audiotext_embs * (audiotext_ids != self.config.audio_config.audiotext_pad_token_id).unsqueeze(-1).to(audiotext_embs.dtype)
                # audiotext在完整input_ids中的位置，包括了audiotext_pad_token, audiotext_start_token, audio_pad_token
                audiotext_pad_mask = torch.logical_or(torch.logical_or(sp_input_ids == self.config.audio_config.audiotext_pad_token_id, sp_input_ids == self.config.audio_config.audiotext_start_token_id), sp_input_ids == self.config.audio_config.audio_pad_token_id)
                # audiotext在完整input_ids中的delay token位置，即audiotext_pad_token
                audiotext_delay_mask = sp_input_ids == self.config.audio_config.audiotext_pad_token_id
                # 去掉input_embeds中delay token位置
                inputs_embeds = (1 - audiotext_delay_mask.to(inputs_embeds)).unsqueeze(-1) * inputs_embeds
                # 把audiotext_embedding填充到input_ids长度，并加到input_embeds中
                audiotext_input_ids = torch.where(audiotext_pad_mask, torch.cumsum(audiotext_pad_mask.view(-1).to(sp_input_ids), dim=0).view(sp_input_ids.shape)-1, sp_input_ids)
                audiotext_embeds = torch.embedding(audiotext_embs, audiotext_input_ids * audiotext_pad_mask)
                audiotext_embeds = audiotext_pad_mask.to(audiotext_embeds).unsqueeze(-1) * audiotext_embeds
                inputs_embeds = inputs_embeds + audiotext_embeds.to(inputs_embeds)

            if (self.config.visual_config.enable or self.config.video_config.enable):
                # 视觉 tokenizer 已解耦：vision_tokens 由外部（cached 读盘 / on-the-fly glue）
                # encode+reorder 后传入，此处不再 encode/reorder。
                # vision_tokens 为 None 时走 zero3 fake 占位（见 get_visual_embed_given_tokens）。
                inputs_embeds, group_index, visual_indices = \
                    self.get_visual_embed_given_tokens(
                        sp_input_ids,
                        inputs_embeds,
                        vision_tokens,
                        rar_generation_step=rar_generation_step,
                        group_index=group_index)  # 注意更新group index
            else:
                visual_indices = None

        if seqlens is not None and seqlens.ndim == 2:
            cu_seqlens = []
            offset, seqlen = 0, seqlens.size(1)
            for lens in seqlens:
                cu_seqlens.append(offset)
                # Dataset stores positive sample LENGTHS, then zeros; FlashAttention
                # requires cumulative endpoints. Padding is its own isolated segment.
                lengths = lens[lens > 0]
                ends = lengths.cumsum(0)
                if bool((lens < 0).any()) or (ends.numel() and int(ends[-1]) > seqlen):
                    raise ValueError("invalid packed sample lengths")
                cu_seqlens.extend((ends[ends < seqlen] + offset).tolist())
                offset += seqlen
            cu_seqlens.append(offset)
            seqlens = torch.tensor(cu_seqlens, dtype=seqlens.dtype, device=seqlens.device)
        elif seqlens is None and self.training:
            seqlens = torch.arange(
                end=input_ids.size(0) + 1,
                dtype=torch.int32,
                device=input_ids.device
            ) * input_ids.size(1)
        if seqlens is not None:
            attention_mask = None  # unset attention_mask to save memory

        if seqlens is None and attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, seq_length_with_past), dtype=torch.bool, device=inputs_embeds.device
            )
        if attention_mask is not None:
            attention_mask = _prepare_4d_causal_attention_mask(
                attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
            )

        # embed positions
        hidden_states = inputs_embeds
        hidden_states = self.tok_dropout(hidden_states)

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        # Batch-local immutable scalars: compute once, including when checkpoint
        # backward recomputes layers. Never cache across batches or on a module.
        attention_metadata = batch_attention_metadata
        if (attention_metadata is None and getattr(self.config, "attention_metadata_once", False)
                and self.training and seqlens is not None and position_ids is not None):
            attention_metadata = (
                position_ids.max().item() + 1,
                (seqlens.to(torch.int32)[1:] - seqlens.to(torch.int32)[:-1]).max().item(),
            )

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None
        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            past_key_value = past_key_values[idx] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # None for past_key_value
                        return module(*inputs, output_attentions, False, group_index, attention_metadata=attention_metadata)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(decoder_layer),
                    hidden_states,
                    attention_mask,
                    position_ids,
                    seqlens,
                    None,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    seqlens=seqlens,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    group_index=group_index,
                    attention_metadata=attention_metadata,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return OmniModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            visual_indices=visual_indices, 
        )


class NormHead(nn.Module):
    def __init__(self, hidden_size, vocab_size, bias=False):
        super().__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.weight = nn.Parameter(torch.empty((self.vocab_size, self.hidden_size)))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, hidden_states, mask=None):
        norm_weight = nn.functional.normalize(self.weight)
        if mask is not None:
            mask = mask.to(norm_weight)
            norm_weight = norm_weight * mask + (1 - mask) * norm_weight.detach()
        return nn.functional.linear(hidden_states, norm_weight)

    def extra_repr(self) -> str:
        return f'in_features={self.hidden_size}, out_features={self.vocab_size}'

@dataclass
class OmniMMCausalLMOutputWithPast(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    text_nt_loss: Optional[torch.FloatTensor] = None
    flatten_loss: Optional[torch.FloatTensor] = None
    audios_logits: Optional[List[torch.FloatTensor]] = None
    audios_loss_list:  Optional[List[torch.FloatTensor]] = None
    audios_eos_loss: Optional[torch.FloatTensor] = None
    audios_emb_rc_loss: Optional[torch.FloatTensor] = None # 音频embedding的重建 loss
    audios_emb_for_infer: Optional[torch.FloatTensor] = None # 用于audio head 推理的 embeddings
    visual_logits: Optional[List[torch.FloatTensor]] = None
    visual_eos_loss: Optional[torch.FloatTensor] = None
    visual_loss_list:  Optional[List[torch.FloatTensor]] = None
    vision_emb_for_infer: Optional[torch.FloatTensor] = None # 用于image_head 推理的 embeddings
    audiotext_loss: Optional[torch.FloatTensor] = None

class CasualDepthTransformerLayer(nn.Module):
    def __init__(self, config, depth):
        super().__init__()
        self.config = config
        self.depth = depth
        assert config.audio_config.audio_head_transformer_dims % 128 == 0
        assert config.audio_config.audio_head_transformer_dims % depth == 0
        num_heads = config.audio_config.audio_head_transformer_dims // 128

        # 在transformers==4.51.3版本时使用MultiheadAttention会出现NaN
        # self.self_attention = nn.MultiheadAttention(embed_dim=config.audio_config.audio_head_transformer_dims, num_heads=num_heads,batch_first=True)
        
        self.self_attention = FlashVarLenAttention(
            embed_dim=config.audio_config.audio_head_transformer_dims, 
            num_heads=num_heads, causal=True)
        
        self.layernorm1 = RMSNorm(config.audio_config.audio_head_transformer_dims)
        self.layernorm2 = RMSNorm(config.audio_config.audio_head_transformer_dims)
        if config.audio_config.audio_head_transformer_ffn_scale <= 0:  # 兼容baichuan-omni-1d5老逻辑
            self.linear1 = nn.Linear(config.audio_config.audio_head_transformer_dims * depth, 2 * config.audio_config.audio_head_transformer_dims)
            self.linear2 = nn.Linear(2 * config.audio_config.audio_head_transformer_dims * depth, config.audio_config.audio_head_transformer_dims)
        else:
            self.linear1 = nn.Linear(
                config.audio_config.audio_head_transformer_dims, 
                config.audio_config.audio_head_transformer_ffn_scale * config.audio_config.audio_head_transformer_dims
            )
            self.linear2 = nn.Linear(
                config.audio_config.audio_head_transformer_ffn_scale * config.audio_config.audio_head_transformer_dims,
                config.audio_config.audio_head_transformer_dims
            )

    def forward(self, x):
        seq_len = x.size(1)
        res = x
        x = self.layernorm1(x)
        # MultiheadAttention
        # _x, _ = self.self_attention(x, x, x,  is_causal=True, attn_mask=src_mask)
        # src_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool().to(x.device)
        
        _bsz, _transformer_dims = x.shape[0], self.self_attention.embed_dim
        seqlens = self.depth * torch.ones((_bsz,), dtype=torch.int32, device=x.device)
        _x = self.self_attention(x.view(-1, _transformer_dims), seqlens)
        _x = _x.view(_bsz, self.depth, _transformer_dims).contiguous()
        
        _res = _x + res  # (bs, sl, d)
        res = self.layernorm2(_res)
        if self.config.audio_config.audio_head_transformer_ffn_scale <= 0:  # 兼容baichuan-omni-1d5老逻辑
            x = torch.einsum('bld,tld->blt', res, torch.reshape(self.linear1.weight, (2 * self.config.hidden_size, -1, self.config.hidden_size)))
            x = torch.nn.functional.gelu(x)
            x = torch.einsum('blt,dlt->bld', x, torch.reshape(self.linear2.weight, (self.config.hidden_size, -1, 2 * self.config.hidden_size)))
        else:
            if self.depth > 1:
                x = torch.einsum('bld,tld->blt', 
                    res, 
                    torch.reshape(self.linear1.weight, 
                        (self.config.audio_config.audio_head_transformer_ffn_scale * self.config.audio_config.audio_head_transformer_dims // self.depth,
                        self.depth,
                        self.config.audio_config.audio_head_transformer_dims)
                    ))
            else:
                x = self.linear1(res)
            x = torch.nn.functional.gelu(x)
            if self.depth > 1:
                x = torch.einsum('blt,dlt->bld',
                    x, 
                    torch.reshape(self.linear2.weight, 
                        (self.config.audio_config.audio_head_transformer_dims,
                        self.depth,
                        self.config.audio_config.audio_head_transformer_ffn_scale * self.config.audio_config.audio_head_transformer_dims // self.depth)
                    ))
            else:
                x = self.linear2(x)
        return _res + x

class OmniAudioHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        if config.audio_config.audio_head_transformer_ffn_scale > 0:
            self.hidden_norm = RMSNorm(config.hidden_size)
            self.hidden_proj = nn.Linear(config.hidden_size, config.audio_config.audio_head_transformer_dims, bias=False)
        self.transformer_layers = nn.ModuleList([
            CasualDepthTransformerLayer(config, len(config.audio_config.vq_config.codebook_sizes)) 
            for _ in range(config.audio_config.audio_head_transformer_layers)
        ])
        self.headnorm = RMSNorm(config.audio_config.audio_head_transformer_dims) 
        self.heads = nn.ModuleList([
            nn.Linear(config.audio_config.audio_head_transformer_dims, vq_size+1)
            for vq_size in config.audio_config.vq_config.codebook_sizes
        ])
        self.gradient_checkpointing = True
        if not config.audio_config.enable:
            for param in self.parameters():  # 修复SFT阶段可能错误加载导致OOM的问题
                param.requires_grad = False

    def forward(self, x, audios_tokens, audio_emb_layers):
        cumsum_audio_embed = torch.stack([
            audio_emb_layers[i](audios_tokens[..., i]) 
            for i, vq_size in enumerate(self.config.audio_config.vq_config.codebook_sizes[:-1])
            ], dim=1)
        cumsum_audio_embed = torch.cumsum(cumsum_audio_embed, dim=1)  # (bs, depth-1, d)
        hidden_states = torch.concat([x.reshape(-1, 1, self.config.hidden_size), cumsum_audio_embed], dim=1)  # (bs, depth, d)
        assert hidden_states.size(1) == len(self.config.audio_config.vq_config.codebook_sizes)

        if self.config.audio_config.audio_head_transformer_ffn_scale > 0:
            hidden_states = self.hidden_norm(hidden_states)
            hidden_states = self.hidden_proj(hidden_states)

        for i, tlayer in enumerate(self.transformer_layers):
            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # None for past_key_value
                        return module(*inputs)

                    return custom_forward

                hidden_states  = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(tlayer), hidden_states,
                )
            else:
                hidden_states  = tlayer(
                    hidden_states,
                )
        hidden_states = self.headnorm(hidden_states)
        logits = [head(hidden_states[:,i]) for i,head in enumerate(self.heads)]
        return logits


class OmniForCausalLM(OmniPreTrainedModel, GenerationMixin):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.model = OmniModel(config)
        if config.audio_config.enable:
            self.audio_tokenizer = OmniAudioTokenizer(config)
            self.audio_head = OmniAudioHead(config)
        if config.visual_config.enable and config.visual_config.image_head_config.enable:
            self.image_head = OmniImageHead(self.config)
        if config.use_norm_head:
            self.lm_head = NormHead(config.hidden_size, config.vocab_size, bias=False)
        else:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        # Initialize weights and apply final processing
        self.post_init()

    @property
    def main_device(self):
        return self.lm_head.weight.device

    def bind_processor(self, tokenizer, **kwargs):
        self.processor = OmniMMProcessor(
                tokenizer=tokenizer,
                config=self.config,
                **kwargs,
                )
        return self.processor

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model
    
    def set_rar_random_ratio(self, new_ratio):
        self.model.set_rar_random_ratio(new_ratio)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        seqlens: Optional[torch.IntTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        audios: Union[List, torch.Tensor, None] = None,
        audios_tokens: Union[List, torch.Tensor, None] = None,
        audiotext_ids: Union[List, torch.LongTensor, None] = None,
        vision_tokens: Union[List, torch.Tensor, None] = None,
        rar_generation_step: Optional[int] = None,
        encoder_length: Optional[torch.Tensor] = None,
        bridge_length: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None, 
        patch_nums: Optional[torch.Tensor] = None, 
        images_grid: Optional[torch.Tensor] = None, 
        videos: Optional[torch.Tensor] = None, 
        videos_patch_nums: Optional[torch.Tensor] = None, 
        videos_grid: Optional[torch.Tensor] = None, 
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        batch_attention_metadata=None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if self.config.audio_config.enable:
            if audios_tokens is not None:
                assert isinstance(audios_tokens,torch.Tensor)
            else:
                if audios is None or len(audios) == 0:
                    audios_tokens = None
                    _ = self.audio_tokenizer(*self.audio_tokenizer.audio_model.fake_input(input_ids.device))
                else:
                    audios_tokens = self.audio_tokenizer(audios,encoder_length,bridge_length)
        else:
            audios_tokens = None
        
        if self.config.visual_config.enable:
            if vision_tokens is not None:
                assert isinstance(vision_tokens,torch.Tensor)
            else:
                pass
        else:
            vision_tokens = None

        _, sp_mask, _ = self.model.get_multimodal_mask(input_ids, self.config.audio_config.audio_pad_token_id, self.config.multimodal_special_token_list)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            seqlens=seqlens,
            batch_attention_metadata=batch_attention_metadata,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            audios_tokens=audios_tokens,
            audiotext_ids=audiotext_ids,
            vision_tokens=vision_tokens,
            rar_generation_step=rar_generation_step,
            images=images,
            patch_nums=patch_nums, 
            images_grid=images_grid,
            videos=videos,
            videos_patch_nums=videos_patch_nums, 
            videos_grid=videos_grid,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = outputs.last_hidden_state
        visual_tokens = outputs.visual_indices
        if self.config.visual_config.image_head_config.enable:
            if visual_tokens is not None:
                assert isinstance(visual_tokens,torch.Tensor)
        else:
            visual_tokens = None

        # 部分可学习的special token放开lm head梯度
        special_with_loss_list = list(set(self.config.multimodal_special_token_list) - set(self.config.multimodal_special_token_no_loss_list))
        _, sp_with_loss_mask, lm_head_mask = self.model.get_multimodal_mask(input_ids, self.config.audio_config.audio_pad_token_id, special_with_loss_list)
        if self.config.train_multimodal_special_tokens_only and self.training and len(special_with_loss_list) > 0:
            if self.config.use_norm_head:
                logits = self.lm_head(hidden_states, mask=lm_head_mask)
            else:
                lm_head_mask = lm_head_mask.to(self.lm_head.weight)
                norm_weight = self.lm_head.weight * lm_head_mask + (1 - lm_head_mask) * self.lm_head.weight.detach()
                logits = torch.einsum('bsd,ld->bsl', hidden_states, norm_weight)
        else:
            logits = self.lm_head(hidden_states)

        audios_logits = None
        audios_emb_for_infer = None
        audios_loss_list = [torch.tensor(0, device=hidden_states.device, dtype=hidden_states.dtype) for _ in self.config.audio_config.vq_config.codebook_sizes]
        audiotext_loss = torch.tensor(0, device=hidden_states.device, dtype=hidden_states.dtype)
        audios_eos_loss = torch.tensor(0, device=hidden_states.device, dtype=hidden_states.dtype)
        audios_emb_rc_loss = torch.tensor(0, device=hidden_states.device, dtype=hidden_states.dtype)
        # 可训练的音频token
        if labels is not None:
            shift_labels = torch.nn.functional.pad(labels[:,1:], (0, 1), value=0)
            shift_input_ids = torch.nn.functional.pad(input_ids[:,1:], (0, 1), value=0)
            audio_trainable_mask = (((shift_labels == self.config.audio_config.audio_pad_token_id) |
                                     (shift_labels == self.config.audio_config.audiogen_end_token_id)) |
                                     (shift_labels == self.config.audio_config.audio_end_token_id)).to(hidden_states.device)
            audio_tokens_trainable_mask = (shift_labels == self.config.audio_config.audio_pad_token_id)[
                (shift_input_ids == self.config.audio_config.audio_pad_token_id)
            ]
            audiotext_trainable_mask = None
            if audiotext_ids is not None:
                audiotext_trainable_mask = audiotext_ids != self.config.audio_config.audiotext_pad_token_id
                first_audiotext_pad_tag = (audiotext_trainable_mask[:-1] & ~audiotext_trainable_mask[1:]).nonzero(as_tuple=True)[0] + 1 # audiotext后的第一个pad token算loss
                audiotext_trainable_mask[first_audiotext_pad_tag] = True
            # TODO：考虑video
            visual_trainable_mask = ((shift_labels == self.config.visual_config.image_pad_token_id) |
                                     (shift_labels == self.config.visual_config.image_end_token_id)).to(hidden_states.device)
            visual_tokens_trainable_mask = (shift_labels == self.config.visual_config.image_pad_token_id)[
                (shift_input_ids == self.config.visual_config.image_pad_token_id)
            ]
        else:
            audio_trainable_mask = (((input_ids == self.config.audio_config.audio_pad_token_id)
                                    | (input_ids == self.config.audio_config.audiotext_start_token_id))
                                    ).to(hidden_states.device)
            audio_tokens_trainable_mask = ((input_ids == self.config.audio_config.audio_pad_token_id))[
                (input_ids == self.config.audio_config.audio_pad_token_id)
            ]
            visual_trainable_mask = (((input_ids == self.config.visual_config.image_pad_token_id)
                                    | (input_ids == self.config.visual_config.image_start_token_id))
                                    ).to(hidden_states.device)
            visual_tokens_trainable_mask = ((input_ids == self.config.visual_config.image_pad_token_id))[
                (input_ids == self.config.visual_config.image_pad_token_id)
            ]
            if use_cache == True and past_key_values is not None and not self.training:
                assert (audio_trainable_mask.sum() == input_ids.size(0)) or (audio_trainable_mask.sum().item()==0), "AudioGeneration must use cache and input only one audio token per sample"
                assert (visual_trainable_mask.sum() == input_ids.size(0)) or (visual_tokens_trainable_mask.sum().item()==0), "AudioGeneration must use cache and input only one audio token per sample"

        if audio_trainable_mask.sum() > 0 and self.config.audio_config.enable:
            if labels is not None:
                audios_eos_mask = ((torch.nn.functional.pad(labels[:,1:], (0, 1), value=0) == self.config.audio_config.audiogen_end_token_id) | 
                (torch.nn.functional.pad(labels[:,1:], (0, 1), value=0) == self.config.audio_config.audio_end_token_id))[audio_trainable_mask]
                
                newh = torch.cat([hidden_states[audio_trainable_mask][~audios_eos_mask],hidden_states[audio_trainable_mask][audios_eos_mask]],dim=0)
                audios_tokens = audios_tokens[audio_tokens_trainable_mask]
                new_audios_tokens = torch.stack([torch.full([int(audios_eos_mask.sum())],fill_value=self.config.audio_config.vq_config.codebook_sizes[i])
                                                  for i in range(len(self.config.audio_config.vq_config.codebook_sizes))],dim=-1).to(audios_tokens)
                new_audios_tokens = torch.cat([audios_tokens,new_audios_tokens])
                audios_logits = self.audio_head(newh, new_audios_tokens, self.model.audio_embed_layers)
                
                for i in range(len(audios_logits)):
                    audios_loss_list[i]=(F.cross_entropy(audios_logits[i].view(-1,audios_logits[i].shape[-1]),new_audios_tokens[...,i].view(-1)))
                    
                if int(audios_eos_mask.sum()) > 0:
                    audios_eos_loss = F.cross_entropy(audios_logits[0].view(-1,audios_logits[0].shape[-1])[-int(audios_eos_mask.sum()):],new_audios_tokens[...,0].view(-1)[-int(audios_eos_mask.sum()):])
                # greedy 忽略eos
                audios_greedy_tokens = torch.stack([logits.view(-1,logits.shape[-1])[:,:self.config.audio_config.vq_config.codebook_sizes[i]].argmax(-1)  for i,logits in enumerate(audios_logits) ],dim = 1)
                audios_greedy_tokens = audios_greedy_tokens[:audios_greedy_tokens.shape[0]-int(audios_eos_mask.sum())]
                assert audios_greedy_tokens.shape[0] == audios_tokens.shape[0]
                audios_emb_rc_loss = (self.audio_tokenizer.audio_bridge_model.decode(audios_tokens) - self.audio_tokenizer.audio_bridge_model.decode(audios_greedy_tokens)).abs().sum(-1).mean()
            else:
                audios_logits = None
                audios_emb_for_infer = hidden_states[:,-1,:]

        elif self.training and self.config.audio_config.enable:  # training的时候就用fake的loss
            logger.warning("gpu rank={} No trainable audiogen token found, pass fake audio head forward".format(dist.get_rank()))  
            fake_hidden_states = hidden_states.view(-1, hidden_states.shape[-1])[:2]
            fake_logits = self.audio_head(fake_hidden_states, 
                                          torch.full((2, len(self.config.audio_config.vq_config.codebook_sizes)), 0, device=hidden_states.device, dtype=torch.int32),
                                          self.model.audio_embed_layers)
            for i in range(len(audios_loss_list)):
                audios_loss_list[i] = audios_loss_list[i] * fake_logits[i].mean() # fake compute for deepspeed zero3 not lock 
            fake_greedy_tokens = torch.stack([logits.view(-1,logits.shape[-1])[:,:self.config.audio_config.vq_config.codebook_sizes[i]].argmax(-1)  for i,logits in enumerate(fake_logits)],dim = -1)
            _ = (self.audio_tokenizer.audio_bridge_model.decode(fake_greedy_tokens)-self.audio_tokenizer.audio_bridge_model.decode(fake_greedy_tokens)).abs().sum(-1).mean()
        if labels is not None and audiotext_trainable_mask is not None and audiotext_trainable_mask.sum() > 0 and self.config.audio_config.enable:
            shifted_audiotext_logits = logits[..., :-1, :].contiguous()[torch.logical_or(torch.logical_or(input_ids[..., 1:].contiguous() == self.config.audio_config.audio_pad_token_id,
                                                                        input_ids[..., 1:].contiguous() == self.config.audio_config.audiotext_pad_token_id),
                                                                        input_ids[..., 1:].contiguous() == self.config.audio_config.audiotext_start_token_id)]
            shifted_audiotext_logits = shifted_audiotext_logits[audiotext_trainable_mask]
            shifted_audiotext_labels = audiotext_ids[audiotext_trainable_mask]
            audiotext_loss = F.cross_entropy(shifted_audiotext_logits.view(-1,shifted_audiotext_logits.shape[-1]),shifted_audiotext_labels.view(-1))

        visual_logits = None
        vision_emb_for_infer = None
        visual_loss_list = [torch.tensor(0, device=hidden_states.device, dtype=hidden_states.dtype) for _ in self.config.visual_quantizer_config.codebook_sizes]
        visual_eos_loss = torch.tensor(0, device=hidden_states.device, dtype=hidden_states.dtype)
        # visual_emb_rc_loss = torch.tensor(0, device=hidden_states.device, dtype=hidden_states.dtype)
        if visual_trainable_mask.sum() > 0 and self.config.visual_config.image_head_config.enable:
            if labels is not None:
                # SpatialMerger 启用时 visual_tokens 长度是 N_lvl1 = N_lvl2 * mms²，
                # mask 仍然是 N_lvl2 维度（image_pad token 数）。把 visual_tokens 折叠后再 mask 选，再展平。
                mms = int(getattr(self.config.visual_config, "merger_merge_size", 1))
                mms_unit = mms * mms
                visual_eos_mask = (torch.nn.functional.pad(labels[:,1:], (0, 1), value=0) == self.config.visual_config.image_end_token_id)[visual_trainable_mask]
                newh = torch.cat([hidden_states[visual_trainable_mask][~visual_eos_mask],hidden_states[visual_trainable_mask][visual_eos_mask]],dim=0)
                if mms_unit > 1:
                    vt = visual_tokens.view(-1, mms_unit, visual_tokens.shape[-1])  # [N_lvl2, mms², depth]
                    visual_tokens_trainable = vt[visual_tokens_trainable_mask].reshape(-1, visual_tokens.shape[-1])
                    eos_count = int(visual_eos_mask.sum()) * mms_unit
                else:
                    visual_tokens_trainable = visual_tokens[visual_tokens_trainable_mask]
                    eos_count = int(visual_eos_mask.sum())
                new_visual_tokens = torch.stack([torch.full([eos_count],fill_value=self.config.visual_quantizer_config.codebook_sizes[i])
                                                  for i in range(len(self.config.visual_quantizer_config.codebook_sizes))],dim=-1).to(visual_tokens_trainable)
                new_visual_tokens = torch.cat([visual_tokens_trainable,new_visual_tokens])
                visual_logits = self.image_head(newh, new_visual_tokens, self.model.visual_bridge_model.embedding_layers)

                # MaskGit 头: masked=1.0 / unmasked=0.1 加权 (MBM 式), 把学习集中到被遮 level。
                # 头在 forward 里把本次训练遮罩存到 _last_train_mask ([N, depth] bool, 与 logits 行序对齐)。
                # Both the legacy boolean MaskGiT config and the new explicit
                # grouped MaskGiT head expose ``image_head.mask``.  Dispatch on
                # the constructed module so f64g32 does not silently fall back
                # to unweighted CE merely because it uses image_head_type.
                _mask_train = getattr(self.image_head, "_last_train_mask", None) if \
                    getattr(self.image_head, "mask", False) else None
                if _mask_train is not None:
                    # self-refine 第三档: 被扰动的"未遮"槽是真正的去噪任务 (输入近邻错 token, target 干净),
                    # 权重抬到 refine_loss_weight; 干净未遮仍 0.1, 被遮仍 1.0。CE target 始终是干净 GT。
                    _perturb_train = getattr(self.image_head, "_last_perturb_mask", None)
                    _refine_w = float(getattr(self.image_head, "mask_refine_loss_weight", 1.0))
                    for i in range(len(visual_logits)):
                        ce = F.cross_entropy(visual_logits[i].view(-1, visual_logits[i].shape[-1]),
                                             new_visual_tokens[..., i].view(-1), reduction='none')
                        w = torch.where(_mask_train[:, i], 1.0, 0.1).to(ce)
                        if _perturb_train is not None:
                            w = torch.where(_perturb_train[:, i], _refine_w, w).to(ce)
                        visual_loss_list[i] = (ce * w).sum() / (w.sum() + 1e-8)
                else:
                    for i in range(len(visual_logits)):
                        visual_loss_list[i]=(F.cross_entropy(visual_logits[i].view(-1,visual_logits[i].shape[-1]),new_visual_tokens[...,i].view(-1)))

                if eos_count > 0:
                    visual_eos_loss = F.cross_entropy(visual_logits[0].view(-1,visual_logits[0].shape[-1])[-eos_count:],new_visual_tokens[...,0].view(-1)[-eos_count:])
                ## greedy 忽略eos
                # visual_greedy_tokens = torch.stack([logits.view(-1,logits.shape[-1])[:,:self.config.visual_quantizer_config.codebook_sizes[i]].argmax(-1)  for i,logits in enumerate(visual_logits) ],dim = 1)
                # visual_greedy_tokens = visual_greedy_tokens[:visual_greedy_tokens.shape[0]-int(visual_eos_mask.sum())]
                # assert visual_greedy_tokens.shape[0] == visual_tokens.shape[0]
                # audios_emb_rc_loss = (self.audio_tokenizer.audio_bridge_model.decode(audios_tokens) - self.audio_tokenizer.audio_bridge_model.decode(audios_greedy_tokens)).abs().sum(-1).mean()
            else:
                visual_logits = None
                vision_emb_for_infer = hidden_states[:,-1,:]

        elif self.training and self.config.visual_config.enable and self.config.visual_config.image_head_config.enable:  # training的时候就用fake的loss
            logger.warning("gpu rank={} No trainable image token found, pass fake image head forward".format(dist.get_rank()))
            mms = int(getattr(self.config.visual_config, "merger_merge_size", 1))
            mms_unit = mms * mms
            fake_hidden_states = hidden_states.view(-1, hidden_states.shape[-1])[:2]
            fake_logits = self.image_head(fake_hidden_states,
                                          torch.full((2 * mms_unit, len(self.config.visual_quantizer_config.codebook_sizes)), 0, device=hidden_states.device, dtype=torch.int32),
                                          self.model.visual_bridge_model.embedding_layers)
            for i in range(len(visual_loss_list)):
                visual_loss_list[i] = visual_loss_list[i] * fake_logits[i].mean() # fake compute for deepspeed zero3 not lock
            # fake_greedy_tokens = torch.stack([logits.view(-1,logits.shape[-1])[:,:self.config.audio_config.vq_config.codebook_sizes[i]].argmax(-1)  for i,logits in enumerate(fake_logits)],dim = -1)
            # _ = (self.audio_tokenizer.audio_bridge_model.decode(fake_greedy_tokens)-self.audio_tokenizer.audio_bridge_model.decode(fake_greedy_tokens)).abs().sum(-1).mean()

        loss = torch.tensor(0, device=hidden_states.device, dtype=hidden_states.dtype)
        text_nt_loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            valid_mask = torch.gt(shift_labels, -1)  # label < 0 视为pad位置
            sp_mask = sp_mask[..., 1:].contiguous()
            sp_with_loss_mask = sp_with_loss_mask[..., 1:].contiguous()
            text_mask = torch.logical_and(valid_mask, torch.logical_not(sp_mask))
            valid_mask = torch.logical_or(torch.logical_and(valid_mask, torch.logical_not(sp_mask)),sp_with_loss_mask)
            
            
            # Flatten the tokens
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            flatten_loss = F.cross_entropy(shift_logits, shift_labels, ignore_index=-100, reduction='none')

            # 只在对应 head 实际启用时才将其 token 数计入分母，
            # 否则这些 token 不贡献 loss 分子却会稀释梯度。
            loss_denominator = valid_mask.sum()
            if self.config.audio_config.enable:
                loss_denominator = loss_denominator + audio_trainable_mask.sum()
            if self.config.visual_config.image_head_config.enable:
                loss_denominator = loss_denominator + visual_trainable_mask.sum()

            loss = torch.sum(torch.masked_select(flatten_loss, valid_mask.view(-1))) / (1e-10 + loss_denominator)
            text_nt_loss = torch.sum(torch.masked_select(flatten_loss, text_mask.view(-1))) / (1e-10+text_mask.sum())
            # 音频 token 和 text token 权重保持一致
            audio_loss_weight = getattr(self.config.audio_config, "audio_loss_weight", 1.0)
            visual_loss_weight = getattr(self.config.visual_config, "visual_loss_weight", 1.0)

            # 直接对 visual_loss_list 应用权重
            for i in range(len(visual_loss_list)):
                visual_loss_list[i] *= visual_loss_weight

            loss = loss + (sum([audios_loss_list[i] * self.config.audio_config.audio_codebook_loss_weights[i] for i in range(len(audios_loss_list))])
                                * audio_trainable_mask.sum() / (1e-10 + loss_denominator)
                                * audio_loss_weight)
            if audiotext_trainable_mask is not None and audiotext_trainable_mask.sum() > 0 and self.config.audio_config.enable:
                loss = loss + audiotext_loss * self.config.audio_config.audiotext_loss_weight * audiotext_trainable_mask.sum() / (1e-10+valid_mask.sum() + visual_trainable_mask.sum() + audiotext_trainable_mask.sum())
            loss = loss + (sum([visual_loss_list[i] * self.config.visual_config.image_head_config.visual_codebook_loss_weights[i] for i in range(len(visual_loss_list))])
                                * visual_trainable_mask.sum() / (1e-10 + loss_denominator))

        return OmniMMCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            text_nt_loss=text_nt_loss,
            audios_logits = audios_logits,
            audios_loss_list=audios_loss_list,
            audios_eos_loss=audios_eos_loss,
            audios_emb_rc_loss=audios_emb_rc_loss,
            audios_emb_for_infer=audios_emb_for_infer,
            vision_emb_for_infer=vision_emb_for_infer,
            visual_logits=visual_logits,
            visual_eos_loss=visual_eos_loss,
            visual_loss_list=visual_loss_list,
            audiotext_loss=audiotext_loss
        )

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        # NOTE: during generation, `past_key_values` may be "truthy" but contain None (e.g. past_key_values[0][0] is None),
        # which would crash on `.shape`. Here we compute `past_len` safely and treat empty/broken cache as no cache.
        pkv = past_key_values
        past_len = 0
        try:
            if pkv is not None and hasattr(pkv, "get_seq_length"):
                past_len = int(pkv.get_seq_length())
            elif pkv is not None:
                first_layer = pkv[0]
                first_k = first_layer[0]
                if first_k is None:
                    past_len = 0
                else:
                    past_len = int(first_k.shape[-2])
        except Exception:
            past_len = 0

        if past_len <= 0:
            pkv = None
            past_len = 0
        else:
            input_ids = input_ids[:, past_len:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1)
            # position_ids.masked_fill_(attention_mask == 0, 1)
            if past_len > 0:
                position_ids = position_ids[:, past_len:]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and pkv is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        elif pkv is not None:
            model_inputs = {"input_ids": input_ids}
        else:
            model_inputs = {"input_ids": input_ids, 
                            "audios": kwargs.get("audios", None), "encoder_length": kwargs.get("encoder_length", None), "bridge_length": kwargs.get("bridge_length", None), "audiotext_ids": kwargs.get("audiotext_ids", None),
                            "audios_tokens": kwargs.get("audios_tokens", None),
                            "images": kwargs.get("images", None),
                            "videos": kwargs.get("videos", None),
                            "vision_tokens": kwargs.get("vision_tokens", None),
                            }

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": pkv,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "images_grid":  kwargs.get("images_grid"),
                "videos_grid":  kwargs.get("videos_grid"),
                "patch_nums":  kwargs.get("patch_nums"),
                "videos_patch_nums":  kwargs.get("videos_patch_nums"),
            }
        )
        return model_inputs
        
    # def prepare_inputs_for_generation(
    #     self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    # ):
    #     if past_key_values:
    #         input_ids = input_ids[:, past_key_values[0][0].shape[-2]:]

    #     position_ids = kwargs.get("position_ids", None)
    #     if attention_mask is not None and position_ids is None:
    #         # create position_ids on the fly for batch generation
    #         position_ids = attention_mask.long().cumsum(-1)
    #         # position_ids.masked_fill_(attention_mask == 0, 1)
    #         if past_key_values:
    #             position_ids = position_ids[:, past_key_values[0][0].shape[-2]:]

    #     # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
    #     if inputs_embeds is not None and past_key_values is None:
    #         model_inputs = {"inputs_embeds": inputs_embeds}
    #     elif past_key_values is not None:
    #         model_inputs = {"input_ids": input_ids}
    #     else:
    #         model_inputs = {"input_ids": input_ids, 
    #                         "audios": kwargs.get("audios", None), "encoder_length": kwargs.get("encoder_length", None), "bridge_length": kwargs.get("bridge_length", None), "audiotext_ids": kwargs.get("audiotext_ids", None),
    #                         "audios_tokens": kwargs.get("audios_tokens", None),
    #                         "images": kwargs.get("images", None),
    #                         "videos": kwargs.get("videos", None),
    #                         "vision_tokens": kwargs.get("vision_tokens", None),
    #                         }

    #     model_inputs.update(
    #         {
    #             "position_ids": position_ids,
    #             "past_key_values": past_key_values,
    #             "use_cache": kwargs.get("use_cache"),
    #             "attention_mask": attention_mask,
    #             "images_grid":  kwargs.get("images_grid"),
    #             "videos_grid":  kwargs.get("videos_grid"),
    #             "patch_nums":  kwargs.get("patch_nums"),
    #             "videos_patch_nums":  kwargs.get("videos_patch_nums"),
    #         }
    #     )
    #     return model_inputs
    
    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (tuple(past_state.index_select(0, beam_idx) for past_state in layer_past),)
        return reordered_past

    def chat(self, tokenizer, messages: List[dict], stream=False,
             generation_config: Optional[GenerationConfig]=None):
        generation_config = generation_config or self.generation_config
        input_ids = build_chat_input(self, tokenizer, messages, generation_config.max_new_tokens)
        if stream:
            streamer = TextIterStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
            Thread(target=self.generate, kwargs=dict(
                inputs=input_ids, streamer=streamer,
                generation_config=generation_config,
            )).start()
            return streamer
        else:
            outputs = self.generate(input_ids, generation_config=generation_config)
            response = tokenizer.decode(outputs[0][len(input_ids[0]):], skip_special_tokens=True)
            return response


class OmniAudioTokenizer(OmniPreTrainedModel):
    """
    Construct an audio tokenizer and decoder.
    """
    def __init__(self, config: OmniConfig):
        super().__init__(config)
        self.padding_idx = None
        self.vocab_size = config.vocab_size
        self.training = False
        self.eval()
        self.audio_model = OmniAudioEncoder(config.audio_config)
        self.audio_bridge_model = OmniAudioVQBridgeTokenizer(config)
        if config.vocoder_config.enable:
            self.audio_decoder = OmniAudioDecoder(config)
            if config.flow_matching_config.enable:
                self.audio_flow_matching_decoder = OmniAudioFlowMatchingDecoder(config)
        
        for param in self.parameters():  # 修复SFT阶段可能错误加载导致OOM的问题
            param.requires_grad = False

    def encode(self, x, encoder_length: Optional[torch.Tensor] = None, bridge_length: Optional[torch.Tensor] = None):
        audio_emb = self.audio_model(x, encoder_length)
        audios_tokens = self.audio_bridge_model(audio_emb, bridge_length)
        return audios_tokens
    
    def decode(self, audio_code_ids, bridge_length: Optional[torch.Tensor] = None):
        assert self.config.vocoder_config.enable, "Vocoder is not enabled in config."
        audio_emb = self.audio_bridge_model.decode(audio_code_ids)
        audio_dec = self.audio_decoder(
            audio_emb.to(next(self.audio_decoder.parameters())), bridge_length
        )
        if self.config.flow_matching_config.enable:
            if self.config.flow_matching_config.use_hidden_states_before_dconv2:
                hidden_states, hidden_states_length = (
                    self.audio_flow_matching_decoder.unpack_hidden_states(
                        audio_dec.hidden_states_before_dconv2,
                        audio_dec.output_length_before_dconv2,
                    )
                )
                audio_flow_matching_decoder_ret = self.audio_flow_matching_decoder(
                    hidden_states, hidden_states_length
                )

            else:
                audio_flow_matching_decoder_ret = self.audio_flow_matching_decoder(
                    audio_dec.refined_mel, audio_dec.mel_length
                )
            return audio_flow_matching_decoder_ret
        else:
            return audio_dec
    
    @torch.no_grad()    
    def forward(self, audios, encoder_length: Optional[torch.Tensor] = None, bridge_length: Optional[torch.Tensor] = None):
        self.eval()
        audios_tokens = self.encode(audios, encoder_length, bridge_length)
        return audios_tokens
