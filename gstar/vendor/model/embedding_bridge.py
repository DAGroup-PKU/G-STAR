import torch
from torch import nn
from transformers.activations import ACT2FN

from .spatial_merger import SpatialMerger


class _VEmbMLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, hidden_act: str):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.act_fn = ACT2FN[hidden_act]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class _VEmbDecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.mlp = _VEmbMLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
        )
        self.post_attention_layernorm = nn.LayerNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask=None,
        position_ids=None,
        seqlens=None,
        past_key_value=None,
        output_attentions: bool = False,
        use_cache: bool = False,
        group_index=None,
    ):
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return (hidden_states,)


class VisualEmbeddingBridge(nn.Module):
    """Maps discrete VQ indices to LLM hidden space via per-level Embedding + lightweight FFN."""

    def __init__(self, config):
        super().__init__()
        # PQ 模式 (image_head_pq): tokenizer 是 Product Quantization, 8 个 group 是并行独立子空间。
        # 输入侧不该像 RVQ 那样把 per-group embedding 求和(残差语义), 而是 concat 后过一个新 linear
        # (UniTok idx_to_f 式)。per-group embedding 宽度 = hidden_size // depth, concat 回 hidden_size。
        _ihc = getattr(config.visual_config, "image_head_config", None)
        self.pq = bool(getattr(_ihc, "image_head_pq", False)) if _ihc is not None else False
        codebook_sizes = config.visual_quantizer_config.codebook_sizes
        depth = len(codebook_sizes)
        if self.pq:
            assert config.hidden_size % depth == 0, "PQ concat bridge 要求 hidden_size 能被 group 数整除"
            emb_width = config.hidden_size // depth
            self.embedding_layers = nn.ModuleList([
                nn.Embedding(codedim + 1, emb_width) for codedim in codebook_sizes
            ])
            self.input_proj = nn.Linear(emb_width * depth, config.hidden_size)
        else:
            self.embedding_layers = nn.ModuleList([
                nn.Embedding(codedim + 1, config.hidden_size)
                for _, codedim in enumerate(codebook_sizes)
            ])
        self.config = config
        self.transformer_block = _VEmbDecoderLayer(config)

        merger_merge_size = int(getattr(config.visual_config, "merger_merge_size", 1))
        self.merger_merge_size = merger_merge_size
        if merger_merge_size > 1:
            merger_inter = int(getattr(
                config.visual_config,
                "merger_intermediate_size",
                config.hidden_size * merger_merge_size * merger_merge_size,
            ))
            merger_act = getattr(config.visual_config, "merger_hidden_act", "gelu")
            self.spatial_merger = SpatialMerger(
                hidden_size=config.hidden_size,
                intermediate_size=merger_inter,
                merge_size=merger_merge_size,
                hidden_act=merger_act,
            )
        else:
            self.spatial_merger = None

        # RAR needs two independent spatial-position tables.  The source table
        # tells the LLM where the visible token came from; the target-aware
        # table tells it which original raster position should be predicted
        # next.  Keeping them under visual_bridge_model also makes the existing
        # bridge_llm_image_head optimizer mode train them automatically.
        self.rar_random_order = bool(getattr(
            config.visual_config, "rar_random_order", False))
        self.rar_num_spatial_tokens = int(getattr(
            config.visual_config, "rar_num_spatial_tokens", 0) or 0)
        if self.rar_random_order:
            assert self.rar_num_spatial_tokens > 0, \
                "RAR requires visual_config.rar_num_spatial_tokens > 0"
            self.rar_source_pos_embed = nn.Embedding(
                self.rar_num_spatial_tokens, config.hidden_size)
            self.rar_target_pos_embed = nn.Embedding(
                self.rar_num_spatial_tokens, config.hidden_size)

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        """indices: [N, depth]. 当 SpatialMerger 启用时，调用方需要保证 indices 已经按 block-major reorder。"""
        if self.pq:
            # PQ: concat 8 个 group 的 per-group embedding 后过 input_proj (非残差求和)。
            embedding = self.input_proj(torch.cat(
                [emb(indices[..., i]) for i, emb in enumerate(self.embedding_layers)], dim=-1
            ))
        else:
            for i, embedding_layer in enumerate(self.embedding_layers):
                if i == 0:
                    embedding = embedding_layer(indices[..., i])
                else:
                    embedding = embedding + embedding_layer(indices[..., i])

        embedding = self.transformer_block(embedding)[0]
        embedding = embedding.view(-1, embedding.shape[-1])

        if self.spatial_merger is not None:
            embedding = self.spatial_merger(embedding)
        return embedding
