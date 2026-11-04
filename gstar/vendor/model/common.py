import torch
from torch import nn
from torch.nn import functional as F
from flash_attn import flash_attn_varlen_func
from .audio_modeling_omni import RMSNorm

try:
    from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2RMSNorm
except ImportError:
    Qwen2RMSNorm = None


class FlashVarLenAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, causal=False, window_size=(-1, -1)):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)

        self.causal = causal
        self.window_size = window_size

    def forward(self, hidden_states: torch.Tensor, seq_len: torch.Tensor):
        bsz, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        query_states = query_states.view(bsz, self.num_heads, self.head_dim).contiguous()
        key_states = self.k_proj(hidden_states)
        key_states = key_states.view(bsz, self.num_heads, self.head_dim).contiguous()
        value_states = self.v_proj(hidden_states)
        value_states = value_states.view(bsz, self.num_heads, self.head_dim).contiguous()

        cu_len = F.pad(torch.cumsum(seq_len, dim=0), (1, 0), "constant", 0).to(torch.int32)
        max_seqlen = torch.max(seq_len).to(torch.int32).detach()
        attn_output = flash_attn_varlen_func(query_states, key_states, value_states, cu_len, cu_len, max_seqlen,
                                             max_seqlen, causal=self.causal, window_size=self.window_size)
        attn_output = attn_output.reshape(bsz, self.embed_dim)
        attn_output = self.out_proj(attn_output)
        return attn_output
