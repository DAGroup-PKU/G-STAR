"""
Interleaved-MRoPE (交错多模态旋转位置编码) 实现
基于Qwen3-VL的Interleaved-MRoPE技术，支持3D位置编码和多模态处理
"""
import torch
import torch.nn as nn
from typing import Optional, Tuple
import math


def get_rope_index_3d(
    spatial_merge_size: Optional[int] = 2,
    input_ids: Optional[torch.LongTensor] = None,
    image_grid_thw: Optional[torch.LongTensor] = None,
    video_grid_thw: Optional[torch.LongTensor] = None,
    second_per_grid_ts: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    config: Optional[object] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Calculate the 3D rope index based on image and video's temporal, height and width in LLM.
    
    Args:
        spatial_merge_size: Spatial merge size for reducing resolution
        input_ids: Input token IDs
        image_grid_thw: Image grid dimensions (T, H, W)
        video_grid_thw: Video grid dimensions (T, H, W)
        second_per_grid_ts: Time interval per grid for videos
        attention_mask: Attention mask
        
    Returns:
        position_ids: 3D position IDs (3, batch_size, sequence_length)
        mrope_position_deltas: Position deltas for MRoPE
    """
    # Get token IDs from config
    if config is not None:
        image_start_token_id = config.visual_config.image_start_token_id
        image_pad_token_id = config.visual_config.image_pad_token_id
        image_end_token_id = config.visual_config.image_end_token_id
        video_start_token_id = config.video_config.video_start_token_id
        video_end_token_id = config.video_config.video_end_token_id
        vision_start_token_id = config.visual_config.image_start_token_id
    else:
        # Fallback to default values if config not provided
        image_start_token_id = 151679
        image_pad_token_id = 151681
        image_end_token_id = 151680
        video_start_token_id = 151695
        video_end_token_id = 151696
        vision_start_token_id = 151679
    
    mrope_position_deltas = []
    
    if input_ids is not None and (image_grid_thw is not None or video_grid_thw is not None):
        total_input_ids = input_ids
        if attention_mask is None:
            attention_mask = torch.ones_like(total_input_ids)
            
        # Initialize 3D position IDs
        position_ids = torch.ones(
            3,  # 3 dimensions: temporal, height, width
            input_ids.shape[0],
            input_ids.shape[1],
            dtype=input_ids.dtype,
            device=input_ids.device,
        )
        
        image_index, video_index = 0, 0
        attention_mask = attention_mask.to(total_input_ids.device)
        
        for i, input_ids in enumerate(total_input_ids):
            input_ids = input_ids[attention_mask[i] == 1]
            image_nums, video_nums = 0, 0
            
            # Find vision start tokens and count images/videos
            vision_start_indices = torch.argwhere(input_ids == vision_start_token_id).squeeze(1)
            image_nums, video_nums = 0, 0
            
            # Count images and videos by looking for start tokens followed by pad tokens
            for idx in vision_start_indices:
                if idx + 1 < len(input_ids):
                    next_token = input_ids[idx + 1]
                    if next_token == image_pad_token_id:
                        image_nums += 1
                    elif next_token == video_start_token_id:
                        video_nums += 1
            
            input_tokens = input_ids.tolist()
            llm_pos_ids_list: list = []
            st = 0
            remain_images, remain_videos = image_nums, video_nums
            
            # Process each vision segment
            for _ in range(image_nums + video_nums):
                # Find next image pad token
                if image_pad_token_id in input_tokens and remain_images > 0:
                    ed_image = input_tokens.index(image_pad_token_id, st)
                else:
                    ed_image = len(input_tokens) + 1
                # Find next video start token  
                if video_start_token_id in input_tokens and remain_videos > 0:
                    ed_video = input_tokens.index(video_start_token_id, st)
                else:
                    ed_video = len(input_tokens) + 1
                    
                if ed_image < ed_video:
                    # Process image
                    t, h, w = (
                        image_grid_thw[image_index][0],
                        image_grid_thw[image_index][1],
                        image_grid_thw[image_index][2],
                    )
                    image_index += 1
                    remain_images -= 1
                    ed = ed_image
                else:
                    # Process video
                    t, h, w = (
                        video_grid_thw[video_index][0],
                        video_grid_thw[video_index][1],
                        video_grid_thw[video_index][2],
                    )
                    video_index += 1
                    remain_videos -= 1
                    ed = ed_video
                
                # Calculate LLM grid dimensions
                llm_grid_t, llm_grid_h, llm_grid_w = (
                    t.item() if hasattr(t, 'item') else int(t),
                    (h.item() if hasattr(h, 'item') else int(h)) // spatial_merge_size,
                    (w.item() if hasattr(w, 'item') else int(w)) // spatial_merge_size,
                )
                text_len = ed - st
                
                # Calculate starting index
                st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                
                # Add text position IDs (1D for text)
                llm_pos_ids_list.append(
                    torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx
                )
                
                # Generate 3D position IDs for vision tokens
                t_index = torch.arange(llm_grid_t).view(-1, 1).expand(-1, llm_grid_h * llm_grid_w).flatten()
                h_index = torch.arange(llm_grid_h).view(1, -1, 1).expand(llm_grid_t, -1, llm_grid_w).flatten()
                w_index = torch.arange(llm_grid_w).view(1, 1, -1).expand(llm_grid_t, llm_grid_h, -1).flatten()
                
                llm_pos_ids_list.append(
                    torch.stack([t_index, h_index, w_index]) + text_len + st_idx
                )
                st = ed + llm_grid_t * llm_grid_h * llm_grid_w
            
            # Handle remaining text
            if st < len(input_tokens):
                st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                text_len = len(input_tokens) - st
                llm_pos_ids_list.append(
                    torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx
                )
            
            # Concatenate all position IDs
            llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
            position_ids[..., i, attention_mask[i] == 1] = llm_positions.to(position_ids.device)
            mrope_position_deltas.append(llm_positions.max() + 1 - len(total_input_ids[i]))
            
        mrope_position_deltas = torch.tensor(mrope_position_deltas, device=input_ids.device).unsqueeze(1)
        return position_ids, mrope_position_deltas
    else:
        # Fallback to standard 1D position encoding
        if attention_mask is not None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            position_ids = position_ids.unsqueeze(0).expand(3, -1, -1).to(attention_mask.device)
            max_position_ids = position_ids.max(0, keepdim=False)[0].max(-1, keepdim=True)[0]
            mrope_position_deltas = max_position_ids + 1 - attention_mask.shape[-1]
        else:
            position_ids = (
                torch.arange(input_ids.shape[1], device=input_ids.device)
                .view(1, 1, -1)
                .expand(3, input_ids.shape[0], -1)
            )
            mrope_position_deltas = torch.zeros(
                [input_ids.shape[0], 1],
                device=input_ids.device,
                dtype=input_ids.dtype,
            )
        
        return position_ids, mrope_position_deltas


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos_, sin_, position_ids):
    """Apply standard rotary position embedding."""
    cos = cos_.squeeze(1).squeeze(0)  # [seq_len, dim]
    sin = sin_.squeeze(1).squeeze(0)  # [seq_len, dim]
    
    # Handle different position_ids shapes
    if position_ids.dim() == 3:  # [batch_size, num_heads, seq_len]
        batch_size, num_heads, seq_len = position_ids.shape
        # Flatten position_ids for indexing
        position_ids_flat = position_ids.reshape(-1)  # [batch_size * num_heads * seq_len]
        cos_selected = cos[position_ids_flat]  # [batch_size * num_heads * seq_len, dim]
        sin_selected = sin[position_ids_flat]  # [batch_size * num_heads * seq_len, dim]
        # Reshape back to [batch_size, num_heads, seq_len, dim]
        cos = cos_selected.reshape(batch_size, num_heads, seq_len, -1)
        sin = sin_selected.reshape(batch_size, num_heads, seq_len, -1)
    else:  # [batch_size, seq_len] or [seq_len]
        cos = cos[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
        sin = sin[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    
    q_embed = (q.float() * cos) + (rotate_half(q.float()) * sin)
    k_embed = (k.float() * cos) + (rotate_half(k.float()) * sin)
    return q_embed.to(q.dtype), k_embed.to(k.dtype)


def apply_multimodal_rotary_pos_emb(q, k, cos_, sin_, mrope_section, position_ids_3d=None):
    """
    Apply multimodal rotary position embedding with interleaved frequency allocation.
    
    Args:
        q: Query tensor
        k: Key tensor  
        cos_: Cosine values tuple (cos_t, cos_h, cos_w)
        sin_: Sine values tuple (sin_t, sin_h, sin_w)
        mrope_section: MRoPE section configuration [t_dim, h_dim, w_dim]
        position_ids_3d: Optional 3D position IDs [3, batch_size, seq_len]
        
    Returns:
        q_embed: Rotated query tensor
        k_embed: Rotated key tensor
    """
    # Split dimensions according to mrope_section
    t_dim, h_dim, w_dim = mrope_section
    total_dim = q.shape[-1]
    
    # Calculate dimension splits
    d_t = total_dim * t_dim // sum(mrope_section)
    d_h = total_dim * h_dim // sum(mrope_section)  
    d_w = total_dim - d_t - d_h
    
    # Split q and k into three parts
    q_t = q[..., :d_t]
    q_h = q[..., d_t:d_t+d_h]
    q_w = q[..., d_t+d_h:]
    
    k_t = k[..., :d_t]
    k_h = k[..., d_t:d_t+d_h]
    k_w = k[..., d_t+d_h:]
    
    # Apply RoPE to each part with proper position_ids
    batch_size, num_heads, seq_len = q.shape[:3]
    batch_size_k, num_kv_heads, seq_len_k = k.shape[:3]
    if position_ids_3d is not None:
        # Use 3D position IDs for MRoPE
        # For q: expand to num_heads
        position_ids_t_q = position_ids_3d[0].unsqueeze(1).expand(-1, num_heads, -1)  # [batch_size, num_heads, seq_len]
        position_ids_h_q = position_ids_3d[1].unsqueeze(1).expand(-1, num_heads, -1)  # [batch_size, num_heads, seq_len]
        position_ids_w_q = position_ids_3d[2].unsqueeze(1).expand(-1, num_heads, -1)  # [batch_size, num_heads, seq_len]
        # For k: expand to num_kv_heads
        position_ids_t_k = position_ids_3d[0].unsqueeze(1).expand(-1, num_kv_heads, -1)  # [batch_size, num_kv_heads, seq_len]
        position_ids_h_k = position_ids_3d[1].unsqueeze(1).expand(-1, num_kv_heads, -1)  # [batch_size, num_kv_heads, seq_len]
        position_ids_w_k = position_ids_3d[2].unsqueeze(1).expand(-1, num_kv_heads, -1)  # [batch_size, num_kv_heads, seq_len]
    else:
        # Fallback to sequential position IDs
        position_ids_t_q = torch.arange(seq_len, device=q.device).unsqueeze(0).unsqueeze(0).expand(batch_size, num_heads, -1)
        position_ids_h_q = torch.arange(seq_len, device=q.device).unsqueeze(0).unsqueeze(0).expand(batch_size, num_heads, -1)
        position_ids_w_q = torch.arange(seq_len, device=q.device).unsqueeze(0).unsqueeze(0).expand(batch_size, num_heads, -1)
        position_ids_t_k = torch.arange(seq_len_k, device=k.device).unsqueeze(0).unsqueeze(0).expand(batch_size_k, num_kv_heads, -1)
        position_ids_h_k = torch.arange(seq_len_k, device=k.device).unsqueeze(0).unsqueeze(0).expand(batch_size_k, num_kv_heads, -1)
        position_ids_w_k = torch.arange(seq_len_k, device=k.device).unsqueeze(0).unsqueeze(0).expand(batch_size_k, num_kv_heads, -1)
    
    # Apply RoPE separately for q and k since they may have different dimensions
    q_t_rot = apply_rotary_pos_emb(q_t, q_t, cos_[0], sin_[0], position_ids_t_q)
    k_t_rot = apply_rotary_pos_emb(k_t, k_t, cos_[0], sin_[0], position_ids_t_k)
    q_h_rot = apply_rotary_pos_emb(q_h, q_h, cos_[1], sin_[1], position_ids_h_q)
    k_h_rot = apply_rotary_pos_emb(k_h, k_h, cos_[1], sin_[1], position_ids_h_k)
    q_w_rot = apply_rotary_pos_emb(q_w, q_w, cos_[2], sin_[2], position_ids_w_q)
    k_w_rot = apply_rotary_pos_emb(k_w, k_w, cos_[2], sin_[2], position_ids_w_k)
    
    # Concatenate rotated parts
    q_embed = torch.cat([q_t_rot[0], q_h_rot[0], q_w_rot[0]], dim=-1)
    k_embed = torch.cat([k_t_rot[0], k_h_rot[0], k_w_rot[0]], dim=-1)
    
    return q_embed, k_embed


class MRoPERotaryEmbedding(torch.nn.Module):
    """Rotary Position Embedding with support for Interleaved-MRoPE."""
    
    def __init__(self, dim, max_position_embeddings=2048, base=5e6, device=None, mrope_section=None):
        super().__init__()
        # 修复RePE初始化精度问题 https://zhuanlan.zhihu.com/p/678963442
        # DeepSpeed 会 Hack torch.arange 强制在 GPU 上运行，这里使用原生的 torch.arange
        try:
            import deepspeed
            self.arange = deepspeed.runtime.zero.partition_parameters._orig_torch_arange
        except:
            self.arange = torch.arange

        self.dim = dim
        self.mrope_section = mrope_section
        
        if mrope_section is not None:
            # Interleaved-MRoPE: Create separate frequency sets for T, H, W dimensions
            self._init_mrope_frequencies(dim, base, device, max_position_embeddings)
            # Also initialize standard RoPE cache for 1D fallback compatibility
            self.inv_freq = 1.0 / (base ** (self.arange(0, dim, 2).float().to(device) / dim))
            self.max_seq_len_cached = max_position_embeddings
            t = self.arange(self.max_seq_len_cached, device=self.inv_freq.device, dtype=torch.float32)
            freqs = torch.outer(t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1)
            self.cos_cached = emb.cos()[None, None, :, :].to(torch.float32)
            self.sin_cached = emb.sin()[None, None, :, :].to(torch.float32)
        else:
            # Standard RoPE
            self.inv_freq = 1.0 / (base ** (self.arange(0, dim, 2).float().to(device) / dim))
            self.max_seq_len_cached = max_position_embeddings
            t = self.arange(self.max_seq_len_cached, device=self.inv_freq.device, dtype=torch.float32)
            freqs = torch.outer(t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1)
            self.cos_cached = emb.cos()[None, None, :, :].to(torch.float32)
            self.sin_cached = emb.sin()[None, None, :, :].to(torch.float32)

    def _init_mrope_frequencies(self, dim, base, device, max_position_embeddings):
        """Initialize frequencies for Interleaved-MRoPE with 3D dimensions."""
        t_dim, h_dim, w_dim = self.mrope_section
        
        # Calculate dimension splits
        d_t = dim * t_dim // sum(self.mrope_section)
        d_h = dim * h_dim // sum(self.mrope_section)
        d_w = dim - d_t - d_h
        
        # Create separate frequency sets for each dimension
        self.inv_freq_t = 1.0 / (base ** (self.arange(0, d_t, 2).float().to(device) / d_t))
        self.inv_freq_h = 1.0 / (base ** (self.arange(0, d_h, 2).float().to(device) / d_h))
        self.inv_freq_w = 1.0 / (base ** (self.arange(0, d_w, 2).float().to(device) / d_w))
        
        self.max_seq_len_cached = max_position_embeddings
        
        # Pre-compute cos/sin for each dimension
        t = self.arange(self.max_seq_len_cached, device=device, dtype=torch.float32)
        
        freqs_t = torch.outer(t, self.inv_freq_t)
        freqs_h = torch.outer(t, self.inv_freq_h)
        freqs_w = torch.outer(t, self.inv_freq_w)
        
        emb_t = torch.cat((freqs_t, freqs_t), dim=-1)
        emb_h = torch.cat((freqs_h, freqs_h), dim=-1)
        emb_w = torch.cat((freqs_w, freqs_w), dim=-1)
        
        self.cos_cached_t = emb_t.cos()[None, None, :, :].to(torch.float32)
        self.sin_cached_t = emb_t.sin()[None, None, :, :].to(torch.float32)
        self.cos_cached_h = emb_h.cos()[None, None, :, :].to(torch.float32)
        self.sin_cached_h = emb_h.sin()[None, None, :, :].to(torch.float32)
        self.cos_cached_w = emb_w.cos()[None, None, :, :].to(torch.float32)
        self.sin_cached_w = emb_w.sin()[None, None, :, :].to(torch.float32)

    def forward(self, x, seq_len=None, position_ids_3d=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if self.mrope_section is not None and position_ids_3d is not None:
            # Interleaved-MRoPE with 3D position IDs
            return self._forward_mrope(x, seq_len, position_ids_3d)
        else:
            # Standard RoPE
            rank0_debug()
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

    def _forward_mrope(self, x, seq_len, position_ids_3d):
        """Forward pass for Interleaved-MRoPE with 3D position IDs."""
        # position_ids_3d: [3, batch_size, seq_len] (temporal, height, width)
        if seq_len > self.max_seq_len_cached:
            # extend all cached tensors for T/H/W
            self.max_seq_len_cached = seq_len
            t = self.arange(self.max_seq_len_cached, device=x.device, dtype=torch.float32)
            freqs_t = torch.outer(t, self.inv_freq_t)
            freqs_h = torch.outer(t, self.inv_freq_h)
            freqs_w = torch.outer(t, self.inv_freq_w)
            emb_t = torch.cat((freqs_t, freqs_t), dim=-1)
            emb_h = torch.cat((freqs_h, freqs_h), dim=-1)
            emb_w = torch.cat((freqs_w, freqs_w), dim=-1)
            self.cos_cached_t = emb_t.cos()[None, None, :, :].to(torch.float32).to(x.device)
            self.sin_cached_t = emb_t.sin()[None, None, :, :].to(torch.float32).to(x.device)
            self.cos_cached_h = emb_h.cos()[None, None, :, :].to(torch.float32).to(x.device)
            self.sin_cached_h = emb_h.sin()[None, None, :, :].to(torch.float32).to(x.device)
            self.cos_cached_w = emb_w.cos()[None, None, :, :].to(torch.float32).to(x.device)
            self.sin_cached_w = emb_w.sin()[None, None, :, :].to(torch.float32).to(x.device)
        cos_t = self.cos_cached_t[:, :, :seq_len, ...].to(torch.float32).to(x.device)
        sin_t = self.sin_cached_t[:, :, :seq_len, ...].to(torch.float32).to(x.device)
        cos_h = self.cos_cached_h[:, :, :seq_len, ...].to(torch.float32).to(x.device)
        sin_h = self.sin_cached_h[:, :, :seq_len, ...].to(torch.float32).to(x.device)
        cos_w = self.cos_cached_w[:, :, :seq_len, ...].to(torch.float32).to(x.device)
        sin_w = self.sin_cached_w[:, :, :seq_len, ...].to(torch.float32).to(x.device)
        
        return (
            (cos_t, cos_h, cos_w),
            (sin_t, sin_h, sin_w)
        )


# 导出所有公共函数和类
__all__ = [
    'MRoPERotaryEmbedding',
    'apply_multimodal_rotary_pos_emb', 
    'get_rope_index_3d',
    'rotate_half',
    'apply_rotary_pos_emb'
]