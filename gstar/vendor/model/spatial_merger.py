"""SpatialMerger / SpatialUnmerger.

在 LLM 边界两侧做 ms x ms 的空间合并 / 还原。这里的 ms 就是 VLM 顶层
visual_config.merger_merge_size——与 ViT.spatial_merge_size 和
visual_input_bridge.merge_size **完全独立**：
- ViT.spatial_merge_size    : 2D RoPE 块边长，不改 token 数
- visual_input_bridge.merge_size : tokenizer 内的 patch-merger（在 LLM 之前先 m×m 折 1）
- merger_merge_size (本文件 ms): LLM 边界再做一次 m×m → 1（即"merger 的 merger"），
  把 LLM 看到的 image token 数再压 m² 倍

形状语义：
- SpatialMerger: 假定输入 [N_lvl1, H] 已经是 *block-major* 顺序（相邻 ms*ms 个 token
  属于同一空间块），做两层 GELU MLP 把 ms*ms*H 压成 H -> [N_lvl2, H]。N_lvl2 = N_lvl1 / ms²。
- SpatialUnmerger: [N_lvl2, H] -> [N_lvl2*ms*ms, H]，两层 GELU MLP 展回 ms*ms*H 然后 reshape，
  输出依然是 block-major（让 image_head 的 logits 与 reorder 后的 indices 对位做 CE）。

block-major 顺序由 `spatial_merge_reorder` 工具产生（外部对 indices/labels 应用同一函数即可保持对齐）。
reorder 公式与 rvq-tokenizer 中 `spatial_merge_reorder` 一致：
feat[B, h, w, D] -> view(B, mh, ms, mw, ms, D) -> permute(0, 1, 3, 2, 4, 5) -> view(B*mh*mw*ms*ms, D)

调用方约束：grid_thw 的 h/w 必须能被 ms 整除。这一点由 processor 端 smart_resize
（factor = patch_size * spatial_merge_size * merger_merge_size）和 *Encoder.fake_input
（grid 边长取 max(merge_size, spatial_merge_size, merger_merge_size)）共同保证。
"""

import torch
from torch import nn
from transformers.activations import ACT2FN


def spatial_merge_reorder(
    tokens: torch.Tensor, batch: int, h: int, w: int, merge_size: int
) -> torch.Tensor:
    """[B*h*w, ...] (row-major) -> [B*h*w, ...] (block-major: 相邻 ms*ms 个属于同一块).

    支持任意尾维 (e.g. tokens [N, D] 或 indices [N, depth]).
    """
    tail_shape = tuple(tokens.shape[1:])
    feat = tokens.view(batch, h, w, *tail_shape)
    mh, mw = h // merge_size, w // merge_size
    feat = feat.view(batch, mh, merge_size, mw, merge_size, *tail_shape)
    perm = (0, 1, 3, 2, 4) + tuple(range(5, 5 + len(tail_shape)))
    feat = feat.permute(*perm).contiguous()
    return feat.view(batch * h * w, *tail_shape)


def block_major_reorder_per_image(
    tokens: torch.Tensor, grid_thw: torch.Tensor, merge_size: int
) -> torch.Tensor:
    """对每张图（按 grid_thw 切片）应用 spatial_merge_reorder。tokens: [N_lvl1, ...].

    grid_thw 必须已经是"LLM 边界看到的 token 网格"——即 ViT patch 网格除掉
    visual_input_bridge.merge_size 之后的结果。每张图的 (h, w) 必须能被
    merge_size（= merger_merge_size）整除。

    会触发 assert 的几种情形：
    1) fake_input 路径产出的 grid 边长没并入 merger_merge_size。
       修法：*Encoder.fake_input 里 merge_size=max(self.merge_size, spatial_merge_size, merger_merge_size)。
    2) processor smart_resize.factor 没乘上 merger_merge_size。
       修法：factor = patch_size * spatial_merge_size * merger_merge_size。
    3) visual_input_bridge.merge_size（tokenizer_ms）与 VLM 顶层 visual_config.merge_size 不一致，
       导致 grid_thw_after_tok = grid // tokenizer_ms 之后再除 mms 不再整除。
       修法：让 RVQ 端 bridge.merge_size 与 VLM 顶层 visual_config.merge_size 对齐。
    """
    chunks = []
    offset = 0
    for i in range(grid_thw.shape[0]):
        t, h, w = int(grid_thw[i, 0]), int(grid_thw[i, 1]), int(grid_thw[i, 2])
        if h % merge_size != 0 or w % merge_size != 0:
            raise AssertionError(
                f"spatial reorder requires (h, w) divisible by merger_merge_size; "
                f"image[{i}] (t={t}, h={h}, w={w}) %% mms={merge_size}; "
                f"grid_thw rows total={grid_thw.shape[0]}; "
                f"all_grid_hw={[tuple(grid_thw[k, 1:].tolist()) for k in range(grid_thw.shape[0])]}; "
                f"hint: 若 (h, w) == (1, 1) 通常是 *Encoder.fake_input 没并入 merger_merge_size; "
                f"若是真实图触发，检查 processor smart_resize.factor 与 visual_input_bridge.merge_size 是否与配置对齐."
            )
        n = t * h * w
        feat = tokens[offset:offset + n]
        chunks.append(spatial_merge_reorder(feat, t, h, w, merge_size))
        offset += n
    return torch.cat(chunks, dim=0)


class SpatialMerger(nn.Module):
    """ms*ms 个 block-major 子 token -> 1 个 token。两层 MLP + 激活."""

    def __init__(self, hidden_size: int, intermediate_size: int, merge_size: int = 2, hidden_act: str = "gelu"):
        super().__init__()
        self.hidden_size = hidden_size
        self.merge_size = merge_size
        self.merge_unit = merge_size * merge_size
        self.norm = nn.LayerNorm(hidden_size)
        self.fc1 = nn.Linear(hidden_size * self.merge_unit, intermediate_size)
        self.act = ACT2FN[hidden_act]
        self.fc2 = nn.Linear(intermediate_size, hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """hidden_states: [N_lvl1, H]，必须是 block-major 顺序。返回 [N_lvl2, H]."""
        n_lvl1 = hidden_states.shape[0]
        assert n_lvl1 % self.merge_unit == 0, (
            f"SpatialMerger expects N divisible by {self.merge_unit}, got {n_lvl1}"
        )
        x = self.norm(hidden_states)
        x = x.view(n_lvl1 // self.merge_unit, self.merge_unit * self.hidden_size)
        x = self.fc2(self.act(self.fc1(x)))
        return x


class SpatialUnmerger(nn.Module):
    """1 个 token -> ms*ms 个 out_dim 的子 token，输出 block-major。两层 MLP + 激活."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        merge_size: int = 2,
        hidden_act: str = "gelu",
        out_dim: int = None,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.out_dim = out_dim if out_dim is not None else hidden_size
        self.merge_size = merge_size
        self.merge_unit = merge_size * merge_size
        self.norm = nn.LayerNorm(hidden_size)
        self.fc1 = nn.Linear(hidden_size, intermediate_size)
        self.act = ACT2FN[hidden_act]
        self.fc2 = nn.Linear(intermediate_size, self.out_dim * self.merge_unit)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [N_lvl2, H] -> [N_lvl2*ms*ms, out_dim] (block-major)."""
        h = self.norm(x)
        h = self.fc2(self.act(self.fc1(h)))
        return h.view(x.shape[0] * self.merge_unit, self.out_dim)
