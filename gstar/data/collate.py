"""Batch collation, and the batch-local attention layout prepared before the Trainer's device transfer."""
from functools import partial

import numpy as np
import torch

from ..vendor.model.processor_omni import OmniProcessorOutput


def collate(items):
    merged = OmniProcessorOutput()
    for item in items:
        merged = merged.concatenate(OmniProcessorOutput(**item))
    result = {}
    for key in ("input_ids", "labels", "position_ids", "seqlens"):
        result[key] = torch.tensor([item[key] for item in items], dtype=torch.long)
    if merged.images is not None:
        result["images"] = [torch.from_numpy(np.asarray(x, dtype=np.float32)) for x in merged.images]
    if merged.patch_nums is not None:
        result["patch_nums"] = torch.tensor(merged.patch_nums, dtype=torch.long)
    if merged.images_grid is not None:
        result["images_grid"] = merged.images_grid
    if merged.vision_tokens is not None:
        result["vision_tokens"] = torch.from_numpy(np.concatenate(merged.vision_tokens).astype(np.int64))
    return result


def prepare_attention_layout(inputs, include_scalars=False):
    lengths, positions = inputs.get('seqlens'), inputs.get('position_ids')
    # GPU-resident/custom inputs keep the model's existing fallback path.
    if lengths is None or lengths.ndim != 2 or lengths.device.type != 'cpu':
        return inputs
    if positions is None or positions.device.type != 'cpu':
        return inputs
    slots = lengths.shape[1]
    boundaries, offset = [], 0
    for row in lengths.tolist():
        if any(n < 0 for n in row) or sum(row) > slots:
            raise ValueError('invalid packed sample lengths')
        boundaries.append(offset)
        end = 0
        for n in row:
            if n > 0:
                end += n
                if end < slots:
                    boundaries.append(offset + end)
        offset += slots
    boundaries.append(offset)
    result = dict(inputs)
    result['seqlens'] = torch.tensor(boundaries, dtype=torch.int32)
    if include_scalars:
        result['batch_attention_metadata'] = (
            int(positions.max()) + 1,
            max(b - a for a, b in zip(boundaries, boundaries[1:])),
        )
    return result


def collate_attention_layout(items, *, include_scalars=False):
    return prepare_attention_layout(collate(items), include_scalars)


def create_collate_fn(attention_layout_cpu=False, attention_scalars_cpu=False):
    """The training collate function: plain collation, or with the attention layout prepared on the CPU."""
    if attention_layout_cpu:
        return partial(collate_attention_layout, include_scalars=bool(attention_scalars_cpu))
    return collate
