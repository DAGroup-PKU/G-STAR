"""Batch-local attention layout preparation before the Trainer device transfer."""
import torch


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
