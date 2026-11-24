"""Training data: weighted JSONL sources, multimodal processing, packing and collation."""
from .collate import collate, collate_attention_layout, create_collate_fn, prepare_attention_layout
from .dataset import (PackedDataset, content_text, conversation, padded, preflight_images, sources,
                      weighted_rows)


def create_dataset(manifest, processor, stage, max_length, roles, seed=42, packing=True):
    """The packed, weighted training stream of a manifest (JSONL file or weighted JSON/JSON5 manifest)."""
    return PackedDataset(manifest, processor, stage, max_length, roles, seed, packing)
