"""Weighted streaming JSONL, actual multimodal processing, and safe packing."""
import glob
import json
import logging
import math
import random
import re
from dataclasses import fields
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import IterableDataset

from .vendor.model.processor_omni import OmniProcessorOutput

LOG = logging.getLogger(__name__)


def sources(manifest):
    path = Path(manifest).resolve()
    if path.suffix == ".jsonl":
        return [(path, 1.0)]
    import json5
    cfg = json5.loads(path.read_text())
    if "train_files" in cfg:
        spec = cfg["train_files"]
        root = Path(spec.get("root") or path.parent)
        if not root.is_absolute():
            root = path.parent / root
        cfg = spec["sample_rate"]
    else:
        root = path.parent
    result = []
    for pattern, weight in cfg.items():
        weight = float(weight)
        if not math.isfinite(weight) or weight < 0:
            raise ValueError("Invalid sampling weight: " + pattern)
        if weight == 0:
            continue
        files = sorted(glob.glob(str(root / pattern)))
        if not files:
            raise FileNotFoundError(pattern)
        for file in files:
            p = Path(file)
            candidates = sorted(p.rglob("*.jsonl")) if p.is_dir() else [p]
            if not candidates:
                raise ValueError("No JSONL files in " + file)
            result.extend((item, weight) for item in candidates if item.suffix == ".jsonl")
    if not result:
        raise ValueError("Manifest has no positive-weight JSONL sources")
    return result


def preflight_images(files):
    """Check first records before loading multi-billion-parameter weights."""
    for path, _ in files:
        with path.open(encoding="utf-8") as handle:
            line = handle.readline()
        if not line.strip():
            raise ValueError("Empty JSONL source: " + str(path))
        row = json.loads(line)
        text = row.get("input", row.get("content", ""))
        if isinstance(text, str):
            for descriptor in re.findall(r"<img_start>(.*?)<img_end>", text):
                if descriptor.startswith("{"):
                    info = json.loads(descriptor)
                    if info.get("type") == "file" and not Path(info["path"]).is_file():
                        raise FileNotFoundError("Local image unavailable: " + info["path"])


def weighted_rows(files, seed):
    rng = random.Random(seed)
    handles = [path.open(encoding="utf-8") for path, _ in files]
    active, buffer = list(range(len(files))), []
    try:
        while True:
            if not active:
                for row in rng.sample(buffer, len(buffer)):
                    yield row
                buffer.clear()
                active = list(range(len(files)))
                for handle in handles:
                    handle.seek(0)
            i = rng.choice(active)
            line = handles[i].readline()
            if not line:
                if handles[i].tell() == 0:
                    raise ValueError("Empty JSONL source: " + str(files[i][0]))
                active.remove(i)
                continue
            if line.strip():
                row = json.loads(line)
                rate = files[i][1]
                copies = int(rate) + int(rng.random() < rate % 1)
                for _ in range(copies):
                    buffer.append(row)
                    if len(buffer) >= 256:
                        yield buffer.pop(rng.randrange(len(buffer)))
    finally:
        for handle in handles:
            handle.close()


def content_text(content):
    if isinstance(content, str):
        return content
    parts = []
    for item in content:
        kind = item["type"]
        if kind == "text":
            parts.append(item["text"])
        elif kind in {"image", "image_url"}:
            value = item.get("text", item.get("image_url", item.get("image")))
            if isinstance(value, dict) and "url" in value:
                value = value["url"]
            if not isinstance(value, str):
                value = json.dumps(value)
            parts.append("<img_start>" + value + "<img_end>")
        else:
            raise ValueError("Unsupported message content: " + kind)
    return "".join(parts)


def conversation(row, processor, roles):
    output = OmniProcessorOutput(input_ids=[], labels=[])
    learnable = False
    messages = row.get("messages", row.get("conversations"))
    if not messages:
        raise ValueError("SFT expects messages or conversations")
    for msg in messages:
        role = msg.get("role", msg.get("from"))
        role = {"human": "user", "gpt": "assistant", "user_system": "system"}.get(role, role)
        output.input_ids.append(roles[role])
        output.labels.append(roles[role] if learnable else -100)
        text = content_text(msg.get("content", msg.get("value")))
        if not text:
            learnable = False
            continue
        item = processor.process_one(text)
        if not item.input_ids:
            raise ValueError("Multimodal message failed to load")
        learnable = msg.get("learnable", role == "assistant")
        item.labels = list(item.input_ids) if learnable else [-100] * len(item.input_ids)
        output = output.concatenate(item)
    if learnable:
        output.input_ids.append(processor.tokenizer.eos_token_id)
        output.labels.append(processor.tokenizer.eos_token_id)
    return output


def padded(item, tokenizer, max_length, lengths):
    if not lengths or any(n <= 0 for n in lengths) or sum(lengths) != len(item.input_ids):
        raise ValueError("packing lengths must partition the unpadded tokens")
    if len(item.labels) != len(item.input_ids) or len(item.input_ids) > max_length:
        raise ValueError("invalid packing token/label lengths")
    # A sample's first label is never predicted in an independent forward.
    # Mask internal starts so an EOS cannot predict the next sample's first token.
    offset = 0
    for n in lengths[:-1]:
        offset += n
        item.labels[offset] = -100
    pad = max_length - len(item.input_ids)
    item.position_ids = [p for n in lengths for p in range(n)] + [0] * pad
    item.seqlens = lengths + [0] * (max_length - len(lengths))
    item.input_ids += [tokenizer.pad_token_id] * pad
    item.labels += [-100] * pad
    return {field.name: getattr(item, field.name) for field in fields(item)
            if getattr(item, field.name) is not None}


class PackedDataset(IterableDataset):
    def __init__(self, manifest, processor, stage, max_length, roles, seed=42, packing=True):
        self.files = sources(manifest)
        self.processor, self.stage = processor, stage
        self.max_length, self.roles = max_length, roles
        self.seed, self.packing = seed, packing

    def __iter__(self):
        output = OmniProcessorOutput(input_ids=[], labels=[])
        lengths, failures = [], 0
        for row in weighted_rows(self.files, self.seed):
            try:
                if self.stage == "sft":
                    item = conversation(row, self.processor, self.roles)
                else:
                    text = (row.get("title") or "") + row.get("input", row.get("content", ""))
                    item = self.processor.process_one(text)
                    if item.input_ids:
                        item.input_ids.append(self.processor.tokenizer.eos_token_id)
                        item.labels.append(self.processor.tokenizer.eos_token_id if row.get("eosloss", True) else -100)
                if not item.input_ids or len(item.input_ids) > self.max_length or not any(x >= 0 for x in item.labels):
                    raise ValueError("Empty, overlength, or unsupervised sample")
            except (ValueError, OSError, KeyError) as error:
                failures += 1
                if failures <= 3 or failures % 100 == 0:
                    LOG.warning("Dropped sample (%d consecutive): %s", failures, error)
                if failures >= 1000:
                    raise RuntimeError("1000 consecutive unusable samples; inspect manifest/image mounts") from error
                continue
            failures = 0
            if lengths and len(output.input_ids) + len(item.input_ids) > self.max_length:
                yield padded(output, self.processor.tokenizer, self.max_length, lengths)
                output, lengths = OmniProcessorOutput(input_ids=[], labels=[]), []
            output = output.concatenate(item)
            lengths.append(len(item.input_ids))
            if not self.packing or len(output.input_ids) == self.max_length:
                yield padded(output, self.processor.tokenizer, self.max_length, lengths)
                output, lengths = OmniProcessorOutput(input_ids=[], labels=[]), []


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
