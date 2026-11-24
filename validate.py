#!/usr/bin/env python3
""" G-STAR Validation Script

The teacher-forced loss of a checkpoint on held-out data: the same packed sequences and loss as training (image-token
and text-token cross entropy), without parameter updates. Reports the token-weighted mean loss over the evaluated
sequences and writes it as JSON.

    python validate.py --model outputs/pretrain_dspark/checkpoint-1000 --visual-tokenizer /path/to/tokenizer \\
        --data /path/to/val_manifest.json --stage pretrain --num-batches 100 --results-file val.json

Several checkpoints: give ``--model`` more than once; each is evaluated on the same sequences.
"""
import argparse
import json
import math
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from gstar.data import collate, create_dataset, preflight_images, sources
from gstar.models import forward_inputs, load, resolve_pretrained
from gstar.utils.config import ROOT, read_json

parser = argparse.ArgumentParser(description="G-STAR validation loss")
parser.add_argument("--model", required=True, action="append", metavar="DIR",
                    help="checkpoint folder or hf-hub name (repeat for several)")
parser.add_argument("--visual-tokenizer", required=True, metavar="DIR", help="frozen visual tokenizer / refiner")
parser.add_argument("--data", required=True, metavar="MANIFEST", help="validation JSONL or weighted manifest")
parser.add_argument("--stage", choices=("pretrain", "sft"), default="pretrain")
parser.add_argument("--roles", default=str(ROOT / "configs/roles.json"), metavar="FILE")
parser.add_argument("--max-length", type=int, default=8192)
parser.add_argument("--packing", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("-b", "--batch-size", type=int, default=1)
parser.add_argument("--num-batches", type=int, default=50, help="batches evaluated (the stream repeats the data)")
parser.add_argument("--seed", type=int, default=42, help="data order")
parser.add_argument("--results-file", default="", metavar="FILE", help="write the results as JSON")


def to_device(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, (list, tuple)):
        return type(value)(to_device(v, device) for v in value)
    if isinstance(value, dict):
        return {k: to_device(v, device) for k, v in value.items()}
    return value


@torch.inference_mode()
def validate(model_path, args):
    model, tokenizer, processor, visual = load(resolve_pretrained(model_path), args.visual_tokenizer,
                                               training=True, max_length=args.max_length)
    model = model.cuda().eval()
    model.config.use_cache = False
    roles_path = Path(resolve_pretrained(model_path)) / "role_config.json"
    roles = read_json(roles_path if roles_path.is_file() else args.roles)
    dataset = create_dataset(args.data, processor, args.stage, args.max_length, roles, args.seed, args.packing)
    loader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=collate, num_workers=0)
    total, tokens, losses = 0.0, 0, []
    start = time.perf_counter()
    for index, batch in enumerate(loader):
        if index >= args.num_batches:
            break
        batch = to_device(batch, "cuda")
        loss = model(**forward_inputs(model, batch, visual)).loss.float().item()
        count = int((batch["labels"][:, 1:] != -100).sum())
        total += loss * count
        tokens += count
        losses.append(loss)
    mean = total / max(tokens, 1)
    result = {"model": str(model_path), "batches": len(losses), "tokens": tokens, "loss": round(mean, 6),
              "perplexity": round(math.exp(mean), 4) if mean < 50 else None,
              "batch_losses": [round(x, 6) for x in losses], "seconds": round(time.perf_counter() - start, 2)}
    del model
    torch.cuda.empty_cache()
    return result


def main():
    args = parser.parse_args()
    args.visual_tokenizer = resolve_pretrained(args.visual_tokenizer)
    preflight_images(sources(args.data))
    results = []
    for model_path in args.model:
        result = validate(model_path, args)
        print(json.dumps({k: v for k, v in result.items() if k != "batch_losses"}), flush=True)
        results.append(result)
    if args.results_file:
        Path(args.results_file).parent.mkdir(parents=True, exist_ok=True)
        Path(args.results_file).write_text(json.dumps(results, indent=1) + "\n")


if __name__ == "__main__":
    main()
