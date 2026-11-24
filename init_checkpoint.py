#!/usr/bin/env python3
""" G-STAR Checkpoint Initialization

A new training checkpoint of a registered architecture (``python -c "import gstar; print(gstar.list_models())"``):
the Qwen backbone's weights and a randomly initialized visual bridge and image head; or an existing checkpoint with a
fresh image head of another architecture (same backbone, bridge, codebooks and spatial merge).

    python init_checkpoint.py --model gstar_dspark --llm /path/to/Qwen3-8B-Base --output outputs/init_dspark
    python init_checkpoint.py --model gstar_dspark --source-model /path/to/ar_checkpoint --output outputs/init_dspark
"""
import argparse

from gstar.models import initialize, list_models, replace_head

parser = argparse.ArgumentParser(description="Initialize a Qwen backbone with a fresh bridge and image head")
arch = parser.add_mutually_exclusive_group(required=True)
arch.add_argument("--model", choices=list_models(), help="registered architecture")
arch.add_argument("--config", metavar="FILE", help="architecture config JSON (instead of --model)")
source = parser.add_mutually_exclusive_group(required=True)
source.add_argument("--llm", help="Qwen backbone safetensors directory")
source.add_argument("--source-model", help="Keep a pretrained image-language model and initialize a fresh head")
parser.add_argument("--output", required=True)
parser.add_argument("--resolution", type=int, help="Override training image size: 256/512/1024")
parser.add_argument("--anyres", action="store_true", help="Variable aspect ratio within the resolution pixel budget")


def main():
    args = parser.parse_args()
    architecture = args.model or args.config
    if args.source_model:
        if args.resolution or args.anyres:
            parser.error("Head conversion preserves source geometry; resolution flags apply to --llm")
        replace_head(architecture, args.source_model, args.output)
    else:
        initialize(architecture, args.llm, args.output, args.resolution, args.anyres)


if __name__ == "__main__":
    main()
