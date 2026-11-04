import argparse
from .model import initialize, replace_head


def main():
    parser = argparse.ArgumentParser(description="Initialize Qwen backbone and fresh bridge/image head")
    parser.add_argument("--config", required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--llm", help="Qwen backbone safetensors directory")
    source.add_argument("--source-model", help="Keep a pretrained image-language model and initialize a fresh head")
    parser.add_argument("--output", required=True)
    parser.add_argument("--resolution", type=int, help="Override training image size: 256/512/1024")
    parser.add_argument("--anyres", action="store_true", help="Variable aspect ratio within the resolution pixel budget")
    args = parser.parse_args()
    if args.source_model:
        if args.resolution or args.anyres:
            parser.error("Head conversion preserves source geometry; resolution flags apply to --llm")
        replace_head(args.config, args.source_model, args.output)
    else:
        initialize(args.config, args.llm, args.output, args.resolution, args.anyres)


if __name__ == "__main__":
    main()
