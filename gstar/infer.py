import argparse
from pathlib import Path

import torch

from .config import ROOT, read_json
from .model import forward_inputs, load


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description="AR/DSpark image generation, editing and image-to-text inference")
    parser.add_argument("--model", required=True)
    parser.add_argument("--visual-tokenizer", required=True)
    parser.add_argument("--task", choices=("t2i", "edit", "i2t", "text"), default="t2i")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--prompt-mode", choices=("auto", "pretrain", "sft"), default="auto",
                        help="T2I prompt template; auto uses role_config.json when present")
    parser.add_argument("--image", help="Local source image for editing, or image descriptor for i2t")
    parser.add_argument("--output", default="outputs/sample.png")
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--cfg", type=float, default=1.0, help="Text guidance scale")
    parser.add_argument("--image-cfg", type=float, default=1.0, help="Image guidance scale for editing")
    parser.add_argument("--top-k", type=int, default=200)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.task in ("edit", "i2t") and not args.image:
        parser.error(args.task + " requires --image")
    torch.manual_seed(args.seed)
    model, tokenizer, processor, visual = load(args.model, args.visual_tokenizer)
    model = model.cuda().eval()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    if args.task == "t2i":
        factor = int(model.config.visual_config.patch_size) * int(model.config.visual_config.merge_size) * int(getattr(model.config.visual_config, "merger_merge_size", 1))
        if args.height % factor or args.width % factor:
            raise ValueError("Image dimensions must be divisible by " + str(factor))
        from .vendor.generation import generate_t2i_batch
        # SFT uses explicit role tokens; carry those into the generation prefix.
        roles_file = Path(args.model) / "role_config.json"
        role_kwargs = {}
        if args.prompt_mode == "sft" and not roles_file.is_file():
            raise ValueError("SFT generation requires role_config.json in the model directory")
        if args.prompt_mode != "pretrain" and roles_file.is_file():
            roles = read_json(roles_file)
            role_kwargs = {"user_token_id": roles["user"], "assistant_token_id": roles["assistant"]}
        image = generate_t2i_batch(model, tokenizer, [args.prompt], image_height=args.height,
                                   image_width=args.width, cfg_scale=args.cfg, top_k=args.top_k,
                                   top_p=args.top_p, temperature=args.temperature,
                                   visual_tokenizer=visual, **role_kwargs)[0]
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        image.save(args.output)
        print(args.output)
    elif args.task == "edit":
        from .edit import generate_edit
        roles_file = Path(args.model) / "role_config.json"
        roles = read_json(roles_file if roles_file.is_file() else ROOT / "configs/roles.json")
        result = generate_edit(
            model, tokenizer, visual, args.image, args.prompt,
            user_token_id=roles["user"], assistant_token_id=roles["assistant"],
            height=args.height, width=args.width, text_cfg=args.cfg,
            image_cfg=args.image_cfg, top_k=args.top_k, top_p=args.top_p,
            temperature=args.temperature, seed=args.seed)
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        result["image"].save(args.output)
        print(args.output)
    else:
        text = args.prompt
        if args.image:
            text = "<img_start>" + args.image + "<img_end>" + text
        roles_file = Path(args.model) / "role_config.json"
        roles = read_json(roles_file if roles_file.is_file() else ROOT / "configs/roles.json")
        text = tokenizer.convert_ids_to_tokens(roles["user"]) + text + tokenizer.convert_ids_to_tokens(roles["assistant"])
        batch = processor([text])
        batch = {k: (v.cuda() if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
        kwargs = forward_inputs(model, batch, visual)
        kwargs.pop("labels", None)
        output = model.generate(**kwargs, max_new_tokens=args.max_new_tokens,
                                do_sample=True, temperature=args.temperature,
                                top_k=args.top_k, top_p=args.top_p, use_cache=True)
        print(tokenizer.decode(output[0, batch["input_ids"].shape[1]:], skip_special_tokens=True))


if __name__ == "__main__":
    main()
