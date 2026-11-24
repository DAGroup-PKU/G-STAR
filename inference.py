#!/usr/bin/env python3
""" G-STAR Inference Script

Text-to-image generation, instruction-based image editing, image-to-text and text generation from a checkpoint and its
matching visual refiner. ``--prompt-file`` generates one image per line (or per JSONL ``prompt`` field) with seeds
``seed + index``, loading the model once.

    python inference.py --task t2i --model /path/to/G-star --visual-tokenizer /path/to/oneStep-SR2x_refiner \\
        --prompt 'A golden retriever ...' --height 1024 --width 1024 --cfg 2.4 \\
        --top-k 4096 --top-p 0.75 --temperature 0.7 --seed 20260915 --output outputs/generated_2048.png

``--model`` and ``--visual-tokenizer`` take local folders or ``hf-hub:<repo>[/<subfolder>]`` names, e.g.
``hf-hub:DAGroup-PKU/G-star`` and ``hf-hub:DAGroup-PKU/G-star/oneStep-SR2x_refiner``.
"""
import argparse
import json
import subprocess
import time
from pathlib import Path

import torch

from gstar.models import forward_inputs, load, resolve_pretrained
from gstar.utils.config import ROOT, read_json

parser = argparse.ArgumentParser(description="AR/DSpark image generation, editing and image-to-text inference")
group = parser.add_argument_group("Model")
group.add_argument("--model", required=True, help="checkpoint folder or hf-hub:<repo>[/<subfolder>]")
group.add_argument("--visual-tokenizer", required=True, help="matching refiner / tokenizer folder or hf-hub name")
group = parser.add_argument_group("Task")
group.add_argument("--task", choices=("t2i", "edit", "i2t", "text"), default="t2i")
group.add_argument("--prompt", help="prompt, edit instruction or question")
group.add_argument("--prompt-file", metavar="FILE",
                   help="t2i: one prompt per line, or JSONL with a 'prompt' field; images go to --output as a folder")
group.add_argument("--prompt-mode", choices=("auto", "pretrain", "sft"), default="auto",
                   help="T2I prompt template; auto uses role_config.json when present")
group.add_argument("--image", help="Local source image for editing, or image descriptor for i2t")
group.add_argument("--output", default="outputs/sample.png", help="output image (a folder with --prompt-file)")
group = parser.add_argument_group("Sampling")
group.add_argument("--height", type=int, default=256)
group.add_argument("--width", type=int, default=256)
group.add_argument("--cfg", type=float, default=1.0, help="Text guidance scale")
group.add_argument("--image-cfg", type=float, default=1.0, help="Image guidance scale for editing")
group.add_argument("--top-k", type=int, default=200)
group.add_argument("--top-p", type=float, default=1.0)
group.add_argument("--temperature", type=float, default=1.0)
group.add_argument("--max-new-tokens", type=int, default=512)
group.add_argument("--seed", type=int, default=42)
group = parser.add_argument_group("Prompt enhancement")
group.add_argument("--prompt-enhance", action="store_true",
                   help="Rewrite the t2i prompt / edit instruction with Qwen-Image-2.1-PE before generation")
group.add_argument("--pe-model", help="Override the PE model id or local path")
group.add_argument("--pe-python",
                   help="Python of a separate env (transformers>=5.4) to run PE in; default runs PE in-process")


def enhance(args):
    if not args.pe_python:
        from gstar.generation.prompt_enhance import enhance_prompt
        return enhance_prompt(args.prompt, args.task, args.image, args.pe_model, args.seed)
    command = [args.pe_python, "-m", "gstar.prompt_enhance", "--task", args.task,
               "--prompt", args.prompt, "--seed", str(args.seed)]
    if args.image:
        command += ["--image", str(Path(args.image).resolve())]
    if args.pe_model:
        command += ["--pe-model", args.pe_model]
    output = subprocess.run(command, cwd=ROOT, check=True, capture_output=True, text=True).stdout
    return json.loads(output.strip().splitlines()[-1])["rewritten_prompt"]


def read_prompts(path):
    prompts = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            prompts.append(json.loads(line)["prompt"] if line.lstrip().startswith("{") else line.strip())
    if not prompts:
        raise ValueError("No prompts in " + str(path))
    return prompts


def t2i_role_kwargs(args):
    # SFT uses explicit role tokens; carry those into the generation prefix.
    roles_file = Path(args.model) / "role_config.json"
    role_kwargs = {}
    if args.prompt_mode == "sft" and not roles_file.is_file():
        raise ValueError("SFT generation requires role_config.json in the model directory")
    if args.prompt_mode != "pretrain" and roles_file.is_file():
        roles = read_json(roles_file)
        role_kwargs = {"user_token_id": roles["user"], "assistant_token_id": roles["assistant"]}
    return role_kwargs


@torch.inference_mode()
def main():
    args = parser.parse_args()
    if (args.prompt is None) == (args.prompt_file is None):
        parser.error("give exactly one of --prompt and --prompt-file")
    if args.prompt_file and (args.task != "t2i" or args.prompt_enhance):
        parser.error("--prompt-file supports t2i without --prompt-enhance")
    if args.task in ("edit", "i2t") and not args.image:
        parser.error(args.task + " requires --image")
    if args.prompt_enhance:
        if args.task not in ("t2i", "edit"):
            parser.error("--prompt-enhance supports t2i and edit only")
        args.prompt = enhance(args)
        print("Enhanced prompt:", args.prompt)
    args.model = resolve_pretrained(args.model)
    args.visual_tokenizer = resolve_pretrained(args.visual_tokenizer)
    torch.manual_seed(args.seed)
    model, tokenizer, processor, visual = load(args.model, args.visual_tokenizer)
    model = model.cuda().eval()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    if args.task == "t2i":
        factor = int(model.config.visual_config.patch_size) * int(model.config.visual_config.merge_size) * int(getattr(model.config.visual_config, "merger_merge_size", 1))
        if args.height % factor or args.width % factor:
            raise ValueError("Image dimensions must be divisible by " + str(factor))
        from gstar.vendor.generation import generate_t2i_batch
        role_kwargs = t2i_role_kwargs(args)

        def generate(prompt):
            return generate_t2i_batch(model, tokenizer, [prompt], image_height=args.height,
                                      image_width=args.width, cfg_scale=args.cfg, top_k=args.top_k,
                                      top_p=args.top_p, temperature=args.temperature,
                                      visual_tokenizer=visual, **role_kwargs)[0]

        if args.prompt_file:
            folder = Path(args.output)
            folder.mkdir(parents=True, exist_ok=True)
            with (folder / "results.jsonl").open("w", encoding="utf-8") as results:
                for index, prompt in enumerate(read_prompts(args.prompt_file)):
                    seed = args.seed + index
                    torch.manual_seed(seed)
                    torch.cuda.manual_seed_all(seed)
                    start = time.perf_counter()
                    image = generate(prompt)
                    seconds = time.perf_counter() - start
                    name = "%05d.png" % index
                    image.save(folder / name)
                    results.write(json.dumps({"index": index, "prompt": prompt, "seed": seed, "image": name,
                                              "seconds": round(seconds, 3)}, ensure_ascii=False) + "\n")
                    print(folder / name, "%.2fs" % seconds, flush=True)
            return
        image = generate(args.prompt)
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        image.save(args.output)
        print(args.output)
    elif args.task == "edit":
        from gstar.generation.edit import generate_edit
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
