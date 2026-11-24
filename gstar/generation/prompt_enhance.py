"""Optional prompt enhancement with the Qwen-Image-2.1 PE (prompt rewriting) models.

The PE models need transformers>=5.4, newer than the version pinned for G-STAR.
Run them in-process when the active environment supports them, or from a second
environment via ``python -m gstar.prompt_enhance`` (see ``--pe-python`` in infer).
"""
import argparse
import gc
import json

import torch

PE_MODELS = {"t2i": "Qwen/Qwen-Image-2.1-PE-T2I", "edit": "Qwen/Qwen-Image-2.1-PE-I2I"}
SAMPLING = dict(do_sample=True, temperature=1.0, top_p=0.95, top_k=20)


def _system_prompt(model_id):
    import huggingface_hub
    with open(huggingface_hub.hf_hub_download(model_id, "system_prompt.txt")) as f:
        return f.read().strip()


def _parse(generated):
    # The answer is a JSON object after the </think> reasoning block.
    _, _, answer = generated.partition("</think>")
    return json.loads((answer or generated).strip())["rewritten_prompt"].strip()


@torch.inference_mode()
def enhance_prompt(prompt, task="t2i", image=None, model_id=None, seed=42):
    """Return the PE-rewritten prompt for a t2i prompt or an edit instruction."""
    if task not in PE_MODELS:
        raise ValueError("Prompt enhancement supports t2i and edit only")
    model_id = model_id or PE_MODELS[task]
    system_prompt = _system_prompt(model_id)
    torch.manual_seed(seed)
    if task == "t2i":
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16, device_map="auto").eval()
        text = tokenizer.apply_chat_template(
            [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}],
            tokenize=False, add_generation_prompt=True, enable_thinking=True)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        output = model.generate(**inputs, max_new_tokens=16256, **SAMPLING)
    else:
        from PIL import Image
        from transformers import AutoModelForImageTextToText, AutoProcessor
        processor = AutoProcessor.from_pretrained(model_id)
        tokenizer = processor.tokenizer
        model = AutoModelForImageTextToText.from_pretrained(model_id, dtype=torch.bfloat16, device_map="auto").eval()
        messages = [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {"role": "user", "content": [{"type": "image", "image": Image.open(image).convert("RGB")},
                                         {"type": "text", "text": prompt}]},
        ]
        inputs = processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True, return_dict=True,
            return_tensors="pt", enable_thinking=True).to(model.device)
        output = model.generate(**inputs, max_new_tokens=24000, **SAMPLING)
    generated = tokenizer.decode(output[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    # Release the PE model before G-STAR is loaded on the same GPU.
    del model, output, inputs
    gc.collect()
    torch.cuda.empty_cache()
    return _parse(generated)


def main():
    parser = argparse.ArgumentParser(description="Rewrite a prompt with Qwen-Image-2.1-PE")
    parser.add_argument("--task", choices=tuple(PE_MODELS), default="t2i")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--image", help="Source image for edit prompts")
    parser.add_argument("--pe-model", help="Override the PE model id or local path")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.task == "edit" and not args.image:
        parser.error("edit requires --image")
    print(json.dumps({"rewritten_prompt": enhance_prompt(args.prompt, args.task, args.image, args.pe_model, args.seed)},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
