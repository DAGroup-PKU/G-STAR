import argparse
import json
import re
from pathlib import Path

import torch
from transformers import Trainer, TrainingArguments, set_seed

from .config import read_json, recipe
from .data import PackedDataset, collate, preflight_images, sources
from .model import forward_inputs, load, save_sidecars


def collate_attention_layout(items, *, include_scalars=False):
    from .batch_metadata import prepare_attention_layout
    return prepare_attention_layout(collate(items), include_scalars)


class ImageTrainer(Trainer):
    def __init__(self, *args, visual, roles, head_lr_scale=1.0, decay_mode="standard", **kwargs):
        super().__init__(*args, **kwargs)
        self.visual, self.roles, self.head_lr_scale = visual, roles, head_lr_scale
        self.decay_mode = decay_mode

    def _prepare_inputs(self, inputs):
        if getattr(self.model.config, "attention_layout_cpu", False):
            from .batch_metadata import prepare_attention_layout
            inputs = prepare_attention_layout(
                inputs, getattr(self.model.config, "attention_scalars_cpu", False))
        return super()._prepare_inputs(inputs)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        output = model(**forward_inputs(self.model, inputs, self.visual))
        return (output.loss, output) if return_outputs else output.loss

    def create_optimizer(self):
        if self.optimizer is None:
            decay = {name for name, _ in self.model.named_parameters()
                     if not re.search(r"embed_tokens|lm_head", name)} if self.decay_mode == "pretrain" else set(self.get_decay_parameter_names(self.model))
            groups = []
            for is_head in (False, True):
                for use_decay in (False, True):
                    params = [p for n, p in self.model.named_parameters() if p.requires_grad
                              and ("image_head" in n) == is_head and (n in decay) == use_decay]
                    if params:
                        groups.append({"params": params,
                                       "lr": self.args.learning_rate * (self.head_lr_scale if is_head else 1),
                                       "weight_decay": self.args.weight_decay if use_decay else 0.0})
            self.optimizer = torch.optim.AdamW(groups, betas=(self.args.adam_beta1, self.args.adam_beta2), eps=self.args.adam_epsilon)
        return self.optimizer

    def _save(self, output_dir=None, state_dict=None):
        super()._save(output_dir, state_dict)
        save_sidecars(output_dir or self.args.output_dir, self.processing_class, self.roles)


def main():
    parser = argparse.ArgumentParser(description="Pretrain/SFT with frozen G-STAR and AR/DSpark heads")
    parser.add_argument("--config", required=True)
    parser.add_argument("--model", required=True, help="Initialized HF directory or pretrain checkpoint for SFT")
    parser.add_argument("--data")
    parser.add_argument("--output")
    parser.add_argument("--visual-tokenizer")
    parser.add_argument("--resume", help="Full Trainer checkpoint, including optimizer/scheduler/RNG")
    parser.add_argument("--max-steps", type=int)
    args = parser.parse_args()
    cfg = recipe(args.config)
    for key, value in (("data", args.data), ("output_dir", args.output),
                       ("visual_tokenizer", args.visual_tokenizer)):
        if value:
            cfg[key] = value
    if not cfg.get("visual_tokenizer"):
        parser.error("Set the frozen visual tokenizer with --visual-tokenizer or in the recipe")
    if args.max_steps is not None:
        cfg["training"]["max_steps"] = args.max_steps
    preflight_images(sources(cfg["data"]))
    set_seed(cfg.get("seed", 42))
    model, tokenizer, processor, visual = load(args.model, cfg["visual_tokenizer"],
                                             training=True, max_length=cfg["max_length"])
    model.config.use_cache = False
    model.config.attention_metadata_once = bool(cfg.get("attention_metadata_once", False))
    model.config.attention_layout_cpu = bool(cfg.get("attention_layout_cpu", False))
    model.config.attention_scalars_cpu = bool(cfg.get("attention_scalars_cpu", False))
    model.config.gstar_training_stage = cfg["stage"]
    pattern = re.compile(r"visual_bridge|embed_tokens|lm_head|^model\.layers|^model\.norm|image_head|model\.visual_embed_layers")
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(bool(pattern.search(name)))
    roles_path = str(Path(args.model) / "role_config.json")
    if not Path(roles_path).is_file():
        roles_path = cfg["roles"]
    dataset = PackedDataset(cfg["data"], processor, cfg["stage"], cfg["max_length"],
                            read_json(roles_path), cfg.get("seed", 42), cfg.get("packing", True))
    training = dict(cfg["training"])
    training.update(output_dir=cfg["output_dir"], remove_unused_columns=False,
                    dataloader_num_workers=0, dataloader_pin_memory=False,
                    seed=cfg.get("seed", 42), ddp_find_unused_parameters=False,
                    accelerator_config={"dispatch_batches": False, "split_batches": False})
    if cfg.get("deepspeed"):
        training["deepspeed"] = cfg["deepspeed"]
    data_collator = collate
    if cfg.get("attention_layout_cpu", False):
        from functools import partial
        data_collator = partial(collate_attention_layout,
                                include_scalars=bool(cfg.get("attention_scalars_cpu", False)))
    trainer = ImageTrainer(model=model, args=TrainingArguments(**training),
                           train_dataset=dataset, data_collator=data_collator,
                           processing_class=tokenizer, visual=visual,
                           roles=roles_path if cfg["stage"] == "sft" else None,
                           head_lr_scale=cfg.get("head_lr_scale", 1.0), decay_mode=cfg["stage"])
    if cfg.get("evaluation", {}).get("every"):
        from .evaluation import ImageSamples
        trainer.add_callback(ImageSamples(tokenizer, visual, cfg["evaluation"]))
    if trainer.is_world_process_zero():
        Path(cfg["output_dir"]).mkdir(parents=True, exist_ok=True)
        (Path(cfg["output_dir"]) / "recipe.json").write_text(json.dumps(cfg, indent=2) + "\n")
    trainer.train(resume_from_checkpoint=args.resume)
    trainer.save_model(str(Path(cfg["output_dir"]) / "final"))


if __name__ == "__main__":
    main()
