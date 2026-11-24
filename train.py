#!/usr/bin/env python3
""" G-STAR Training Script

Pretraining and SFT of the image-language model (Qwen3 backbone, visual embedding bridge, AR or DSpark image head)
on visual tokens from a frozen visual tokenizer. Launch with distributed_train.sh for several GPUs or nodes.

Arguments come from the command line or from a YAML config (``-c configs/pretrain.yaml``) whose keys are the
argument names below; flags given on the command line override the config. Nested JSON recipes of earlier versions
are also accepted by ``-c``.

    python train.py -c configs/pretrain.yaml --model outputs/init_dspark \\
        --visual-tokenizer /path/to/tokenizer --output outputs/pretrain_dspark
"""
import argparse
import json
import logging
from pathlib import Path

from transformers import TrainingArguments, set_seed

from gstar.data import create_collate_fn, create_dataset, preflight_images, sources
from gstar.models import load
from gstar.trainer import ImageSamples, ImageTrainer, set_trainable
from gstar.utils.config import ROOT, parse_args, read_json

_logger = logging.getLogger("train")

# The first parser, used only to find the config file; its defaults are applied to the main parser.
config_parser = parser = argparse.ArgumentParser(description="Training Config", add_help=False)
parser.add_argument("-c", "--config", default="", type=str, metavar="FILE",
                    help="YAML config file specifying default arguments")

parser = argparse.ArgumentParser(description="G-STAR pretraining and SFT")

# Dataset parameters
group = parser.add_argument_group("Dataset parameters")
group.add_argument("--data", metavar="MANIFEST",
                   help="JSONL file, or JSON/JSON5 manifest of weighted JSONL sources")
group.add_argument("--stage", choices=("pretrain", "sft"), default="pretrain",
                   help="pretrain: plain text with images; sft: role-tagged conversations")
group.add_argument("--roles", default=str(ROOT / "configs/roles.json"), metavar="FILE",
                   help="role token ids, used when the model directory has no role_config.json")
group.add_argument("--max-length", type=int, default=8192, metavar="N", help="tokens per packed sequence")
group.add_argument("--packing", action=argparse.BooleanOptionalAction, default=True,
                   help="pack several samples into one sequence")

# Model parameters
group = parser.add_argument_group("Model parameters")
group.add_argument("--model", required=True, metavar="DIR",
                   help="initialized checkpoint (init_checkpoint.py), or a pretrain checkpoint for SFT")
group.add_argument("--visual-tokenizer", metavar="DIR", help="frozen visual tokenizer (or its refiner export)")
group.add_argument("--resume", default=None, metavar="DIR",
                   help="full Trainer checkpoint to resume, with optimizer, scheduler and RNG state")
group.add_argument("--grad-checkpointing", action=argparse.BooleanOptionalAction, default=True,
                   help="activation checkpointing (non-reentrant)")
group.add_argument("--attention-metadata-once", action="store_true", default=False,
                   help="compute the packed attention metadata once per forward")
group.add_argument("--attention-layout-cpu", action="store_true", default=False,
                   help="prepare the packed attention layout on the CPU in the collate function")
group.add_argument("--attention-scalars-cpu", action="store_true", default=False,
                   help="with --attention-layout-cpu: also the scalar attention bounds")

# Optimizer parameters
group = parser.add_argument_group("Optimizer parameters")
group.add_argument("--lr", type=float, default=1e-4, metavar="LR", help="peak learning rate")
group.add_argument("--head-lr-scale", type=float, default=1.0, help="learning-rate multiplier of the image head")
group.add_argument("--opt-betas", type=float, nargs=2, default=[0.9, 0.999], metavar=("BETA1", "BETA2"),
                   help="AdamW betas")
group.add_argument("--opt-eps", type=float, default=None, metavar="EPS", help="AdamW epsilon (default 1e-8)")
group.add_argument("--weight-decay", type=float, default=0.0,
                   help="weight decay (pretrain: all but embeddings and LM head; sft: transformers' default set)")

# Learning rate schedule parameters
group = parser.add_argument_group("Learning rate schedule parameters")
group.add_argument("--sched", default="cosine", metavar="SCHEDULER",
                   help="transformers schedule name: cosine, cosine_with_min_lr, linear, constant_with_warmup ...")
group.add_argument("--min-lr-rate", type=float, default=None, help="cosine_with_min_lr: final LR as a fraction of --lr")
group.add_argument("--sched-kwargs", type=json.loads, default=None, metavar="JSON",
                   help="other keyword arguments of the schedule")
group.add_argument("--warmup-steps", type=int, default=0, metavar="N", help="linear warmup steps")
group.add_argument("--max-steps", type=int, default=None, metavar="N", help="optimizer steps to train")

# Training parameters
group = parser.add_argument_group("Training parameters")
group.add_argument("-b", "--batch-size", type=int, default=1, metavar="N", help="packed sequences per GPU and step")
group.add_argument("--grad-accum-steps", type=int, default=1, metavar="N", help="gradient accumulation steps")
group.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True, help="bf16 mixed precision")
group.add_argument("--deepspeed", default=None, metavar="FILE", help="DeepSpeed config, e.g. configs/deepspeed.json (ZeRO-2)")
group.add_argument("--seed", type=int, default=42, help="random seed (data order and initialization)")
group.add_argument("--training-args", type=json.loads, default=None, metavar="JSON",
                   help="further transformers.TrainingArguments as a JSON object")

# Logging, checkpoints and sampling
group = parser.add_argument_group("Logging, checkpoint and sampling parameters")
group.add_argument("--output", default="outputs/train", metavar="DIR", help="output directory")
group.add_argument("--log-interval", type=int, default=10, metavar="N", help="steps between log lines")
group.add_argument("--log-to", default="tensorboard", help="transformers report_to backend(s), or none")
group.add_argument("--checkpoint-interval", type=int, default=1000, metavar="N", help="steps between checkpoints")
group.add_argument("--checkpoint-hist", type=int, default=3, metavar="N", help="checkpoints kept")
group.add_argument("--save-safetensors", action=argparse.BooleanOptionalAction, default=False,
                   help="save checkpoints as safetensors")
group.add_argument("--sample-every", type=int, default=0, metavar="N",
                   help="rank zero writes sample images to <output>/samples/step-N every N steps (0: off)")
group.add_argument("--sample-image-size", type=int, default=256, help="sample image size")
group.add_argument("--sample-cfg", type=float, default=1.0, help="sample guidance scale")
group.add_argument("--sample-prompts", nargs="+", default=["A red apple on a wooden table."], metavar="PROMPT",
                   help="sample prompts")


def training_arguments(args):
    """The transformers.TrainingArguments keywords of the run."""
    training = dict(
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum_steps,
        learning_rate=args.lr,
        adam_beta1=args.opt_betas[0],
        adam_beta2=args.opt_betas[1],
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        lr_scheduler_type=args.sched,
        bf16=args.bf16,
        gradient_checkpointing=args.grad_checkpointing,
        logging_steps=args.log_interval,
        save_steps=args.checkpoint_interval,
        save_total_limit=args.checkpoint_hist,
        save_safetensors=args.save_safetensors,
        report_to=args.log_to,
    )
    if args.opt_eps is not None:
        training["adam_epsilon"] = args.opt_eps
    sched_kwargs = dict(args.sched_kwargs or {})
    if args.min_lr_rate is not None:
        sched_kwargs["min_lr_rate"] = args.min_lr_rate
    if sched_kwargs:
        training["lr_scheduler_kwargs"] = sched_kwargs
    if args.grad_checkpointing:
        training["gradient_checkpointing_kwargs"] = {"use_reentrant": False}
    training.update(args.training_args or {})
    training.update(output_dir=args.output, remove_unused_columns=False,
                    dataloader_num_workers=0, dataloader_pin_memory=False,
                    seed=args.seed, ddp_find_unused_parameters=False,
                    accelerator_config={"dispatch_batches": False, "split_batches": False})
    if args.deepspeed:
        training["deepspeed"] = args.deepspeed
    return training


def main():
    logging.basicConfig(level=logging.INFO)
    args, args_text = parse_args(config_parser, parser)
    if not args.data:
        parser.error("Set the training data with --data or in the config")
    if not args.visual_tokenizer:
        parser.error("Set the frozen visual tokenizer with --visual-tokenizer or in the config")
    if args.max_steps is None:
        parser.error("Set --max-steps or max_steps in the config")

    preflight_images(sources(args.data))
    set_seed(args.seed)
    model, tokenizer, processor, visual = load(args.model, args.visual_tokenizer,
                                             training=True, max_length=args.max_length)
    model.config.use_cache = False
    model.config.attention_metadata_once = bool(args.attention_metadata_once)
    model.config.attention_layout_cpu = bool(args.attention_layout_cpu)
    model.config.attention_scalars_cpu = bool(args.attention_scalars_cpu)
    model.config.gstar_training_stage = args.stage
    set_trainable(model)
    roles_path = str(Path(args.model) / "role_config.json")
    if not Path(roles_path).is_file():
        roles_path = args.roles
    dataset = create_dataset(args.data, processor, args.stage, args.max_length,
                             read_json(roles_path), args.seed, args.packing)
    trainer = ImageTrainer(model=model, args=TrainingArguments(**training_arguments(args)),
                           train_dataset=dataset,
                           data_collator=create_collate_fn(args.attention_layout_cpu, args.attention_scalars_cpu),
                           processing_class=tokenizer, visual=visual,
                           roles=roles_path if args.stage == "sft" else None,
                           head_lr_scale=args.head_lr_scale, decay_mode=args.stage)
    if args.sample_every:
        trainer.add_callback(ImageSamples(tokenizer, visual, {
            "every": args.sample_every, "image_size": args.sample_image_size, "cfg": args.sample_cfg,
            "prompts": list(args.sample_prompts)}))
    if trainer.is_world_process_zero():
        Path(args.output).mkdir(parents=True, exist_ok=True)
        (Path(args.output) / "args.yaml").write_text(args_text)
    trainer.train(resume_from_checkpoint=args.resume)
    trainer.save_model(str(Path(args.output) / "final"))


if __name__ == "__main__":
    main()
