"""Paths, JSON helpers and the config-file handling shared by the entry-point scripts.

The scripts follow timm's convention: ``-c/--config`` names a YAML file whose keys are the argument names of the
script (dashes as underscores); its values become the parser defaults and command-line flags override them. The
nested JSON recipes of earlier versions (``{"stage": ..., "training": {...}}``) are still accepted and translated.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODEL_CODE = ROOT / "gstar/vendor/model"

# config-file keys whose relative paths are resolved against the repository root (as the JSON recipes always were)
PATH_KEYS = ("data", "roles", "deepspeed", "model_config")


def read_json(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def recipe(path):
    """A nested JSON recipe with its relative paths resolved against the repository root."""
    cfg = read_json(path)
    for key in ("model_config", "data", "roles", "deepspeed"):
        if cfg.get(key) and not Path(cfg[key]).is_absolute():
            cfg[key] = str((ROOT / cfg[key]).resolve())
    return cfg


def validate_geometry(config):
    visual = config["visual_config"]
    quant = config["visual_quantizer_config"]
    head = visual["image_head_config"]
    depth = quant["depth"]
    if len(quant["codebook_sizes"]) != depth:
        raise ValueError("codebook depth mismatch")
    mms = visual.get("merger_merge_size", 1)
    if head.get("image_head_type") == "dspark_attn":
        if head["image_head_group_count"] != depth * mms * mms:
            raise ValueError("DSpark group count must equal depth * merger_merge_size**2")
        if head["image_head_spatial_group_size"] != mms:
            raise ValueError("DSpark spatial group size mismatch")
    return config


# nested recipe key -> flat argument name of train.py
_RECIPE_TOP = {"stage": "stage", "data": "data", "roles": "roles", "visual_tokenizer": "visual_tokenizer",
               "output_dir": "output", "max_length": "max_length", "packing": "packing", "seed": "seed",
               "head_lr_scale": "head_lr_scale", "deepspeed": "deepspeed",
               "attention_metadata_once": "attention_metadata_once", "attention_layout_cpu": "attention_layout_cpu",
               "attention_scalars_cpu": "attention_scalars_cpu"}
_RECIPE_EVAL = {"every": "sample_every", "image_size": "sample_image_size", "cfg": "sample_cfg",
                "prompts": "sample_prompts"}
_RECIPE_TRAINING = {"max_steps": "max_steps", "per_device_train_batch_size": "batch_size",
                    "gradient_accumulation_steps": "grad_accum_steps", "learning_rate": "lr",
                    "adam_epsilon": "opt_eps", "weight_decay": "weight_decay", "warmup_steps": "warmup_steps",
                    "lr_scheduler_type": "sched", "bf16": "bf16", "gradient_checkpointing": "grad_checkpointing",
                    "logging_steps": "log_interval", "save_steps": "checkpoint_interval",
                    "save_total_limit": "checkpoint_hist", "save_safetensors": "save_safetensors",
                    "report_to": "log_to"}


def flatten_recipe(cfg):
    """Translate a nested JSON recipe into train.py's flat argument names (unknown TrainingArguments keys go to
    ``training_args``)."""
    flat, extra = {}, {}
    for key, value in cfg.items():
        if key in _RECIPE_TOP:
            flat[_RECIPE_TOP[key]] = value
        elif key == "evaluation":
            for k, v in value.items():
                flat[_RECIPE_EVAL[k]] = v
        elif key == "training":
            training = dict(value)
            if "adam_beta1" in training or "adam_beta2" in training:
                flat["opt_betas"] = [training.pop("adam_beta1", 0.9), training.pop("adam_beta2", 0.999)]
            kwargs = dict(training.pop("lr_scheduler_kwargs", None) or {})
            if "min_lr_rate" in kwargs:
                flat["min_lr_rate"] = kwargs.pop("min_lr_rate")
            if kwargs:
                flat["sched_kwargs"] = kwargs
            if training.get("gradient_checkpointing_kwargs") == {"use_reentrant": False}:
                training.pop("gradient_checkpointing_kwargs")
            for k, v in training.items():
                if k in _RECIPE_TRAINING:
                    flat[_RECIPE_TRAINING[k]] = v
                else:
                    extra[k] = v
        else:
            raise ValueError("Unknown recipe key: " + key)
    if extra:
        flat["training_args"] = extra
    return flat


def load_config_file(path):
    """The defaults in a config file: a flat YAML (or JSON) mapping of argument names, or a nested JSON recipe.
    Relative paths of PATH_KEYS are resolved against the repository root."""
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    if path.suffix in (".yaml", ".yml"):
        import yaml
        cfg = yaml.safe_load(text) or {}
    else:
        cfg = json.loads(text)
    if "training" in cfg and isinstance(cfg["training"], dict):
        cfg = flatten_recipe(cfg)
    cfg = {key.replace("-", "_"): value for key, value in cfg.items()}
    for key in PATH_KEYS:
        if cfg.get(key) and not Path(cfg[key]).is_absolute():
            cfg[key] = str((ROOT / cfg[key]).resolve())
    return cfg


def parse_args(config_parser, parser, argv=None):
    """timm's two-stage parsing: the config file gives the defaults, the command line overrides them. Returns the
    arguments and their YAML text (saved next to the outputs)."""
    import yaml
    args_config, remaining = config_parser.parse_known_args(argv)
    saved = ({action.dest: action.default for action in parser._actions}, dict(parser._defaults))
    try:
        if args_config.config:
            cfg = load_config_file(args_config.config)
            known = {action.dest for action in parser._actions}
            unknown = sorted(set(cfg) - known)
            if unknown:
                parser.error("Unknown keys in " + args_config.config + ": " + ", ".join(unknown))
            parser.set_defaults(**cfg)
        args = parser.parse_args(remaining)
    finally:  # the parser keeps its own defaults (several parses in one process)
        for action in parser._actions:
            action.default = saved[0][action.dest]
        parser._defaults.clear()
        parser._defaults.update(saved[1])
    args.config = args_config.config
    args_text = yaml.safe_dump(vars(args), default_flow_style=False, sort_keys=False)
    return args, args_text
