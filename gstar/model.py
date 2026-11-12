import inspect
import os
import shutil
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import MODEL_CODE, read_json, validate_geometry
from .vendor.visual_tokenizer import encode_images_to_vision_tokens, get_reorder_fn, load_frozen_visual_tokenizer


def load(model_path, visual_path, training=False, max_length=8192):
    # Load the implementation shipped with this repository, including when the
    # checkpoint carries older remote-code sidecars.
    from .vendor.model.configuration_omni import OmniConfig
    from .vendor.model.modeling_omni import OmniForCausalLM
    config = OmniConfig(**validate_geometry(read_json(Path(model_path) / "config.json")))
    model, loading = OmniForCausalLM.from_pretrained(
        model_path, config=config, torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True, output_loading_info=True)
    problems = {key: loading[key] for key in
                ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")
                if loading.get(key)}
    if problems:
        raise ValueError("Checkpoint does not match the G-STAR implementation: " + str(problems))
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True,
                                             model_max_length=max_length)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = model.config.pad_token_id
    from .vendor.model.processor_omni import OmniMMProcessor
    processor = OmniMMProcessor(tokenizer, model.config, training=training, relative_path="")
    model.processor = processor
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    visual = load_frozen_visual_tokenizer(visual_path, model.config, torch.device("cuda", local_rank))
    return model, tokenizer, processor, visual


def forward_inputs(model, batch, visual):
    accepted = inspect.signature(model.forward).parameters
    kwargs = {key: value for key, value in batch.items() if key in accepted and value is not None}
    if kwargs.get("vision_tokens") is None and batch.get("images"):
        mms = int(getattr(model.config.visual_config, "merger_merge_size", 1) or 1)
        reorder = get_reorder_fn(model)
        if mms > 1 and reorder is None:
            raise RuntimeError("Spatial merge requires the block-major reorder implementation")
        kwargs["vision_tokens"] = encode_images_to_vision_tokens(
            visual, batch["images"], batch["images_grid"], mms=mms, reorder_fn=reorder)
    return kwargs


def save_sidecars(output, tokenizer, roles=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    for src in MODEL_CODE.glob("*.py"):
        shutil.copy2(src, output / src.name)
    tokenizer.save_pretrained(output)
    if roles:
        shutil.copy2(roles, output / "role_config.json")


def check_new_output(output):
    path = Path(output)
    if path.exists() and any(path.iterdir()):
        raise ValueError("Initialization output must be new or empty: " + str(path))


def initialize(config_path, llm_path, output, resolution=None, anyres=False):
    check_new_output(output)
    from safetensors.torch import load_file
    from .vendor.model.configuration_omni import OmniConfig
    from .vendor.model.modeling_omni import OmniForCausalLM
    config = validate_geometry(read_json(config_path))
    if resolution:
        visual = config["visual_config"]
        factor = visual["patch_size"] * visual["merge_size"] * visual.get("merger_merge_size", 1)
        if resolution < 256 or resolution % factor:
            raise ValueError("Resolution must be >=256 and divisible by " + str(factor))
        visual.update(image_size=resolution, force_fix_res=0 if anyres else resolution,
                      min_pixels=65536 if anyres else resolution ** 2,
                      max_pixels=resolution ** 2)
    model = OmniForCausalLM(OmniConfig(**config)).to(torch.bfloat16)
    target = model.state_dict()
    loaded = set()
    shards = sorted(Path(llm_path).glob("*.safetensors"))
    if not shards:
        raise FileNotFoundError("Qwen backbone requires safetensors shards: " + llm_path)
    for shard in shards:
        state = load_file(str(shard))
        for key, value in list(state.items()):
            if key not in target:
                raise ValueError("Unexpected backbone parameter: " + key)
            if value.shape != target[key].shape:
                raise ValueError("Backbone shape mismatch: " + key)
        model.load_state_dict(state, strict=False)
        loaded.update(state)
    required = {key for key in target if key.startswith(("model.layers.", "model.norm.", "model.embed_tokens.", "lm_head."))}
    if config.get("tie_word_embeddings") and "model.embed_tokens.weight" in loaded:
        required.discard("lm_head.weight")
        model.tie_weights()
    if required - loaded:
        raise ValueError("Missing Qwen backbone keys: " + str(sorted(required - loaded)[:10]))
    model.save_pretrained(output, safe_serialization=False)
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_CODE), trust_remote_code=True)
    save_sidecars(output, tokenizer)


def replace_head(config_path, source, output):
    check_new_output(output)
    from .vendor.model.configuration_omni import OmniConfig
    from .vendor.model.image_head import OmniImageHead
    target = OmniConfig(**validate_geometry(read_json(config_path)))
    model = AutoModelForCausalLM.from_pretrained(source, trust_remote_code=True, torch_dtype=torch.bfloat16)
    if model.config.hidden_size != target.hidden_size:
        raise ValueError("Head conversion requires the same backbone hidden size")
    if model.config.visual_quantizer_config.codebook_sizes != target.visual_quantizer_config.codebook_sizes:
        raise ValueError("Head conversion requires matching visual codebooks")
    if model.config.visual_config.merger_merge_size != target.visual_config.merger_merge_size:
        raise ValueError("Changing spatial merge needs a fresh bridge; use --llm initialization")
    model.config.visual_config.image_head_config = target.visual_config.image_head_config
    model.image_head = OmniImageHead(model.config).to(torch.bfloat16)
    model.save_pretrained(output, safe_serialization=False)
    tokenizer = AutoTokenizer.from_pretrained(source, trust_remote_code=True)
    save_sidecars(output, tokenizer)
