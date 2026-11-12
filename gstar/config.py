import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL_CODE = ROOT / "gstar/vendor/model"


def read_json(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def recipe(path):
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
