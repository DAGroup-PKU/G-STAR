"""G-STAR architectures: a Qwen3 backbone, the visual embedding bridge and an AR or DSpark image head.

Each architecture is a JSON config in ``gstar/models/configs``; ``model_config(name)`` returns it (validated, with an
optional training resolution), and the registered entry points build a model of it. The released Hugging Face
checkpoints carry their own ``config.json`` and are loaded with ``create_model('hf-hub:...')`` or a local path.
"""
from pathlib import Path

import torch

from ..utils.config import read_json, validate_geometry
from ._registry import register_model

CONFIG_DIR = Path(__file__).resolve().parent / "configs"

__all__ = ["model_config", "gstar_ar", "gstar_dspark", "gstar_dspark64"]


def model_config(name_or_path, resolution=None, anyres=False):
    """The config dict of a registered architecture (or of a config JSON file), validated; ``resolution`` sets a fixed
    training resolution, ``anyres`` a variable aspect ratio within that pixel budget."""
    path = Path(name_or_path)
    if not path.is_file():
        path = CONFIG_DIR / (Path(name_or_path).stem + ".json")
        if not path.is_file():
            raise FileNotFoundError("No architecture config for " + str(name_or_path))
    config = validate_geometry(read_json(path))
    if resolution:
        visual = config["visual_config"]
        factor = visual["patch_size"] * visual["merge_size"] * visual.get("merger_merge_size", 1)
        if resolution < 256 or resolution % factor:
            raise ValueError("Resolution must be >=256 and divisible by " + str(factor))
        visual.update(image_size=resolution, force_fix_res=0 if anyres else resolution,
                      min_pixels=65536 if anyres else resolution ** 2,
                      max_pixels=resolution ** 2)
    return config


def _create(name, resolution=None, anyres=False, **overrides):
    from ..vendor.model.configuration_omni import OmniConfig
    from ..vendor.model.modeling_omni import OmniForCausalLM
    config = model_config(name, resolution, anyres)
    config.update(overrides)
    return OmniForCausalLM(OmniConfig(**config)).to(torch.bfloat16)


@register_model
def gstar_ar(**kwargs):
    """Qwen3-8B backbone, 16-codebook visual tokens, autoregressive image head."""
    return _create("gstar_ar", **kwargs)


@register_model
def gstar_dspark(**kwargs):
    """Qwen3-8B backbone, 16-codebook visual tokens (32x), semi-autoregressive DSpark head (G-STAR-Pro layout)."""
    return _create("gstar_dspark", **kwargs)


@register_model
def gstar_dspark64(**kwargs):
    """Qwen3-8B backbone, 2 x 2 spatial merge of the visual tokens (effective 64x), DSpark head over 64 groups."""
    return _create("gstar_dspark64", **kwargs)
