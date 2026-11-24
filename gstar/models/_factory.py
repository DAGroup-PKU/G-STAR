"""create_model and the pretrained-checkpoint helpers (timm-style)."""
from pathlib import Path

from ._helpers import load_backbone, load_model
from ._registry import is_model, model_entrypoint

__all__ = ["create_model", "list_pretrained", "resolve_pretrained"]

HF_HUB_PREFIX = "hf-hub:"

# released checkpoints on the Hugging Face Hub and the refiner each one is paired with
_PRETRAINED = {
    "hf-hub:DAGroup-PKU/G-star": {
        "architecture": "gstar_dspark", "refiner": "hf-hub:DAGroup-PKU/G-star/oneStep-SR2x_refiner",
        "description": "G-STAR-Pro-8B text-to-image (pretrain prompts); refiner/ is the 52-step alternative"},
    "hf-hub:DAGroup-PKU/G-star/gstar-flash-1.7b": {
        "architecture": None, "refiner": "hf-hub:DAGroup-PKU/G-star/gstar-flash-1.7b-refiner",
        "description": "G-STAR-Flash-1.7B text-to-image (SFT prompts, effective 64x)"},
}
_DEFAULT_PRETRAINED = {"gstar_dspark": "hf-hub:DAGroup-PKU/G-star"}


def list_pretrained():
    """The released checkpoints: {name: {"architecture", "refiner", "description"}}."""
    return {name: dict(info) for name, info in _PRETRAINED.items()}


def resolve_pretrained(name):
    """A local directory for a checkpoint or refiner name: a local path is returned as is; ``hf-hub:<repo>[/<sub>]``
    is downloaded (only that folder) with huggingface_hub and its local path returned."""
    if not str(name).startswith(HF_HUB_PREFIX):
        return str(name)
    parts = str(name)[len(HF_HUB_PREFIX):].split("/")
    repo_id, sub = "/".join(parts[:2]), "/".join(parts[2:])
    from huggingface_hub import snapshot_download
    if sub:
        local = snapshot_download(repo_id, allow_patterns=[sub + "/*"])
        return str(Path(local) / sub)
    return snapshot_download(repo_id, ignore_patterns=["*/*"])


def create_model(model_name, pretrained=False, checkpoint_path=None, backbone=None, **kwargs):
    """Create a G-STAR model.

    ``model_name`` is
      - a registered architecture (``list_models()``): built from its config with random weights; ``backbone=<dir>``
        copies a Qwen backbone's safetensors weights in, ``checkpoint_path=<dir>`` loads a checkpoint of it, and
        ``pretrained=True`` loads its released checkpoint (``list_pretrained()``);
      - a checkpoint directory or ``hf-hub:<repo>[/<subfolder>]``: that checkpoint, on its own config.json.
    Checkpoints load strictly into this repository's implementation, in bf16. ``kwargs`` override the architecture
    config (``resolution``, ``anyres`` or config fields) when building from a registered name.
    """
    if not is_model(model_name):
        if kwargs or backbone or checkpoint_path:
            raise ValueError("Arguments other than the name apply to registered architectures only")
        return load_model(resolve_pretrained(model_name))
    if checkpoint_path or pretrained:
        if kwargs or backbone:
            raise ValueError("A checkpoint carries its own config; drop the architecture overrides")
        if checkpoint_path:
            return load_model(resolve_pretrained(checkpoint_path))
        if model_name not in _DEFAULT_PRETRAINED:
            raise RuntimeError("No released checkpoint for " + model_name)
        return load_model(resolve_pretrained(_DEFAULT_PRETRAINED[model_name]))
    model = model_entrypoint(model_name)(**kwargs)
    if backbone:
        from .gstar import model_config
        load_backbone(model, model_config(model_name, kwargs.get("resolution"), kwargs.get("anyres", False)),
                      str(backbone))
    return model
