"""G-STAR image-language models: training, evaluation and inference (layout after timm).

    gstar.models      architecture registry, create_model, checkpoint loading and initialization
    gstar.data        weighted JSONL sources, multimodal processing, packing, collation
    gstar.optim       AdamW with the image-head learning rate and name-based weight decay
    gstar.scheduler   learning-rate schedules
    gstar.trainer     the Trainer subclass and the periodic sampling callback
    gstar.generation  text-to-image, editing, prompt enhancement
    gstar.utils       paths and config-file handling
    gstar.fast        switches of the fast inference paths

The entry points are the scripts in the repository root: train.py, validate.py, inference.py, init_checkpoint.py.
Importing the package itself stays light (no torch / transformers) so ``python -m gstar.prompt_enhance`` runs in a
separate environment; the model API is loaded on first use.
"""
_MODEL_API = ("create_model", "list_models", "is_model", "model_entrypoint", "register_model", "list_pretrained",
              "resolve_pretrained", "load", "load_model")


def __getattr__(name):
    if name in _MODEL_API:
        from . import models
        return getattr(models, name)
    raise AttributeError("module 'gstar' has no attribute " + repr(name))
