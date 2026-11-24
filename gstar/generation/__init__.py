"""Generation: text-to-image (AR / DSpark heads), instruction-based editing and optional prompt enhancement.

Loaded on first use, so that ``gstar.generation.prompt_enhance`` imports without the model code."""


def __getattr__(name):
    if name == "generate_t2i_batch":
        from ..vendor.generation import generate_t2i_batch
        return generate_t2i_batch
    if name == "generate_edit":
        from .edit import generate_edit
        return generate_edit
    if name in ("enhance_prompt", "PE_MODELS"):
        from . import prompt_enhance
        return getattr(prompt_enhance, name)
    raise AttributeError("module 'gstar.generation' has no attribute " + repr(name))
