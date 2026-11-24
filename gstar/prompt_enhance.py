"""Moved to gstar.generation.prompt_enhance (``python -m gstar.prompt_enhance`` still works)."""
from .generation.prompt_enhance import PE_MODELS, SAMPLING, enhance_prompt, main

if __name__ == "__main__":
    main()
