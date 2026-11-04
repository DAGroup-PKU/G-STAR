"""KV-cache AR inference for class- and text-conditional image generation.

The pipeline_image_head and rope2d files follow the upstream native multimodal
generation pipeline closely so future cherry-picks stay easy.
"""

from .infer import generate_c2i_batch  # noqa: F401
from .infer import generate_t2i_batch  # noqa: F401
from .pipeline_image_head import GenerationImageHead  # noqa: F401
