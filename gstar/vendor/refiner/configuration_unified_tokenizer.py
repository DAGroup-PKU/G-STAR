"""HF config for the unified RepAR tokenizer + Ideogram4 Refiner."""
from transformers import PretrainedConfig


class GStarRefinerConfig(PretrainedConfig):
    model_type = "repar_ideogram4_unified"

    def __init__(self, bar_config=None, dit_config=None, vae_config=None,
                 refiner_config=None, fixed_sr_scale=1,
                 decode_num_inference_steps=52, decode_seed=0,
                 decode_sampler="native", one_step_schedule_steps=52, **kwargs):
        self.bar_config = bar_config or {}          # RepAR tokenizer 结构 (OmegaConf-style dict)
        self.dit_config = dit_config or {}          # Ideogram4Config 字段
        self.vae_config = vae_config or {}          # Flux2VAE 构造参数
        self.refiner_config = refiner_config or {}  # RepARIdeogram4DiT 条件通路参数
        self.fixed_sr_scale = int(fixed_sr_scale)
        self.decode_num_inference_steps = int(decode_num_inference_steps)
        self.decode_seed = int(decode_seed)
        self.decode_sampler = decode_sampler
        self.one_step_schedule_steps = int(one_step_schedule_steps)
        expected = {"one_step": 1, "dmad4": 4}
        if decode_sampler not in ("native", *expected):
            raise ValueError(f"Unknown sampler: {decode_sampler}")
        if decode_sampler in expected and self.decode_num_inference_steps != expected[decode_sampler]:
            raise ValueError("Distilled sampler and default step count disagree")
        if self.fixed_sr_scale < 1:
            raise ValueError(f"fixed_sr_scale must be positive, got {self.fixed_sr_scale}")
        super().__init__(**kwargs)
