"""HF config for the self-contained BAR tokenizer."""
from transformers import PretrainedConfig


class BARTokenizerConfig(PretrainedConfig):
    model_type = "bar_tokenizer"

    def __init__(self, bar_config=None, **kwargs):
        # ``bar_config`` is the full OmegaConf-style tokenizer config as a plain
        # nested dict (model.vq_model / losses / ...). The model is rebuilt from
        # it in BARTokenizer.__init__.
        self.bar_config = bar_config if bar_config is not None else {}
        super().__init__(**kwargs)
