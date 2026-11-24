"""The Trainer for pretraining and SFT: the loss of the image-language model with the frozen visual tokenizer."""
import re

from transformers import Trainer

from ..models import forward_inputs, save_sidecars
from ..optim import create_optimizer, pretrain_decay_names
from ..scheduler import create_scheduler

__all__ = ["TRAINABLE", "set_trainable", "ImageTrainer"]

# the parameters that train: bridge, embeddings, backbone, final norm, LM head, image head, visual embeddings
TRAINABLE = re.compile(r"visual_bridge|embed_tokens|lm_head|^model\.layers|^model\.norm|image_head|model\.visual_embed_layers")


def set_trainable(model, pattern=TRAINABLE):
    """Train the parameters whose names match ``pattern`` and freeze the others."""
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(bool(pattern.search(name)))
    return model


class ImageTrainer(Trainer):
    def __init__(self, *args, visual, roles, head_lr_scale=1.0, decay_mode="standard", **kwargs):
        super().__init__(*args, **kwargs)
        self.visual, self.roles, self.head_lr_scale = visual, roles, head_lr_scale
        self.decay_mode = decay_mode

    def _prepare_inputs(self, inputs):
        if getattr(self.model.config, "attention_layout_cpu", False):
            from ..data import prepare_attention_layout
            inputs = prepare_attention_layout(
                inputs, getattr(self.model.config, "attention_scalars_cpu", False))
        return super()._prepare_inputs(inputs)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        output = model(**forward_inputs(self.model, inputs, self.visual))
        return (output.loss, output) if return_outputs else output.loss

    def create_optimizer(self):
        if self.optimizer is None:
            decay = pretrain_decay_names(self.model) if self.decay_mode == "pretrain" else set(self.get_decay_parameter_names(self.model))
            self.optimizer = create_optimizer(self.model, self.args.learning_rate, self.args.weight_decay,
                                              betas=(self.args.adam_beta1, self.args.adam_beta2),
                                              eps=self.args.adam_epsilon, head_lr_scale=self.head_lr_scale,
                                              decay_names=decay)
        return self.optimizer

    def create_scheduler(self, num_training_steps, optimizer=None):
        if self.lr_scheduler is None:
            self.lr_scheduler = create_scheduler(self.args.lr_scheduler_type,
                                                 self.optimizer if optimizer is None else optimizer,
                                                 num_warmup_steps=self.args.get_warmup_steps(num_training_steps),
                                                 num_training_steps=num_training_steps,
                                                 scheduler_kwargs=self.args.lr_scheduler_kwargs)
            self._created_lr_scheduler = True
        return self.lr_scheduler

    def _save(self, output_dir=None, state_dict=None):
        super()._save(output_dir, state_dict)
        save_sidecars(output_dir or self.args.output_dir, self.processing_class, self.roles)
