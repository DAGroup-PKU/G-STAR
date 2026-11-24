"""AdamW with separate learning rates for the image head and weight decay by parameter name."""
import re

import torch

__all__ = ["pretrain_decay_names", "param_groups", "create_optimizer"]


def pretrain_decay_names(model):
    """Pretraining decays every parameter except the token embeddings and the LM head."""
    return {name for name, _ in model.named_parameters() if not re.search(r"embed_tokens|lm_head", name)}


def param_groups(model, lr, weight_decay, head_lr_scale=1.0, decay_names=()):
    """Up to four groups, in this order: backbone without / with decay, image head without / with decay; the image
    head (parameters named ``image_head``) runs at ``lr * head_lr_scale``. Frozen parameters are left out."""
    decay_names = set(decay_names)
    groups = []
    for is_head in (False, True):
        for use_decay in (False, True):
            params = [p for n, p in model.named_parameters() if p.requires_grad
                      and ("image_head" in n) == is_head and (n in decay_names) == use_decay]
            if params:
                groups.append({"params": params,
                               "lr": lr * (head_lr_scale if is_head else 1),
                               "weight_decay": weight_decay if use_decay else 0.0})
    return groups


def create_optimizer(model, lr, weight_decay, betas=(0.9, 0.999), eps=1e-8, head_lr_scale=1.0, decay_names=()):
    """torch.optim.AdamW over ``param_groups``."""
    return torch.optim.AdamW(param_groups(model, lr, weight_decay, head_lr_scale, decay_names), betas=betas, eps=eps)
