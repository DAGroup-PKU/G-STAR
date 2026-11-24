"""Learning-rate schedules: the transformers schedulers by name (``cosine``, ``cosine_with_min_lr``, ``linear`` ...)."""
__all__ = ["create_scheduler"]


def create_scheduler(sched, optimizer, num_warmup_steps, num_training_steps, scheduler_kwargs=None):
    """The ``sched`` schedule with linear warmup over ``num_warmup_steps``; ``scheduler_kwargs`` are the schedule's
    own arguments, e.g. ``{"min_lr_rate": 0.1}`` for ``cosine_with_min_lr``."""
    from transformers import get_scheduler
    return get_scheduler(sched, optimizer=optimizer, num_warmup_steps=num_warmup_steps,
                         num_training_steps=num_training_steps, scheduler_specific_kwargs=scheduler_kwargs)
