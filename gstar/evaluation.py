"""Optional rank-zero image sampling during DeepSpeed training."""
from pathlib import Path

import torch
import torch.distributed as dist
from transformers import TrainerCallback

from .vendor.generation import generate_t2i_batch


class ImageSamples(TrainerCallback):
    def __init__(self, tokenizer, visual, config):
        self.tokenizer, self.visual, self.config = tokenizer, visual, config

    @torch.inference_mode()
    def on_step_end(self, args, state, control, model=None, **kwargs):
        every = self.config.get("every", 0)
        if not every or state.global_step % every:
            return
        if dist.is_initialized():
            dist.barrier()
        failed = torch.zeros(1, device=next(model.parameters()).device, dtype=torch.int32)
        error = None
        if state.is_world_process_zero:
            try:
                images = generate_t2i_batch(model, self.tokenizer, self.config["prompts"],
                                           image_size=self.config.get("image_size", 256),
                                           cfg_scale=self.config.get("cfg", 1.0),
                                           visual_tokenizer=self.visual)
                folder = Path(args.output_dir) / "samples" / ("step-%d" % state.global_step)
                folder.mkdir(parents=True, exist_ok=True)
                for index, image in enumerate(images):
                    image.save(folder / ("%02d.png" % index))
            except Exception as exc:
                failed.fill_(1)
                error = exc
        if dist.is_initialized():
            dist.all_reduce(failed, op=dist.ReduceOp.MAX)
        if failed.item():
            raise RuntimeError("Periodic image sampling failed") from error
