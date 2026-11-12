"""Run with torchrun --standalone --nproc_per_node=2 tests/smoke_distributed.py."""
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import IterableDataset
from transformers import AutoTokenizer, TrainingArguments

from gstar.config import MODEL_CODE, ROOT
from gstar.data import collate
from gstar.train import ImageTrainer
from gstar.vendor.model.modeling_omni import OmniForCausalLM
from test_runtime import tiny_config


class Cached(IterableDataset):
    def __iter__(self):
        ids = [10, 151679, 151681, 151681, 151681, 151681, 151680, 151645]
        while True:
            yield {"input_ids": ids[:], "labels": [-100, -100] + ids[2:],
                   "position_ids": list(range(8)), "seqlens": [8] + [0] * 7,
                   "patch_nums": [4], "images_grid": [(1, 4, 4)],
                   "vision_tokens": [np.zeros((4, 16), dtype=np.int64)]}


def main():
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl")
    folder = [tempfile.mkdtemp(prefix="gstar-zero2-") if dist.get_rank() == 0 else None]
    dist.broadcast_object_list(folder, src=0)
    args = TrainingArguments(output_dir=folder[0], max_steps=1,
                             per_device_train_batch_size=1, bf16=True,
                             save_steps=1, save_safetensors=False, report_to="none",
                             gradient_checkpointing=True, remove_unused_columns=False,
                             dataloader_pin_memory=False, ddp_find_unused_parameters=False,
                             deepspeed=str(ROOT / "configs/deepspeed.json"),
                             accelerator_config={"dispatch_batches": False, "split_batches": False})
    trainer = ImageTrainer(model=OmniForCausalLM(tiny_config("dspark")).bfloat16(),
                           args=args, train_dataset=Cached(), data_collator=collate,
                           processing_class=AutoTokenizer.from_pretrained(str(MODEL_CODE)),
                           visual=None, roles=None)
    trainer.train()
    if dist.get_rank() == 0:
        print("DeepSpeed ZeRO-2: two-rank forward/backward/save OK: " + folder[0], flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
