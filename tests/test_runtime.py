"""GPU smoke tests for the real heads, packing, HF reload and generation."""
import itertools
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments

from gstar.config import MODEL_CODE, ROOT, read_json
from gstar.data import PackedDataset, collate, conversation, weighted_rows
from gstar.model import save_sidecars
from gstar.vendor.model.configuration_omni import OmniConfig
from gstar.vendor.model.modeling_omni import OmniForCausalLM
from gstar.vendor.model.processor_omni import OmniMMProcessor, OmniProcessorOutput
from gstar.vendor.generation import generate_t2i_batch
from gstar.train import ImageTrainer


def tiny_config(name):
    cfg = read_json(ROOT / ("configs/models/gstar_" + name + ".json"))
    cfg.update(hidden_size=128, intermediate_size=256, num_hidden_layers=1,
               num_attention_heads=1, num_key_value_heads=1, max_window_layers=1)
    head = cfg["visual_config"]["image_head_config"]
    head.update(image_head_transformer_dims=128, image_head_transformer_layers=1,
                image_head_mask_backbone_layers=1, image_head_dspark_attn_layers=1,
                image_head_token_embed_dims=16, unmerger_intermediate_size=256)
    cfg["visual_config"]["merger_intermediate_size"] = 256
    return OmniConfig(**cfg)


class DataTests(unittest.TestCase):
    def test_local_image_pretrain_and_sft(self):
        from PIL import Image
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "image.png"
            Image.new("RGB", (320, 240), (30, 120, 220)).save(path, format="PNG")
            descriptor = json.dumps({"type": "file", "path": str(path), "preprocess": "bucketresize_256"})
            content = "<img_start>" + descriptor + "<img_end>"
            tokenizer = AutoTokenizer.from_pretrained(str(MODEL_CODE), model_max_length=8192)
            processor = OmniMMProcessor(tokenizer, tiny_config("ar"), training=True, relative_path="")
            row = {"input": "blue image<trainable_start>" + content + "<trainable_end>"}
            manifest = Path(folder) / "data.jsonl"
            manifest.write_text(json.dumps(row) + "\n")
            batch = collate([next(iter(PackedDataset(manifest, processor, "pretrain", 128, {}, packing=False)))])
            self.assertEqual(batch["images_grid"], [(1, 16, 16)])
            self.assertEqual(int(batch["patch_nums"].sum()), 64)
            self.assertEqual(tuple(batch["input_ids"].shape), (1, 128))
            roles = read_json(ROOT / "configs/roles.json")
            row = {"messages": [{"role": "user", "content": "draw blue"},
                                {"role": "assistant", "content": content}]}
            manifest.write_text(json.dumps(row) + "\n")
            item = next(iter(PackedDataset(manifest, processor, "sft", 128, roles, packing=False)))
            self.assertEqual(item["labels"][0], -100)
            self.assertGreater(item["labels"].count(151681), 0)
            print("Local PNG -> patches -> pretrain/SFT labels=OK", flush=True)
    def test_rates_and_replay(self):
        with tempfile.TemporaryDirectory() as folder:
            a, b = Path(folder) / "a.jsonl", Path(folder) / "b.jsonl"
            a.write_text('{"id": "a"}\n')
            b.write_text('{"id": "b"}\n')
            files = [(a, 1), (b, 2)]
            first = list(itertools.islice(weighted_rows(files, 42), 3))
            self.assertEqual(sorted(x["id"] for x in first), ["a", "b", "b"])
            self.assertEqual(first, list(itertools.islice(weighted_rows(files, 42), 3)))

    def test_sft_roles_and_loss_mask(self):
        tokenizer = SimpleNamespace(eos_token_id=99)
        processor = SimpleNamespace(tokenizer=tokenizer,
                                    process_one=lambda s: OmniProcessorOutput(input_ids=[10, 11], labels=[10, 11]))
        row = {"messages": [{"role": "user", "content": "question"},
                            {"role": "assistant", "content": "answer"}]}
        output = conversation(row, processor, {"user": 1, "assistant": 2})
        self.assertEqual(output.input_ids, [1, 10, 11, 2, 10, 11, 99])
        self.assertEqual(output.labels, [-100, -100, -100, -100, 10, 11, 99])

    def test_packing_resets_positions(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "data.jsonl"
            path.write_text('{"input": "a"}\n')
            tokenizer = SimpleNamespace(eos_token_id=9, pad_token_id=0)
            processor = SimpleNamespace(tokenizer=tokenizer,
                                        process_one=lambda s: OmniProcessorOutput(input_ids=[1, 2], labels=[1, 2]))
            dataset = PackedDataset(path, processor, "pretrain", 8, {})
            item = next(iter(dataset))
            self.assertEqual(item["position_ids"], [0, 1, 2, 0, 1, 2, 0, 0])
            self.assertEqual(item["seqlens"][:2], [3, 3])
            self.assertEqual(item["labels"], [1, 2, 9, -100, 2, 9, -100, -100])


@unittest.skipUnless(torch.cuda.is_available(), "CUDA/FlashAttention required")
class ModelTests(unittest.TestCase):
    def test_trainer_checkpoint_and_resume(self):
        from torch.utils.data import IterableDataset
        import numpy as np
        cfg = tiny_config("dspark")
        ids = [10, 151679, 151681, 151681, 151681, 151681, 151680, 151645]
        class Cached(IterableDataset):
            def __iter__(self):
                while True:
                    yield {"input_ids": ids[:], "labels": [-100, -100] + ids[2:],
                           "position_ids": list(range(8)), "seqlens": [8] + [0] * 7,
                           "patch_nums": [4], "images_grid": [(1, 4, 4)],
                           "vision_tokens": [np.zeros((4, 16), dtype=np.int64)]}
        with tempfile.TemporaryDirectory() as folder:
            tokenizer = AutoTokenizer.from_pretrained(str(MODEL_CODE))
            args = TrainingArguments(output_dir=folder, max_steps=1,
                                     per_device_train_batch_size=1, bf16=True,
                                     save_steps=1, save_safetensors=False,
                                     gradient_checkpointing=True,
                                     report_to="none", remove_unused_columns=False,
                                     dataloader_pin_memory=False,
                                     accelerator_config={"dispatch_batches": False, "split_batches": False})
            args._n_gpu = 1
            trainer = ImageTrainer(model=OmniForCausalLM(cfg).bfloat16(), args=args,
                                   train_dataset=Cached(), data_collator=collate,
                                   processing_class=tokenizer, visual=None, roles=None)
            trainer.train()
            self.assertTrue((Path(folder) / "checkpoint-1/modeling_omni.py").is_file())
            trainer.args.max_steps = 2
            trainer.train(resume_from_checkpoint=str(Path(folder) / "checkpoint-1"))
            self.assertEqual(trainer.state.global_step, 2)
            print("Trainer optimizer/checkpoint/resume=OK", flush=True)

    def test_heads_backward_reload_and_generate(self):
        tokenizer = AutoTokenizer.from_pretrained(str(MODEL_CODE))
        for name in ("ar", "dspark", "dspark64"):
            with self.subTest(head=name), tempfile.TemporaryDirectory() as folder:
                cfg = tiny_config(name)
                model = OmniForCausalLM(cfg).cuda().bfloat16().train()
                mms = cfg.visual_config.merger_merge_size
                codes = 4 * mms * mms
                ids = torch.tensor([[10, 151679] + [151681] * 4 + [151680, 151645]], device="cuda")
                labels = ids.clone()
                labels[:, :2] = -100
                indices = torch.randint(0, 16384, (codes, 16), device="cuda")
                output = model(input_ids=ids, labels=labels, vision_tokens=indices,
                               patch_nums=torch.tensor([4], device="cuda"),
                               images_grid=[(1, 4 * mms, 4 * mms)])
                self.assertTrue(torch.isfinite(output.loss))
                output.loss.backward()
                self.assertTrue(any(p.grad is not None for p in model.image_head.parameters()))
                print("head=%s loss=%.4f backward=OK" % (name, output.loss.item()), flush=True)
                model.save_pretrained(folder, safe_serialization=False)
                save_sidecars(folder, tokenizer)
                del model, output
                torch.cuda.empty_cache()
                model = AutoModelForCausalLM.from_pretrained(folder, trust_remote_code=True,
                                                            torch_dtype=torch.bfloat16).cuda().eval()
                model.processor = OmniMMProcessor(tokenizer, model.config, training=False, relative_path="")
                class Decoder:
                    def decode_to_pixels(self, indices, code_h, code_w, height, width):
                        self.shape = indices.shape
                        return torch.zeros((1, 3, height, width), device="cuda")
                decoder = Decoder()
                images = generate_t2i_batch(model, tokenizer, ["test"], image_height=64 * mms,
                                           image_width=64 * mms, do_sample=False,
                                           visual_tokenizer=decoder)
                self.assertEqual(images[0].size, (64 * mms, 64 * mms))
                self.assertEqual(decoder.shape[-1], 16)
                prompt = torch.tensor([[151667, 151679] + [151681] * 4 + [151680, 151668]], device="cuda")
                text = model.generate(input_ids=prompt, attention_mask=torch.ones_like(prompt),
                                      vision_tokens=indices, patch_nums=torch.tensor([4], device="cuda"),
                                      images_grid=[(1, 4 * mms, 4 * mms)], max_new_tokens=2,
                                      do_sample=False, use_cache=True)
                self.assertGreater(text.shape[1], prompt.shape[1])
                print("head=%s HF-reload + image-generation=OK" % name, flush=True)
                del model
                torch.cuda.empty_cache()


if __name__ == "__main__":
    unittest.main()
