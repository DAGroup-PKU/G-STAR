"""The timm-style layout: registry, config files, factories and the entry points of earlier versions."""
import json
import re
import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import gstar  # noqa: E402
from gstar.models import CONFIG_DIR, is_model, list_models, model_config  # noqa: E402
from gstar.utils.config import load_config_file, parse_args, recipe  # noqa: E402

LEGACY = ROOT / "tests/legacy_recipes"


def legacy_training_kwargs(path):
    """TrainingArguments keywords exactly as the nested-recipe trainer of earlier versions built them."""
    cfg = recipe(path)
    training = dict(cfg["training"])
    training.update(output_dir=cfg["output_dir"], remove_unused_columns=False,
                    dataloader_num_workers=0, dataloader_pin_memory=False,
                    seed=cfg.get("seed", 42), ddp_find_unused_parameters=False,
                    accelerator_config={"dispatch_batches": False, "split_batches": False})
    if cfg.get("deepspeed"):
        training["deepspeed"] = cfg["deepspeed"]
    return cfg, training


def parse_train(config):
    import train
    args, _ = parse_args(train.config_parser, train.parser,
                         ["-c", str(config), "--model", "m", "--visual-tokenizer", "v"])
    return args, train.training_arguments(args)


class RegistryTests(unittest.TestCase):
    def test_registered_architectures(self):
        self.assertEqual(list_models(), ["gstar_ar", "gstar_dspark", "gstar_dspark64"])
        self.assertEqual(list_models("*dspark*"), ["gstar_dspark", "gstar_dspark64"])
        self.assertTrue(is_model("gstar_dspark") and not is_model("resnet50"))
        self.assertIs(gstar.create_model, __import__("gstar.models", fromlist=["x"]).create_model)

    def test_model_config_names_and_legacy_paths(self):
        for name in list_models():
            by_name = model_config(name)
            self.assertEqual(by_name, json.loads((CONFIG_DIR / (name + ".json")).read_text()))
            self.assertEqual(model_config("configs/models/%s.json" % name), by_name)  # pre-move path
        cfg = model_config("gstar_dspark64", resolution=512, anyres=True)
        self.assertEqual(cfg["visual_config"]["max_pixels"], 512 ** 2)
        self.assertEqual(cfg["visual_config"]["force_fix_res"], 0)


class ConfigTests(unittest.TestCase):
    def test_yaml_and_legacy_recipes_build_the_same_training_arguments(self):
        for stage in ("pretrain", "sft"):
            with self.subTest(stage=stage):
                cfg, old = legacy_training_kwargs(LEGACY / (stage + ".json"))
                for config in (ROOT / "configs" / (stage + ".yaml"), LEGACY / (stage + ".json")):
                    args, new = parse_train(config)
                    self.assertEqual(new, old, config)
                    self.assertEqual((args.stage, args.data, args.roles, args.max_length, args.packing, args.seed,
                                      args.head_lr_scale, args.output),
                                     (cfg["stage"], cfg["data"], cfg["roles"], cfg["max_length"], cfg["packing"],
                                      cfg["seed"], cfg["head_lr_scale"], cfg["output_dir"]))
                    evaluation = cfg.get("evaluation", {})
                    self.assertEqual(args.sample_every, evaluation.get("every", 0))
                    if evaluation:
                        self.assertEqual((args.sample_image_size, args.sample_cfg, args.sample_prompts),
                                         (evaluation["image_size"], evaluation["cfg"], evaluation["prompts"]))

    def test_command_line_overrides_config(self):
        import train
        args, _ = parse_args(train.config_parser, train.parser,
                             ["-c", str(ROOT / "configs/sft.yaml"), "--model", "m", "--visual-tokenizer", "v",
                              "--lr", "3e-5", "--max-steps", "7", "--no-grad-checkpointing"])
        kwargs = train.training_arguments(args)
        self.assertEqual((kwargs["learning_rate"], kwargs["max_steps"], kwargs["gradient_checkpointing"]),
                         (3e-5, 7, False))
        self.assertNotIn("gradient_checkpointing_kwargs", kwargs)

    def test_unknown_config_key_is_an_error(self):
        import tempfile
        import train
        with tempfile.NamedTemporaryFile("w", suffix=".yaml") as handle:
            handle.write("lr: 1.0e-4\nlearning_rat: 2\n")
            handle.flush()
            with self.assertRaises(SystemExit):
                parse_args(train.config_parser, train.parser, ["-c", handle.name, "--model", "m"])
        self.assertEqual(load_config_file(ROOT / "configs/sft.yaml")["roles"], str(ROOT / "configs/roles.json"))


class FactoryTests(unittest.TestCase):
    def test_optimizer_groups_match_the_earlier_trainer(self):
        from gstar.optim import create_optimizer, pretrain_decay_names
        from test_runtime import tiny_config
        from gstar.vendor.model.modeling_omni import OmniForCausalLM
        model = OmniForCausalLM(tiny_config("dspark"))
        model.lm_head.weight.requires_grad_(False)
        decay = {name for name, _ in model.named_parameters() if not re.search(r"embed_tokens|lm_head", name)}
        self.assertEqual(pretrain_decay_names(model), decay)
        old = []
        for is_head in (False, True):
            for use_decay in (False, True):
                params = [p for n, p in model.named_parameters() if p.requires_grad
                          and ("image_head" in n) == is_head and (n in decay) == use_decay]
                if params:
                    old.append(([id(p) for p in params], 1e-4 * (10.0 if is_head else 1), 0.05 if use_decay else 0.0))
        opt = create_optimizer(model, 1e-4, 0.05, betas=(0.9, 0.96), eps=1e-8, head_lr_scale=10.0, decay_names=decay)
        new = [([id(p) for p in g["params"]], g["lr"], g["weight_decay"]) for g in opt.param_groups]
        self.assertEqual(new, old)
        self.assertEqual(opt.defaults["betas"], (0.9, 0.96))

    def test_collate_factory(self):
        from gstar.data import collate, collate_attention_layout, create_collate_fn
        self.assertIs(create_collate_fn(False), collate)
        fn = create_collate_fn(True, True)
        self.assertIs(fn.func, collate_attention_layout)
        self.assertEqual(fn.keywords, {"include_scalars": True})

    def test_scheduler_factory_matches_transformers(self):
        from transformers import get_scheduler
        from gstar.scheduler import create_scheduler
        for name, kwargs in (("cosine", {}), ("cosine_with_min_lr", {"min_lr_rate": 0.1})):
            a = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=1.0)
            b = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=1.0)
            s1 = create_scheduler(name, a, 10, 100, kwargs)
            s2 = get_scheduler(name, optimizer=b, num_warmup_steps=10, num_training_steps=100,
                               scheduler_specific_kwargs=kwargs)
            lrs = []
            for _ in range(100):
                lrs.append((s1.get_last_lr()[0], s2.get_last_lr()[0]))
                a.step(); b.step(); s1.step(); s2.step()
            self.assertTrue(all(x == y for x, y in lrs), name)


class CompatTests(unittest.TestCase):
    def test_earlier_module_paths(self):
        from gstar.batch_metadata import prepare_attention_layout
        from gstar.config import MODEL_CODE, ROOT as root, read_json, recipe as legacy_recipe, validate_geometry
        from gstar.data import PackedDataset, collate, conversation, weighted_rows
        from gstar.edit import generate_edit
        from gstar.model import forward_inputs, initialize, load, replace_head, save_sidecars
        from gstar.train import ImageTrainer
        self.assertEqual(root, ROOT)
        self.assertTrue((MODEL_CODE / "modeling_omni.py").is_file())

    def test_package_import_is_light(self):
        import subprocess
        code = ("import sys, gstar, gstar.generation.prompt_enhance; "
                "print(int(any(m.startswith('gstar.vendor') or m == 'transformers' for m in sys.modules)))")
        out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, check=True)
        self.assertEqual(out.stdout.strip(), "0")


if __name__ == "__main__":
    unittest.main()
