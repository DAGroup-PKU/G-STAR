"""Exercise editing branch semantics without loading model weights."""
from types import SimpleNamespace
import unittest

import torch

from gstar.config import ROOT, read_json
from gstar.vendor.model.configuration_omni import OmniConfig
from gstar.vendor.model.image_head import _ExplicitGroupedHeadBase, OmniImageHead


class EditingGuidanceTests(unittest.TestCase):
    def sample(self, logits, text=1.8, image=None, stochastic=False):
        captured = []

        def processor(history, scores):
            captured.append(scores.clone())
            return scores

        sampled = _ExplicitGroupedHeadBase._sample_scores(
            SimpleNamespace(codebook_sizes=[3]), torch.tensor(logits, dtype=torch.float32),
            0, cfg_scale=text, image_cfg_scale=image, do_sample=stochastic,
            logits_processor=processor, vision_sequences=None)
        return sampled, captured[0]

    def test_guidance_selects_requested_edit_and_mirrors_all_branches(self):
        # Image-only prefers code 1, empty prefers code 2; full prefers code 0.
        tokens, _ = self.sample([[4, 1, 0], [0, 3, 0], [0, 0, 5]], text=1.8, image=1.8)
        self.assertEqual(tokens.tolist(), [0, 0, 0])

    def test_unit_text_and_image_guidance_reduces_to_full_condition(self):
        _, scores = self.sample([[1, 4, 0], [2, 0, 3], [5, 2, 1]], text=1, image=1)
        torch.testing.assert_close(scores, torch.tensor([[1., 4., 0.]]))

    def test_zero_text_unit_image_ignores_instruction(self):
        tokens, _ = self.sample([[4, 0, 0], [0, 4, 0], [0, 0, 4]], text=0, image=1)
        self.assertEqual(tokens.tolist(), [1, 1, 1])

    def test_image_scale_changes_decision(self):
        logits = [[0, 0, 0], [0, 1, 0], [0, 4, 0]]
        self.assertEqual(self.sample(logits, text=0, image=1)[0].tolist(), [1, 1, 1])
        self.assertEqual(self.sample(logits, text=0, image=2)[0].tolist(), [0, 0, 0])

    def test_native_two_branch_sampling_is_unchanged(self):
        tokens, scores = self.sample([[3, 0, 1], [0, 2, 1]])
        self.assertEqual(tokens.tolist(), [0, 0])
        torch.testing.assert_close(scores, torch.tensor([[5.4, -1.6, 1.]]))

    def test_stochastic_tokens_are_shared_by_three_branches(self):
        torch.manual_seed(42)
        tokens, _ = self.sample([[1, 0, 2]] * 3, image=1.8, stochastic=True)
        self.assertEqual(len(set(tokens.tolist())), 1)

    def test_public_dspark_head_accepts_and_shares_image_guidance(self):
        raw = read_json(ROOT / "configs/models/gstar_dspark.json")
        raw["hidden_size"] = 128
        raw["visual_quantizer_config"].update(codebook_sizes=[8, 8], depth=2, pq_num_groups=2)
        raw["visual_config"]["image_head_config"].update(
            image_head_transformer_dims=128, image_head_transformer_layers=1,
            image_head_mask_backbone_layers=1, image_head_dspark_attn_layers=1,
            image_head_group_count=2, image_head_token_embed_dims=16)
        head = OmniImageHead(OmniConfig(**raw)).eval()
        tokens = head.sample(torch.randn(3, 128), [], cfg_scale=1.8,
                             image_cfg_scale=1.8, do_sample=True)
        self.assertEqual(tuple(tokens.shape), (3, 2))
        self.assertTrue(torch.equal(tokens[0], tokens[1]))
        self.assertTrue(torch.equal(tokens[0], tokens[2]))
        self.assertTrue(bool((tokens >= 0).all() and (tokens < 8).all()))

    def test_incomplete_guidance_branches_fail(self):
        with self.assertRaises(ValueError):
            self.sample([[1, 0, 2]] * 2, image=1.8)


if __name__ == '__main__':
    unittest.main()
