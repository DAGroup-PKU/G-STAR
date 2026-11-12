"""Read manifests/images through the real processor without loading the Qwen backbone."""
import argparse
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-config", default="configs/models/gstar_ar.json")
    parser.add_argument("--data", help="Manifest (.json/.json5) or a single .jsonl file")
    parser.add_argument("--stage", choices=("pretrain", "sft"), default="pretrain")
    parser.add_argument("--samples", type=int, default=2)
    parser.add_argument("--visual-tokenizer", help="Also encode real images on GPU")
    parser.add_argument("--synthetic", action="store_true", help="Use one generated PNG instead of --data")
    parser.add_argument("--decode", action="store_true", help="Also check frozen tokenizer pixel decoding (one image per sample)")
    args = parser.parse_args()
    if args.decode and not args.visual_tokenizer:
        parser.error("--decode requires --visual-tokenizer")
    if not args.data and not args.synthetic:
        parser.error("pass --data or --synthetic")
    import torch
    from transformers import AutoTokenizer
    from gstar.config import MODEL_CODE, ROOT, read_json, validate_geometry
    from gstar.data import PackedDataset, sources, collate, preflight_images
    from gstar.vendor.model.configuration_omni import OmniConfig
    from gstar.vendor.model.processor_omni import OmniMMProcessor
    cfg = OmniConfig(**validate_geometry(read_json(args.model_config)))
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_CODE), model_max_length=8192)
    processor = OmniMMProcessor(tokenizer, cfg, training=True, relative_path="")
    temporary = None
    if args.synthetic:
        from PIL import Image
        temporary = tempfile.TemporaryDirectory()
        image = Path(temporary.name) / "image.png"
        Image.new("RGB", (256, 256), (30, 120, 220)).save(image)
        data = Path(temporary.name) / "data.jsonl"
        text = "blue image<trainable_start><img_start>" + str(image) + "<img_end><trainable_end>"
        data.write_text(json.dumps({"input": text}) + "\n")
        args.data, args.stage = str(data), "pretrain"
    files = sources(args.data)
    print("manifest_files=%d" % len(files), flush=True)
    preflight_images(files)
    dataset = PackedDataset(args.data, processor, args.stage, 8192,
                            read_json(ROOT / "configs/roles.json"), packing=False)
    visual = None
    if args.visual_tokenizer:
        from gstar.vendor.visual_tokenizer import load_frozen_visual_tokenizer
        visual = load_frozen_visual_tokenizer(args.visual_tokenizer, cfg, torch.device("cuda"))
    iterator = iter(dataset)
    for i in range(args.samples):
        batch = collate([next(iterator)])
        info = {"sample": i, "input_shape": list(batch["input_ids"].shape),
                "learnable_tokens": int((batch["labels"] != -100).sum()),
                "images": len(batch.get("images", [])), "grid": batch.get("images_grid")}
        if visual and batch.get("images"):
            from gstar.vendor.visual_tokenizer import encode_images_to_vision_tokens
            from gstar.vendor.model.spatial_merger import block_major_reorder_per_image
            indices = encode_images_to_vision_tokens(visual, batch["images"], batch["images_grid"],
                                                     cfg.visual_config.merger_merge_size,
                                                     block_major_reorder_per_image)
            info["vision_tokens_shape"] = list(indices.shape)
            if args.decode:
                if len(batch["images_grid"]) != 1:
                    raise ValueError("Decode probe expects one image per sample")
                grid = batch["images_grid"][0]
                code_h, code_w = grid[1] // cfg.visual_config.merge_size, grid[2] // cfg.visual_config.merge_size
                mms = cfg.visual_config.merger_merge_size
                raster = indices.view(1, code_h // mms, code_w // mms, mms, mms, -1).permute(0, 1, 3, 2, 4, 5).reshape(1, code_h, code_w, -1)
                with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                    pixels = visual.decode_to_pixels(raster, code_h, code_w,
                                                     grid[1] * cfg.visual_config.patch_size,
                                                     grid[2] * cfg.visual_config.patch_size)
                info["reconstructed_pixels_shape"] = list(pixels.shape)
                info["finite_pixels"] = bool(torch.isfinite(pixels).all())
        print(json.dumps(info), flush=True)
    if temporary:
        temporary.cleanup()


if __name__ == "__main__":
    main()
