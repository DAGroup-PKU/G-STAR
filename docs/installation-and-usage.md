# G-STAR Installation & Usage

[← Back to README](../README.md)

G-STAR supports text-to-image generation, instruction-based image editing,
image understanding, pretraining and SFT with a discrete visual tokenizer and
an AR or DSpark image head on a Qwen3 backbone.

**Ready for local inference: install the dependencies, download a complete
model checkpoint and its matching visual refiner, then run the commands below.
No training, training dataset, or internal job platform is required.**

The repository contains code; model weights are distributed separately.
The text-to-image and editing paths have both been verified with real model
weights on NVIDIA H100 GPUs.

## Contents

- [Install](#install)
- [Download the weights](#download-the-weights)
- [Generate an image](#generate-an-image)
- [Prompt enhancement](#prompt-enhancement)
- [Edit an image](#edit-an-image)
- [Verified inference](#verified-inference)
- [Image heads](#image-heads)
- [Data](#data): [records](#records) · [image descriptors](#image-descriptors) · [manifests](#manifests) · [building data](#building-data)
- [Training](#training)
- [Tests](#tests)

## Install

Run all commands from the repository root. Use a CUDA-capable NVIDIA GPU.
The verified environment uses Python 3.12, PyTorch 2.6.0 with CUDA 12.4,
Transformers 4.57.0, FlashAttention 2.7.4.post1 and Diffusers 0.38.0.
Inference was verified on an H100 with 80 GB of memory per worker; it does
not require multiple GPUs. Editing peaked at approximately 43 GiB of
allocated CUDA memory in the verified configuration.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip wheel packaging ninja
python -m pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0
python -m pip install --no-build-isolation -r requirements.txt \
  'flash-attn==2.7.4.post1' 'diffusers==0.38.0' 'accelerate==1.12.0'

export CUDA_VISIBLE_DEVICES=0
export USE_TF=0
export TOKENIZERS_PARALLELISM=false
```

PyTorch and FlashAttention must match the CUDA runtime and GPU on the machine.

## Download the weights

The text-to-image export is on Hugging Face at
[DAGroup-PKU/G-star](https://huggingface.co/DAGroup-PKU/G-star): its root is the
generation model (`gstar-t2i/` below) and its `refiner/` folder is the matching
refiner (`gstar-t2i-refiner/` below). The editing exports are not published yet.

Download the **complete Hugging Face exports** accompanying the model release
and keep their directory contents intact. The local folder names below are
used by the examples:

```text
weights/
  gstar-t2i/             # Full generation model: backbone, bridge and DSpark head
  gstar-t2i-refiner/     # Matching g16 SR2x encoder/decoder export
  gstar-edit/            # Full editing SFT model: backbone, bridge and DSpark head
  gstar-edit-refiner/    # Matching g16 reconstruction encoder/decoder export
```

| Task | Model | Matching visual export | Prompt format |
| --- | --- | --- | --- |
| Text-to-image | Qwen3-8B, DSpark, 16 codebooks | g16 SR2x refiner supplied with that model | Pretrain |
| Image editing | Qwen3-8B editing SFT, DSpark, 16 codebooks | g16 reconstruction refiner supplied with that model | SFT |

Keep the model's `config.json`, text tokenizer files, all weight shards and
their index file, and `role_config.json` for SFT checkpoints. Keep the visual
export's `config.json`, all weight files and its Python implementation files.
The refiner export includes the visual encoder, codebooks and image decoder.
Downloading the full paired exports supplies the components required for
inference; a base Qwen checkpoint or an encoder alone is insufficient.

Use the visual export paired with each checkpoint. Matching the number of
codebooks alone is insufficient: the learned codebooks and decoder must also
match. The g8 and g16 exports are incompatible.

Set the paths to your downloaded directories:

```bash
export T2I_MODEL="$PWD/weights/gstar-t2i"
export T2I_REFINER="$PWD/weights/gstar-t2i-refiner"
export EDIT_MODEL="$PWD/weights/gstar-edit"
export EDIT_REFINER="$PWD/weights/gstar-edit-refiner"
```

## Generate an image

This example uses the verified g16 pretrain generation configuration: CFG 2.4,
top-k 4096, top-p 0.75 and temperature 0.7. It generates a 32 x 32 grid of g16 image codes for a 1024 x 1024 target,
then uses the matching SR2x refiner to produce a native 2048 image.

```bash
OMNI_DISABLE_WAVELET_COLOR_FIX=1 python -m gstar.infer \
  --task t2i --model "$T2I_MODEL" --visual-tokenizer "$T2I_REFINER" \
  --prompt-mode pretrain \
  --prompt 'A golden retriever lying on a wooden floor, looking up at its owner, warm morning sunlight, realistic photography.' \
  --height 1024 --width 1024 --cfg 2.4 \
  --top-k 4096 --top-p 0.75 --temperature 0.7 --seed 20260915 \
  --output outputs/generated_2048.png
```

For the 1024 benchmark output, resize the SR image with Lanczos:

```bash
python - <<'PY'
from PIL import Image
with Image.open('outputs/generated_2048.png') as image:
    image.resize((1024, 1024), Image.Resampling.LANCZOS).save('outputs/generated_1024.png')
PY
```

Prompts are used as supplied. Pass the rewritten text to `--prompt` when using
an externally rewritten prompt. `--prompt-mode pretrain` omits SFT roles even
if the checkpoint includes a role configuration. For a generation checkpoint
trained with SFT roles, select `--prompt-mode sft` instead. The default `auto`
uses SFT roles when `role_config.json` is present.

## Prompt enhancement

We recommend rewriting short prompts with Qwen-Image-2.1-PE before generation.
Add `--prompt-enhance` to any `t2i` or `edit` command: text-to-image uses
[Qwen/Qwen-Image-2.1-PE-T2I](https://huggingface.co/Qwen/Qwen-Image-2.1-PE-T2I)
and editing uses [Qwen/Qwen-Image-2.1-PE-I2I](https://huggingface.co/Qwen/Qwen-Image-2.1-PE-I2I)
(which also reads `--image`). The rewritten prompt is printed, the PE model is
released from the GPU, and G-STAR then generates as usual. `--pe-model` overrides
the PE model id or local path. Enhancement is off by default. The PE
`wh_ratio` suggestion is ignored; `--height`/`--width` still set the output size.

The PE models need `transformers>=5.4`, newer than the version pinned in
`requirements.txt`. Install it in a separate environment and point
`--pe-python` at that environment's interpreter; the rewrite then runs as a
subprocess (`python -m gstar.prompt_enhance`):

```bash
OMNI_DISABLE_WAVELET_COLOR_FIX=1 python -m gstar.infer \
  --task t2i --model "$T2I_MODEL" --visual-tokenizer "$T2I_REFINER" \
  --prompt-mode pretrain --prompt '一只在雨中弹吉他的柯基' \
  --prompt-enhance --pe-python /path/to/pe-env/bin/python \
  --height 1024 --width 1024 --cfg 2.4 \
  --top-k 4096 --top-p 0.75 --temperature 0.7 --seed 20260915 \
  --output outputs/enhanced_2048.png
```

## Edit an image

Use the editing model with its matching reconstruction refiner. For the
verified square-image configuration, prepare a 1024 x 1024 source image:

```bash
python - <<'PY'
from PIL import Image, ImageOps
with Image.open('source.png') as image:
    ImageOps.fit(image.convert('RGB'), (1024, 1024),
                 method=Image.Resampling.LANCZOS).save('source_1024.png')
PY

python -m gstar.infer \
  --task edit --model "$EDIT_MODEL" --visual-tokenizer "$EDIT_REFINER" \
  --image source_1024.png --prompt '将背景改为城市街道' \
  --height 1024 --width 1024 --cfg 1.8 --image-cfg 1.8 \
  --top-k 4096 --top-p 0.75 --temperature 0.7 --seed 20260908 \
  --output outputs/edited_1024.png
```

The image-preparation command center-crops to a square. Editing encodes the
source image, conditions generation on the image and instruction, and decodes
the generated tokens with the refiner. The processed source dimensions must
match the requested output dimensions.

`--cfg` controls text guidance; `--image-cfg` controls image guidance. Image
CFG 1.0 uses full/image branches. Other image guidance scales use
full/image/empty branches with the formula
`L_image + text_cfg * (L_full - L_image) + (image_cfg - 1) * (L_image - L_empty)`.
Three-branch guidance supports the explicit AR and DSpark heads.

## Verified inference

| Path | Actual validation |
| --- | --- |
| g16 text-to-image + SR2x | 4/4 rewritten benchmark prompts generated complete code grids and finite 2048 images, resized to 1024 |
| g16 image editing + reconstruction | 4/4 background edits generated finite 1024 images; city, forest, grassland and beach edits inspected |
| Editing guidance | 8 regression tests passed, including real DSpark head sampling |
| Public CLI quickstart | Both `python -m gstar.infer` examples produced valid images: 2048 text-to-image and 1024 editing |

The model backbone, bridge, image head and generation pipeline run from this
repository. Weights are loaded into the local implementation with strict
loading checks; missing or incompatible model weights raise an error.
The standalone visual export loads its bundled implementation through
`trust_remote_code=True`.

These checks establish that inference runs and produces images. Fine details
of edited subjects can be redrawn, and identical seeds do not guarantee exact
token or pixel equality across hardware and inference configurations.

## Image heads

| Config | Tokenizer groups | Spatial merge | Head |
| --- | --- | --- | --- |
| `configs/models/gstar_ar.json` | 16 | 1x1 | Causal PQ AR |
| `configs/models/gstar_dspark.json` | 16 | 1x1 | DSpark, 16 groups: 3 all-MASK layers + 1 group-causal layer |
| `configs/models/gstar_dspark64.json` | 16 | 2x2 | DSpark, 64 groups |

For DSpark, `image_head_group_count` must equal `depth * merger_merge_size**2`
(checked by `gstar.config.validate_geometry`). A g16 tokenizer with a 2x2
merge therefore needs 64 head groups.

## Data

### Records

Each training source is a JSONL file. **Pretraining** rows carry text in
`input` (or `content`), with an optional `title` prefix and `eosloss`
(default `true`). Only spans wrapped in `<trainable_start>...<trainable_end>`
are supervised; without these markers the whole row is supervised. Images are
embedded inline as `<img_start>DESCRIPTOR<img_end>`:

```json
{"input": "A red apple on a wooden table.<trainable_start><img_start>/data/images/apple.jpg<img_end><trainable_end>"}
```

**SFT** rows use `messages` (or ShareGPT-style `conversations` with
`from`/`value`). Only assistant turns are supervised by default; set
`"learnable": true/false` on a message to override. `content` may be a string
with inline image markers, or a list of `{"type": "text"}` / `{"type": "image"}`
/ `{"type": "image_url"}` parts:

```json
{"messages": [{"role": "user", "content": "Draw a red apple."},
              {"role": "assistant", "content": "<img_start>{\"type\": \"file\", \"path\": \"/data/images/apple.jpg\"}<img_end>"}]}
```

Role tokens come from `configs/roles.json`.

### Image descriptors

| Descriptor | Source |
| --- | --- |
| `/abs/path/img.jpg` | Local file |
| `{"type": "file", "path": "/abs/path/img.jpg"}` | Local file |
| `{"path": "key/img.jpg"}` | Local file, relative to the processor's `relative_path` |
| `cached://shard.bin:OFFSET` | Pre-encoded token indices from `$OMNI_CACHED_RVQ_ROOT` |

JSON descriptors may set `"preprocess": "fixres_N"` (center crop + resize to
N x N) or `"bucketresize_N"` (resize to the nearest of 40 aspect-ratio buckets
at an N x N pixel budget; N in 256/512/1024). The legacy prefix form
`<fixres_256>/path.jpg` is also accepted. `force_fix_res` in the model config
(or `OMNI_FORCE_FIX_RES` / `OMNI_FORCE_BUCKET_RES`) overrides per-image rules.

Only local images are supported; a missing file raises an error for that sample.

### Manifests

A manifest maps glob patterns (files, or directories searched recursively for
`*.jsonl`) to sampling multipliers. The integer part repeats each row and the
fractional part adds one more copy with that probability. Patterns are
resolved against `root`, relative to the manifest file:

```json
{"train_files": {"root": "../../data/pretrain",
                 "sample_rate": {"general/*.jsonl": 1.0, "text_rendering/*.jsonl": 2.0}}}
```

A flat `{"pattern": weight}` object and a single `.jsonl` path are also
accepted. See `configs/data/pretrain.json` and `configs/data/sft.json5`.

Sources are mixed with a fixed seed and shuffled through a 256-row buffer.
Packing keeps per-sample `position_ids`/`seqlens`, never truncates images, and
drops over-length samples whole.

### Building data

`tools/prepare_data.py` converts a JSONL/CSV of `{"image", "caption"}` pairs
into either record format (images must be local files):

```bash
python tools/prepare_data.py pairs.jsonl data/pretrain/general/part0.jsonl \
  --preprocess bucketresize_256
python tools/prepare_data.py pairs.jsonl data/sft/t2i/part0.jsonl --stage sft
```

Check a manifest without loading the LLM:

```bash
python tools/check_data.py --data configs/data/pretrain.json --samples 2
python tools/check_data.py --data configs/data/sft.json5 --stage sft
python tools/check_data.py --synthetic --visual-tokenizer "$VISUAL_TOKENIZER" --decode
```

## Training

Training uses a base Qwen checkpoint and a frozen visual tokenizer. Set:

```bash
export QWEN=/path/to/Qwen3-8B-Base
export VISUAL_TOKENIZER=/path/to/gstar-training-tokenizer
export HEAD=dspark  # ar | dspark | dspark64
```

```bash
# Once: load the Qwen backbone and randomly initialize bridge + image head.
python -m gstar.init --config "configs/models/gstar_${HEAD}.json" \
  --llm "$QWEN" --output "outputs/init_${HEAD}"

# Pretrain (single node).
NPROC_PER_NODE=8 bash scripts/train.sh --config configs/pretrain.json \
  --model "outputs/init_${HEAD}" --visual-tokenizer "$VISUAL_TOKENIZER" \
  --output "outputs/pretrain_${HEAD}"

# SFT starts directly from a full pretrain checkpoint.
NPROC_PER_NODE=8 bash scripts/train.sh --config configs/sft.json \
  --model "outputs/pretrain_${HEAD}/checkpoint-1000" --visual-tokenizer "$VISUAL_TOKENIZER" \
  --output "outputs/sft_${HEAD}"
```

For multiple nodes set `NNODES`, `NODE_RANK`, `MASTER_ADDR` and `MASTER_PORT`
on every node (e.g. 16 nodes x 8 GPUs: `NNODES=16 NPROC_PER_NODE=8`).
`--data` overrides the manifest in the recipe; `visual_tokenizer` can also be
set in the recipe instead of passing `--visual-tokenizer`.

**Resolution.** For fixed 512/1024 training pass `--resolution 512` (or 1024)
to `gstar.init` and point `--data` at a matching manifest. Add `--anyres` for
variable aspect ratios within that pixel budget. Inference height/width must be
divisible by `patch_size * merge_size * merger_merge_size`.

**Sampling during training.** Every `evaluation.every` steps (default 1000)
rank zero writes samples to `<output>/samples/step-N/`. Prompts, image size and
CFG are set in the recipe; `every: 0` disables it.

**Switching heads.** Convert an AR checkpoint to a fresh DSpark head while
keeping the backbone, bridge and resolution:

```bash
python -m gstar.init --config configs/models/gstar_dspark.json \
  --source-model /path/to/ar_checkpoint --output outputs/init_dspark
```

Changing the head starts a new optimizer. To resume the same architecture, use
`--resume outputs/.../checkpoint-N`.

**Checkpoints.** Each `checkpoint-N` holds model weights, config, model code
sidecars, tokenizer and Trainer state; SFT checkpoints also store
`role_config.json`. The frozen visual tokenizer is not copied into checkpoints,
so pass the same tokenizer at inference. Resume replays the fixed data order
up to the consumed batch, so keep the manifest, seed, packing, world size and
batch/accumulation settings unchanged.

**Recipe.** Pretraining uses a 10x learning rate for the image head, trains
the full embedding table, AdamW (0.9, 0.96), linear warmup and cosine decay to
10% of the peak LR.

## Tests

```bash
python -m unittest discover -s tests -v
python -m torch.distributed.run --standalone --nproc_per_node=2 tests/smoke_distributed.py
```

Data tests use synthetic images and need no GPU (FlashAttention must still be
importable). Model tests need CUDA and use shrunken model configs. The distributed smoke test runs
DeepSpeed ZeRO-2 on two GPUs.

