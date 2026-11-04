"""Convert image-caption pairs into the JSONL records read by gstar.data.

Input is a JSONL file with one {"image": ..., "caption": ...} object per line,
or a CSV file with a header row (column names are configurable). `image`
must be a local file path.

Examples:
  python tools/prepare_data.py pairs.jsonl data/pretrain/general/part0.jsonl
  python tools/prepare_data.py pairs.csv data/pretrain/general/part0.jsonl \
      --preprocess bucketresize_256
  python tools/prepare_data.py pairs.jsonl data/sft/t2i/part0.jsonl --stage sft
"""
import argparse
import csv
import json
from pathlib import Path

def read_pairs(path, image_key, caption_key):
    with open(path, encoding="utf-8", newline="") as handle:
        if path.suffix == ".csv":
            records = csv.DictReader(handle)
        else:
            records = (json.loads(line) for line in handle if line.strip())
        for record in records:
            yield record[image_key], record[caption_key]


def descriptor(image, preprocess):
    # "file" keeps absolute paths intact; untyped JSON paths are treated as relative keys.
    info = {"type": "file", "path": str(Path(image).resolve())}
    if preprocess:
        info["preprocess"] = preprocess
    return json.dumps(info, ensure_ascii=False)


def record(caption, image, stage, task, instruction):
    image = "<img_start>" + image + "<img_end>"
    if stage == "pretrain":
        if task == "t2i":
            return {"input": caption + "<trainable_start>" + image + "<trainable_end>"}
        return {"input": image + "<trainable_start>" + caption + "<trainable_end>"}
    if task == "t2i":
        turns = [("user", caption), ("assistant", image)]
    else:
        turns = [("user", image + instruction), ("assistant", caption)]
    return {"messages": [{"role": role, "content": content} for role, content in turns]}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", type=Path, help="JSONL or CSV of image-caption pairs")
    parser.add_argument("output", type=Path, help="Output JSONL")
    parser.add_argument("--stage", choices=("pretrain", "sft"), default="pretrain")
    parser.add_argument("--task", choices=("t2i", "i2t"), default="t2i",
                        help="t2i supervises the image tokens; i2t supervises the caption")
    parser.add_argument("--image-key", default="image")
    parser.add_argument("--caption-key", default="caption")
    parser.add_argument("--instruction", default="Describe this image.", help="User turn for SFT i2t")
    parser.add_argument("--preprocess", help="Per-image resize rule, e.g. bucketresize_256 or fixres_512")
    args = parser.parse_args()
    if args.preprocess and not args.preprocess.startswith(("fixres_", "bucketresize_")):
        parser.error("--preprocess must look like fixres_N or bucketresize_N")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with args.output.open("w", encoding="utf-8") as out:
        for image, caption in read_pairs(args.input, args.image_key, args.caption_key):
            if not caption or not caption.strip():
                continue
            if not Path(image).is_file():
                raise FileNotFoundError(image)
            row = record(caption.strip(), descriptor(image, args.preprocess),
                         args.stage, args.task, args.instruction)
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    print("wrote %d records to %s" % (count, args.output))


if __name__ == "__main__":
    main()
