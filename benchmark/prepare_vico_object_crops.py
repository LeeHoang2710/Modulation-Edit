#!/usr/bin/env python3
"""Detect and tightly crop one ViCo concept from all its reference photos.

Grounding DINO detects the class noun used by our existing ViCo prompt map.
Review the saved preview before trusting a crop of a compound or small object.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageOps
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

from prepare_vico import CLASSES, LIVE, images, prompt_templates


def expanded_box(box: list[float], size: tuple[int, int], margin: float) -> tuple[int, int, int, int]:
    width, height = size
    x1, y1, x2, y2 = box
    dx, dy = (x2 - x1) * margin, (y2 - y1) * margin
    return (
        max(0, math.floor(x1 - dx)), max(0, math.floor(y1 - dy)),
        min(width, math.ceil(x2 + dx)), min(height, math.ceil(y2 + dy)),
    )


def make_preview(pairs: list[tuple[Image.Image, Image.Image, tuple[int, int, int, int]]], path: Path) -> None:
    tile = 320
    canvas = Image.new("RGB", (tile * 2, tile * len(pairs)), "white")
    for index, (original, cropped, box) in enumerate(pairs):
        marked = original.copy()
        draw = ImageDraw.Draw(marked)
        draw.rectangle(box, outline="red", width=max(3, min(original.size) // 150))
        canvas.paste(ImageOps.contain(marked, (tile, tile)), (0, index * tile))
        canvas.paste(ImageOps.contain(cropped, (tile, tile)), (tile, index * tile))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path, quality=92)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("subject", choices=sorted(CLASSES))
    parser.add_argument("--inputs", default="data/vico")
    parser.add_argument("--output-root", default="data/vico_auto_cropped")
    parser.add_argument("--manifest-root", default="benchmark/manifests")
    parser.add_argument("--detector", default="IDEA-Research/grounding-dino-tiny")
    parser.add_argument("--label", default="", help="detector phrase; default is the subject's ViCo class noun")
    parser.add_argument("--box-threshold", type=float, default=0.25)
    parser.add_argument("--text-threshold", type=float, default=0.25)
    parser.add_argument("--margin", type=float, default=0.12, help="fraction of detected box width/height on each side")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    args = parser.parse_args()
    if not 0 < args.box_threshold < 1 or not 0 < args.text_threshold < 1:
        parser.error("detection thresholds must be between 0 and 1")
    if not 0 <= args.margin <= 0.5:
        parser.error("--margin must be between 0 and 0.5")

    root = Path(__file__).resolve().parents[1]
    inputs = (root / args.inputs).resolve()
    source = inputs / args.subject
    output = (root / args.output_root).resolve() / args.subject
    manifest_path = (root / args.manifest_root).resolve() / f"{args.subject}_auto_cropped.json"
    if not source.is_dir():
        parser.error(f"Missing {source}; run scripts/prepare_vico_assets.sh first")
    photos = images(source)
    if not photos:
        parser.error(f"No reference photos in {source}")
    is_live = args.subject in LIVE
    prompt_file = inputs / ("prompts_live_objects.txt" if is_live else "prompts_nonlive_objects.txt")
    templates = prompt_templates(prompt_file)
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device)
    if device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but is not available")
    phrase = (args.label or CLASSES[args.subject]).strip().rstrip(".")
    print(f"[detector] {args.detector}; phrase={phrase!r}; device={device}", flush=True)
    processor = AutoProcessor.from_pretrained(args.detector)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(args.detector).to(device).eval()

    planned = []
    for photo in photos:
        with Image.open(photo) as loaded:
            original = loaded.convert("RGB")
        inputs_tensor = processor(images=original, text=f"a {phrase}.", return_tensors="pt").to(device)
        with torch.no_grad():
            prediction = model(**inputs_tensor)
        result = processor.post_process_grounded_object_detection(
            prediction, inputs_tensor.input_ids, box_threshold=args.box_threshold,
            text_threshold=args.text_threshold, target_sizes=[(original.height, original.width)],
        )[0]
        if len(result["scores"]) == 0:
            raise SystemExit(
                f"No {phrase!r} detected in {photo}. No crops saved. "
                "Try --label with a clearer noun or a lower --box-threshold, then inspect the preview."
            )
        best = int(result["scores"].argmax().item())
        raw_box = result["boxes"][best].tolist()
        box = expanded_box(raw_box, original.size, args.margin)
        if box[2] <= box[0] or box[3] <= box[1]:
            raise SystemExit(f"Invalid detection box in {photo}: {raw_box}")
        planned.append((photo, original, box, float(result["scores"][best].item())))

    # Only write training inputs after every reference has a valid detection.
    output.mkdir(parents=True, exist_ok=True)
    expected_names = {f"{photo.stem}.png" for photo, _, _, _ in planned}
    for previous in output.iterdir():
        if previous.is_file() and previous.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp"} and previous.name not in expected_names:
            raise SystemExit(f"Unexpected image in crop folder: {previous}; move it aside before training")
    references, records, preview_pairs = [], [], []
    for photo, original, box, score in planned:
        cropped = original.crop(box)
        target = output / f"{photo.stem}.png"
        cropped.save(target)
        references.append(str(target))
        records.append({"source": str(photo), "crop": str(target), "box_xyxy": box, "score": score})
        preview_pairs.append((original, cropped, box))
        print(f"[crop] {photo.name}: score={score:.3f} box={box} -> {target}", flush=True)

    preview = output.parent / f"{args.subject}_crop_preview.jpg"
    make_preview(preview_pairs, preview)
    (output.parent / f"{args.subject}_detections.json").write_text(json.dumps(records, indent=2) + "\n")
    manifest = {"benchmark": f"ViCo {args.subject} automatic-crop smoke test", "prompt_count_per_subject": 31,
                "subjects": [{"subject": args.subject, "class_name": CLASSES[args.subject], "is_live": is_live,
                              "reference_dir": str(output), "reference_images": references,
                              "single_reference": references[0], "prompt_file": str(prompt_file),
                              "prompt_templates": templates}]}
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"[ready] {len(references)} crops; preview={preview}; manifest={manifest_path}", flush=True)


if __name__ == "__main__":
    main()
