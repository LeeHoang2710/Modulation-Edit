#!/usr/bin/env python3
"""Reproduce tight, non-generative crops of ARBooth's five alarm-clock photos.

Coordinates are source-image pixels in (left, top, right, bottom) order.
The source images stay untouched. The resulting manifest redirects only the
training references; evaluation can still use the original ViCo photos.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
from pathlib import Path


CROPS = {
    "1.jpeg": ((567, 567), (25, 50, 375, 525)),
    "2.jpeg": ((1200, 1600), (330, 300, 745, 860)),
    "3.jpeg": ((567, 567), (20, 0, 550, 550)),
    "4.jpeg": ((567, 567), (125, 80, 395, 350)),
    "5.jpeg": ((1175, 1174), (145, 0, 1020, 1150)),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="models/ARBooth/inputs/alarm_clock")
    parser.add_argument("--output", default="data/vico_alarm_clock_tight_crop/alarm_clock")
    parser.add_argument("--manifest", default="benchmark/manifests/alarm_clock_tight_crop.json")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    source = (root / args.source).resolve()
    output = (root / args.output).resolve()
    manifest_path = (root / args.manifest).resolve()
    prompt_file = source.parent / "prompts_nonlive_objects.txt"
    if not source.is_dir() or not prompt_file.is_file():
        parser.error(f"Missing ARBooth alarm-clock inputs or prompts under {source.parent}; initialize the submodule")

    templates = [line.strip() for line in prompt_file.read_text().splitlines() if line.strip()]
    if len(templates) != 31 or any(template.count("{}") != 1 for template in templates):
        parser.error(f"Expected 31 single-placeholder prompts in {prompt_file}")

    try:
        from PIL import Image
    except ImportError:
        Image = None
        if shutil.which("sips") is None:
            parser.error("Cropping requires Pillow or macOS sips")

    output.mkdir(parents=True, exist_ok=True)
    references = []
    for name, (expected_size, box) in CROPS.items():
        src = source / name
        if not src.is_file():
            parser.error(f"Missing source image: {src}")
        target = output / f"{Path(name).stem}.png"
        if Image is not None:
            with Image.open(src) as image:
                if image.size != expected_size:
                    parser.error(f"{src}: expected {expected_size}, found {image.size}; crop boxes need review")
                cropped = image.convert("RGB").crop(box)
                cropped.save(target)
                cropped_size = cropped.size
        else:
            metadata = subprocess.run(
                ["sips", "-g", "pixelWidth", "-g", "pixelHeight", str(src)],
                check=True, capture_output=True, text=True,
            ).stdout
            width = re.search(r"pixelWidth:\s*(\d+)", metadata)
            height = re.search(r"pixelHeight:\s*(\d+)", metadata)
            source_size = (int(width.group(1)), int(height.group(1))) if width and height else None
            if source_size != expected_size:
                parser.error(f"{src}: expected {expected_size}, found {source_size}; crop boxes need review")
            left, top, right, bottom = box
            cropped_size = (right - left, bottom - top)
            subprocess.run([
                "sips", "-s", "format", "png", "-c", str(cropped_size[1]), str(cropped_size[0]),
                "--cropOffset", str(top), str(left), str(src), "--out", str(target),
            ], check=True, capture_output=True, text=True)
        references.append(str(target))
        print(f"[crop] {src.name}: {box} -> {target} ({cropped_size[0]}x{cropped_size[1]})")

    manifest = {
        "benchmark": "ViCo alarm_clock tight-crop smoke test",
        "prompt_count_per_subject": 31,
        "subjects": [{
            "subject": "alarm_clock",
            "class_name": "clock",  # Matches the current ViCo prompt mapping.
            "is_live": False,
            "reference_dir": str(output),
            "reference_images": references,
            "single_reference": references[0],
            "prompt_file": str(prompt_file),
            "prompt_templates": templates,
        }],
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"[manifest] {manifest_path}: 1 subject, 5 cropped references, 31 available prompts")


if __name__ == "__main__":
    main()
