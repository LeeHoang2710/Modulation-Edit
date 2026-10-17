#!/usr/bin/env python3
"""Build the exact 16-concept ViCo manifest used by Modulation-Edit.

The downloaded ViCo data lives in ``data/vico``. It contains five live and
eleven non-live concepts with 31 templates each. Templates remain unrendered:
our prompt-conditioned modulation code inserts the generic class noun.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
LIVE = {"black_cat", "brown_cat", "brown_dog", "brown_dog2", "fat_dog"}
CLASSES = {
    "alarm_clock": "clock", "black_cat": "cat", "brown_cat": "cat",
    "brown_dog": "dog", "brown_dog2": "dog", "brown_teddybear": "teddybear",
    "cat_statue": "statue", "duck_toy": "toy", "elephant_statue": "statue",
    "fat_dog": "dog", "monster_toy": "toy", "pink_sunglasses": "sunglasses",
    "red_teapot": "teapot", "red_vase": "vase", "tortoise_plushy": "plushy",
    "wooden_barn": "barn",
}


def images(directory: Path) -> list[Path]:
    return sorted(path for path in directory.iterdir() if path.suffix.lower() in IMAGE_EXTS)


def prompt_templates(path: Path) -> list[str]:
    templates = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if len(templates) != 31:
        raise ValueError(f"Expected 31 prompt templates in {path}, found {len(templates)}")
    if any(template.count("{}") != 1 for template in templates):
        raise ValueError(f"Every ViCo template must contain exactly one '{{}}': {path}")
    return templates


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", default="data/vico")
    parser.add_argument("--output", default="benchmark/manifests/vico.json")
    args = parser.parse_args()
    root = Path.cwd()
    inputs = Path(args.inputs).resolve()
    if not inputs.is_dir():
        raise SystemExit(f"ViCo inputs are missing: {inputs}. Run scripts/prepare_vico_assets.sh first.")
    live_prompts = prompt_templates(inputs / "prompts_live_objects.txt")
    nonlive_prompts = prompt_templates(inputs / "prompts_nonlive_objects.txt")
    subjects = []
    for subject, class_name in sorted(CLASSES.items()):
        reference_dir = inputs / subject
        reference_images = images(reference_dir)
        if not reference_images:
            raise SystemExit(f"No reference images for ViCo subject: {reference_dir}")
        is_live = subject in LIVE
        prompt_file = inputs / ("prompts_live_objects.txt" if is_live else "prompts_nonlive_objects.txt")
        subjects.append({
            "subject": subject,
            "class_name": class_name,
            "is_live": is_live,
            "reference_dir": str(reference_dir.relative_to(root)),
            "reference_images": [str(path.relative_to(root)) for path in reference_images],
            "single_reference": str(reference_images[0].relative_to(root)),
            "prompt_file": str(prompt_file.relative_to(root)),
            "prompt_templates": live_prompts if is_live else nonlive_prompts,
        })
    payload = {"benchmark": "ViCo personalization benchmark", "prompt_count_per_subject": 31,
               "subjects": subjects}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n")
    with output.with_suffix(".csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["subject", "class_name", "is_live", "references", "prompts"])
        writer.writeheader()
        for item in subjects:
            writer.writerow({"subject": item["subject"], "class_name": item["class_name"], "is_live": item["is_live"],
                             "references": len(item["reference_images"]), "prompts": len(item["prompt_templates"])})
    print(f"[manifest] {output}: {len(subjects)} subjects × 31 prompts")


if __name__ == "__main__":
    main()
