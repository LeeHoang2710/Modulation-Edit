#!/usr/bin/env python3
"""Create a deterministic manifest for the official DreamBooth benchmark.

The manifest is intentionally model-agnostic.  It preserves every reference
image, selects reference 0 for explicitly single-reference methods, and gives
each subject the 25 evaluation prompts used by DreamBooth.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
LIVE_SUBJECTS = {"backpack_dog", "cat", "cat2", "dog", "dog2", "dog3", "dog5", "dog6", "dog7", "dog8"}

COMMON = [
    "a {cls} in the jungle", "a {cls} in the snow", "a {cls} on the beach",
    "a {cls} on a cobblestone street", "a {cls} on top of pink fabric",
    "a {cls} on top of a wooden floor", "a {cls} with a city in the background",
    "a {cls} with a mountain in the background", "a {cls} with a blue house in the background",
    "a {cls} on top of a purple rug in a forest",
]
PROPERTIES = ["a red {cls}", "a purple {cls}", "a shiny {cls}", "a wet {cls}", "a cube shaped {cls}"]
OBJECT = COMMON + [
    "a {cls} with a wheat field in the background", "a {cls} with a tree and autumn leaves in the background",
    "a {cls} with the Eiffel Tower in the background", "a {cls} floating on top of water",
    "a {cls} floating in an ocean of milk", "a {cls} on top of green grass with sunflowers around it",
    "a {cls} on top of a mirror", "a {cls} on top of the sidewalk in a crowded street",
    "a {cls} on top of a dirt road", "a {cls} on top of a white rug",
] + PROPERTIES
LIVE = COMMON + [
    "a {cls} wearing a red hat", "a {cls} wearing a santa hat", "a {cls} wearing a rainbow scarf",
    "a {cls} wearing a black top hat and a monocle", "a {cls} in a chef outfit",
    "a {cls} in a firefighter outfit", "a {cls} in a police outfit", "a {cls} wearing pink glasses",
    "a {cls} wearing a yellow shirt", "a {cls} in a purple wizard outfit",
] + PROPERTIES


def image_paths(subject_dir: Path) -> list[Path]:
    return sorted(p.resolve() for p in subject_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)


def read_classes(dataset: Path) -> dict[str, str]:
    """Read official `prompts_and_classes.txt` when present, with safe fallback."""
    path = dataset / "prompts_and_classes.txt"
    classes: dict[str, str] = {}
    if not path.is_file():
        return classes
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        # Official copies have appeared as `subject class`, `subject,class`,
        # and `subject\tclass`; only split once because classes can be phrases.
        for separator in ("\t", ",", " "):
            if separator in line:
                key, value = line.split(separator, 1)
                classes[key.strip()] = value.strip()
                break
    return classes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="data/dreambooth/dataset")
    parser.add_argument("--output", default="benchmark/manifests/dreambooth.json")
    parser.add_argument("--subjects", default="all", help="all or comma-separated subject directory names")
    args = parser.parse_args()

    dataset = Path(args.dataset).resolve()
    if not dataset.is_dir():
        raise SystemExit(f"DreamBooth dataset is missing: {dataset}")
    known = read_classes(dataset)
    all_subjects = [p.name for p in sorted(dataset.iterdir()) if p.is_dir() and image_paths(p)]
    selected = all_subjects if args.subjects == "all" else [s.strip() for s in args.subjects.split(",") if s.strip()]
    missing = sorted(set(selected) - set(all_subjects))
    if missing:
        raise SystemExit(f"Unknown/missing subject directories: {', '.join(missing)}")

    entries = []
    for subject in selected:
        refs = image_paths(dataset / subject)
        cls = known.get(subject, subject.replace("_", " "))
        templates = LIVE if subject in LIVE_SUBJECTS else OBJECT
        entries.append({
            "subject": subject,
            "class_name": cls,
            "reference_images": [str(p) for p in refs],
            "single_reference": str(refs[0]),
            "prompts": [t.format(cls=cls) for t in templates],
            "protocol": {
                "multi_reference_models": "all reference_images",
                "single_reference_models": "single_reference (deterministic first image)",
            },
        })
    payload = {"dataset": str(dataset), "subjects": entries, "prompt_count_per_subject": 25}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n")
    with output.with_suffix(".csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["subject", "class_name", "references", "single_reference", "prompts"])
        writer.writeheader()
        for item in entries:
            writer.writerow({"subject": item["subject"], "class_name": item["class_name"],
                             "references": len(item["reference_images"]), "single_reference": item["single_reference"],
                             "prompts": len(item["prompts"])})
    print(f"[manifest] {output}: {len(entries)} subjects × 25 prompts")


if __name__ == "__main__":
    main()
