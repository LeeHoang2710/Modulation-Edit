#!/usr/bin/env python3
"""Evaluate ViCo outputs with ARBooth metrics plus CSD-VAR descriptors."""
from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path
from statistics import mean

import numpy as np
import torch
from PIL import Image


IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".webp"}
METRICS = ("dino", "clip_i", "clip_t", "clip_t_cls", "div", "pres", "csd_content", "csd_style")


def image_paths(directory: Path) -> list[Path]:
    return sorted(path for path in directory.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)


def image_tensor(path: Path) -> torch.Tensor:
    with Image.open(path) as opened:
        image = opened.convert("RGB").resize((256, 256), Image.Resampling.BICUBIC)
        array = np.asarray(image, dtype=np.uint8)
    return torch.from_numpy((array / 127.5 - 1.0).astype(np.float32)).permute(2, 0, 1)


def load_images(directory: Path) -> torch.Tensor:
    paths = image_paths(directory)
    if not paths:
        raise FileNotFoundError(f"no images found in {directory}")
    return torch.stack([image_tensor(path) for path in paths])


def average(values):
    values = [float(value) for value in values if value != ""]
    return mean(values) if values else ""


def write_csv(path: Path, rows: list[dict], fields: list[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result_dir", required=True)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--arbooth_root", default="/workspace/ARBooth")
    parser.add_argument("--csd_root", default="/workspace/CSD-VAR")
    args = parser.parse_args()

    result_dir = Path(args.result_dir).resolve()
    data_dir = Path(args.data_dir).resolve()
    arbooth_root = Path(args.arbooth_root).resolve()
    csd_root = Path(args.csd_root).resolve()
    sys.path.insert(0, str(arbooth_root))
    sys.path.insert(0, str(csd_root))

    from booth_evaluation.clip_eval import CLIPEvaluator
    from booth_evaluation.dino_eval import DINOEvaluator
    from booth_evaluation.lpips_eval import LPIPSEvaluator
    from vis.metrics import CSDEvaluator

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clip_eval = CLIPEvaluator(device)
    dino_eval = DINOEvaluator(device)
    dino_eval.model = dino_eval.model.to(device).eval()
    lpips_eval = LPIPSEvaluator(device)
    csd_eval = CSDEvaluator(
        device,
        model_path=str(csd_root / "evaluator/CSD/csd_models/pytorch_model.bin"),
    )

    with (data_dir / "subjects.csv").open() as handle:
        subjects = list(csv.DictReader(handle))

    subject_rows = []
    for item in subjects:
        subject = item["subject"]
        subject_result = result_dir / subject
        if not subject_result.is_dir():
            continue
        source = load_images(data_dir / f"{subject}+photo" / "imgs")
        source_device = source.to(device)
        templates = [line.strip() for line in (data_dir / item["prompts"]).read_text().splitlines() if line.strip()]
        prompt_rows = []

        for prompt_index, template in enumerate(templates):
            prompt_dir = subject_result / f"prompt_{prompt_index:02d}"
            if not prompt_dir.is_dir():
                continue
            generated = load_images(prompt_dir)
            generated_device = generated.to(device)
            rendered_class = template.format(item["class_name"])

            with torch.inference_mode():
                dino = float(dino_eval.img_to_img_similarity(source_device, generated_device).item())
                clip_i = float(clip_eval.img_to_img_similarity(source, generated).item())
                clip_t = float(clip_eval.txt_to_img_similarity(template, generated).item())
                clip_t_cls = float(clip_eval.txt_to_img_similarity(rendered_class, generated).item())
                div = float(lpips_eval.compute_pairwise_distance(generated))

            csd_content_values = []
            csd_style_values = []
            for target in generated:
                content, style = csd_eval.calculate_similarities(source, target)
                csd_content_values.append(content)
                csd_style_values.append(style)

            prompt_rows.append({
                "subject": subject,
                "prompt_index": prompt_index,
                "prompt": template,
                "rendered_class_prompt": rendered_class,
                "images": len(generated),
                "dino": dino,
                "clip_i": clip_i,
                "clip_t": clip_t,
                "clip_t_cls": clip_t_cls,
                "div": div,
                "pres": "",
                "csd_content": average(csd_content_values),
                "csd_style": average(csd_style_values),
            })

        if not prompt_rows:
            continue
        write_csv(
            subject_result / f"{subject}_metrics.csv",
            prompt_rows,
            list(prompt_rows[0]),
        )
        summary = {
            "subject": subject,
            "class_name": item["class_name"],
            "prompts": item["prompts"],
            **{metric: average([row[metric] for row in prompt_rows]) for metric in METRICS},
        }
        subject_rows.append(summary)
        print(
            f"[{subject}] " + " ".join(
                f"{metric}={summary[metric]:.5f}" for metric in METRICS if summary[metric] != ""
            ),
            flush=True,
        )

    if not subject_rows:
        raise SystemExit(f"no generated subjects found under {result_dir}")

    groups = {
        "overall": subject_rows,
        "live": [row for row in subject_rows if row["prompts"].startswith("prompts_live")],
        "object": [row for row in subject_rows if row["prompts"].startswith("prompts_nonlive")],
    }
    aggregate_rows = []
    for group, rows in groups.items():
        if not rows:
            continue
        aggregate_rows.append({
            "subject": f"[{group}] ({len(rows)} subjects)",
            "class_name": "",
            "prompts": "",
            **{metric: average([row[metric] for row in rows]) for metric in METRICS},
        })

    write_csv(
        result_dir / "metrics.csv",
        subject_rows + aggregate_rows,
        ["subject", "class_name", "prompts", *METRICS],
    )
    overall = aggregate_rows[0]
    print(
        "[overall] " + " ".join(
            f"{metric}={overall[metric]:.5f}" for metric in METRICS if overall[metric] != ""
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
