#!/usr/bin/env python3
"""Unified DreamBooth metrics for all five baselines.

Expected generated-image layout:
  <results>/<model>/<subject>/<prompt-slug>/prompt.txt
  <results>/<model>/<subject>/<prompt-slug>/seed_000.png
  <results>/<model>/<subject>/timing.json

`timing.json` is written by a runner and contains preparation_seconds and
inference_seconds.  The summary exposes the requested preparation/object and
inference/image measurements together with DINO, CLIP-I, and CLIP-T.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import timm
import torch
import torchvision.transforms as T
from PIL import Image
from transformers import CLIPImageProcessor, CLIPModel, CLIPTokenizer

METRICS = ("dino", "clip_i", "clip_t")
DINO_TRANSFORM = T.Compose([
    T.Resize(256, interpolation=T.InterpolationMode.BICUBIC), T.CenterCrop(224), T.ToTensor(),
    T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


@torch.inference_mode()
def embed_clip_images(paths: list[Path], model, processor, device: torch.device, batch_size: int) -> torch.Tensor:
    out = []
    for start in range(0, len(paths), batch_size):
        images = [Image.open(path).convert("RGB") for path in paths[start:start + batch_size]]
        batch = processor(images=images, return_tensors="pt").pixel_values.to(device)
        out.append(torch.nn.functional.normalize(model.get_image_features(pixel_values=batch).float(), dim=-1).cpu())
    return torch.cat(out)


@torch.inference_mode()
def embed_dino(paths: list[Path], model, device: torch.device, batch_size: int) -> torch.Tensor:
    out = []
    for start in range(0, len(paths), batch_size):
        batch = torch.stack([DINO_TRANSFORM(Image.open(path).convert("RGB")) for path in paths[start:start + batch_size]]).to(device)
        out.append(torch.nn.functional.normalize(model(batch).float(), dim=-1).cpu())
    return torch.cat(out)


@torch.inference_mode()
def embed_text(prompts: list[str], model, tokenizer, device: torch.device) -> torch.Tensor:
    tokens = tokenizer(prompts, padding=True, truncation=True, return_tensors="pt").to(device)
    return torch.nn.functional.normalize(model.get_text_features(**tokens).float(), dim=-1).cpu()


def generated_paths(prompt_dir: Path) -> list[Path]:
    return sorted(p for p in prompt_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS and re.match(r"(?:seed_)?\d+", p.stem))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="benchmark_outputs")
    parser.add_argument("--dataset", default="data/dreambooth/dataset")
    parser.add_argument("--report-dir", default="benchmark_reports")
    parser.add_argument("--models", default="all", help="all or comma-separated output model directory names")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--clip-model", default="openai/clip-vit-base-patch32")
    parser.add_argument("--dino-model", default="vit_small_patch16_224.dino")
    args = parser.parse_args()
    root, dataset, report_dir = Path(args.results), Path(args.dataset), Path(args.report_dir)
    model_dirs = [p for p in sorted(root.iterdir()) if p.is_dir()] if args.models == "all" else [root / x for x in args.models.split(",")]
    if not model_dirs:
        raise SystemExit(f"No model output directories in {root}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clip = CLIPModel.from_pretrained(args.clip_model, use_safetensors=True).to(device).eval()
    processor = CLIPImageProcessor.from_pretrained(args.clip_model)
    tokenizer = CLIPTokenizer.from_pretrained(args.clip_model)
    dino = timm.create_model(args.dino_model, pretrained=True, num_classes=0).to(device).eval()
    cache: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    rows: list[dict] = []

    for model_dir in model_dirs:
        if not model_dir.is_dir():
            print(f"[skip] output missing: {model_dir}")
            continue
        model_name = model_dir.name
        for subject_dir in sorted(p for p in model_dir.iterdir() if p.is_dir()):
            subject = subject_dir.name
            reference_paths = sorted(p for p in (dataset / subject).iterdir() if p.suffix.lower() in IMAGE_EXTS)
            if not reference_paths:
                print(f"[skip] {model_name}/{subject}: no references")
                continue
            if subject not in cache:
                cache[subject] = (
                    embed_dino(reference_paths, dino, device, args.batch_size),
                    embed_clip_images(reference_paths, clip, processor, device, args.batch_size),
                )
            ref_dino, ref_clip = cache[subject]
            timing_path = subject_dir / "timing.json"
            timing = json.loads(timing_path.read_text()) if timing_path.is_file() else {}
            prompt_dirs = [p for p in sorted(subject_dir.iterdir()) if p.is_dir() and (p / "prompt.txt").is_file()]
            for prompt_dir in prompt_dirs:
                prompt = (prompt_dir / "prompt.txt").read_text().strip()
                images = generated_paths(prompt_dir)
                if not images:
                    continue
                dino_scores = (embed_dino(images, dino, device, args.batch_size) @ ref_dino.T).mean(dim=1).tolist()
                clip_images = embed_clip_images(images, clip, processor, device, args.batch_size)
                clip_i_scores = (clip_images @ ref_clip.T).mean(dim=1).tolist()
                clip_t_scores = (clip_images @ embed_text([prompt], clip, tokenizer, device).T)[:, 0].tolist()
                for index, (path, dino_score, clip_i_score, clip_t_score) in enumerate(zip(images, dino_scores, clip_i_scores, clip_t_scores)):
                    rows.append({
                        "model": model_name, "subject": subject, "prompt": prompt, "image": str(path), "sample": index,
                        "dino": dino_score, "clip_i": clip_i_score, "clip_t": clip_t_score,
                        "preparation_seconds_per_object": timing.get("preparation_seconds"),
                        "inference_seconds_per_image": timing.get("inference_seconds_per_image"),
                        "reference_protocol": timing.get("reference_protocol", "unknown"),
                    })
            print(f"[scored] {model_name}/{subject}: {len(prompt_dirs)} prompts")
    if not rows:
        raise SystemExit("No generated images found in the required benchmark layout.")
    write_csv(report_dir / "per_image_metrics.csv", rows, list(rows[0]))
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        groups[(row["model"], row["subject"])].append(row)
    subject_rows = []
    for (model, subject), group in sorted(groups.items()):
        subject_rows.append({
            "model": model, "subject": subject, "images": len(group),
            **{metric: float(np.mean([float(r[metric]) for r in group])) for metric in METRICS},
            "preparation_seconds_per_object": group[0]["preparation_seconds_per_object"],
            "inference_seconds_per_image": group[0]["inference_seconds_per_image"],
            "reference_protocol": group[0]["reference_protocol"],
        })
    write_csv(report_dir / "per_subject_metrics.csv", subject_rows, list(subject_rows[0]))
    summary = []
    for model, group in sorted((m, [r for r in subject_rows if r["model"] == m]) for m in {r["model"] for r in subject_rows}):
        summary.append({
            "model": model, "objects": len(group), "inference_images": sum(r["images"] for r in group),
            **{metric: float(np.mean([r[metric] for r in group])) for metric in METRICS},
            "preparation_seconds_per_object": float(np.nanmean([float(r["preparation_seconds_per_object"]) for r in group if r["preparation_seconds_per_object"] is not None])),
            "inference_seconds_per_image": float(np.nanmean([float(r["inference_seconds_per_image"]) for r in group if r["inference_seconds_per_image"] is not None])),
        })
    write_csv(report_dir / "summary.csv", summary, list(summary[0]))
    print(f"[saved] {report_dir / 'per_image_metrics.csv'}")
    print(f"[saved] {report_dir / 'per_subject_metrics.csv'}")
    print(f"[saved] {report_dir / 'summary.csv'}")


if __name__ == "__main__":
    main()
