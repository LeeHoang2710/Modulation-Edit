#!/usr/bin/env python3
"""Evaluate ViCo outputs with the metric implementation used by ARBooth.

This script deliberately follows ``models/ARBooth/tools/evaluate.py`` and its
``DINOEvaluator`` / ``CLIPEvaluator`` rather than the older generic ViCo
evaluator in this repository.  In particular it uses:

* DINO ViT-S/16 loaded from ``facebookresearch/dino:main``;
* OpenAI CLIP ViT-B/32;
* ARBooth's 256 x 256 bicubic resize and [-1, 1] image range;
* all-pairs mean similarity between every reference and generated image; and
* a mean over prompts, then a mean over the 16 concepts.

The official ARBooth evaluator passes the *unrendered* template (for example
``a {} in the snow``) to CLIP-T.  ``--text-protocol arbooth-code`` is therefore
the default for a strict code-level reproduction of their reported table.  The
``rendered-class`` option is included only for an explicitly different,
human-readable CLIP-T diagnostic.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".webp"}
METRIC_FIELDS = ("i_dino", "i_clip", "t_clip")


def image_paths(directory: Path) -> list[Path]:
    if not directory.is_dir():
        return []
    return sorted(path for path in directory.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS)


def arbooth_image_tensor(path: Path) -> torch.Tensor:
    """Match PersonalizedBase(..., size=256, center_crop=False, flip_p=0)."""
    with Image.open(path) as opened:
        image = opened.convert("RGB")
        image = image.resize((256, 256), resample=Image.Resampling.BICUBIC)
        array = np.asarray(image, dtype=np.uint8)
    return torch.from_numpy((array / 127.5 - 1.0).astype(np.float32)).permute(2, 0, 1)


def batches(items: list[Path], size: int) -> Iterable[list[Path]]:
    for start in range(0, len(items), size):
        yield items[start:start + size]


@torch.inference_mode()
def dino_features(paths: list[Path], model, device: torch.device, batch_size: int) -> torch.Tensor:
    outputs = []
    for group in batches(paths, batch_size):
        pixels = torch.stack([arbooth_image_tensor(path) for path in group]).to(device)
        features = model(pixels).detach()
        features /= features.clone().norm(dim=-1, keepdim=True)
        outputs.append(features.cpu())
    return torch.cat(outputs, dim=0)


@torch.inference_mode()
def clip_image_features(paths: list[Path], model, preprocess, device: torch.device, batch_size: int) -> torch.Tensor:
    outputs = []
    for group in batches(paths, batch_size):
        pixels = torch.stack([arbooth_image_tensor(path) for path in group])
        features = model.encode_image(preprocess(pixels).to(device))
        features /= features.clone().norm(dim=-1, keepdim=True)
        outputs.append(features.cpu())
    return torch.cat(outputs, dim=0)


@torch.inference_mode()
def clip_text_feature(text: str, clip_module, model, device: torch.device) -> torch.Tensor:
    feature = model.encode_text(clip_module.tokenize(text).to(device)).detach()
    return feature / feature.norm(dim=-1, keepdim=True)


def mean_cross_similarity(left: torch.Tensor, right: torch.Tensor) -> float:
    return float((left @ right.T).mean().item())


def select_highest_dino(reference_features: torch.Tensor, generated_features: torch.Tensor) -> int:
    """Return the generated image with the highest mean DINO similarity.

    Each candidate is compared to every ViCo reference of the subject.  This
    is a per-prompt oracle selection, so it must be reported separately from
    the standard mean-over-eight evaluation.
    """
    return int((reference_features @ generated_features.T).mean(dim=0).argmax().item())


def output_images(prompt_dir: Path) -> list[Path]:
    # Both official ARBooth (00.jpg) and our runner (seed_000.png) satisfy this.
    return [path for path in image_paths(prompt_dir) if re.fullmatch(r"(?:seed_)?\d+", path.stem)]


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load_timing(subject_dir: Path) -> dict:
    path = subject_dir / "timing.json"
    return json.loads(path.read_text()) if path.is_file() else {}


def prompt_text(template: str, class_name: str, protocol: str) -> str:
    if protocol == "arbooth-code":
        # This is the exact string supplied to CLIPEvaluator in ARBooth's
        # tools/evaluate.py, despite the paper describing it as placeholder-free.
        return template
    return template.format(class_name)


def selected_subjects(manifest: dict, requested: str) -> list[dict]:
    wanted = None if requested == "all" else {part.strip() for part in requested.split(",") if part.strip()}
    selected = [item for item in manifest["subjects"] if wanted is None or item["subject"] in wanted]
    unknown = set() if wanted is None else wanted - {item["subject"] for item in selected}
    if unknown:
        raise SystemExit(f"Unknown subject(s): {', '.join(sorted(unknown))}")
    if not selected:
        raise SystemExit("No subjects selected")
    return selected


def finite_mean(values: list[float | None]) -> float | None:
    valid = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return float(np.mean(valid)) if valid else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", required=True,
                        help="model root with <subject>/prompt_XX/(00.jpg|seed_000.png) outputs")
    parser.add_argument("--manifest", default="benchmark/manifests/vico.json")
    parser.add_argument("--dataset", default="data/vico")
    parser.add_argument("--report-dir", required=True)
    parser.add_argument("--model-name", default="model")
    parser.add_argument("--subjects", default="all")
    parser.add_argument("--max-prompts", type=int, default=0, help="0 evaluates all 31 prompts")
    parser.add_argument("--expected-images-per-prompt", type=int, default=8)
    parser.add_argument("--selection", choices=["mean8", "best-dino"], default="mean8",
                        help="mean8 scores all outputs; best-dino retains the output with highest I-DINO per prompt")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--text-protocol", choices=["arbooth-code", "rendered-class"], default="arbooth-code")
    parser.add_argument("--clip-device", default="auto", help="auto, cpu, or cuda")
    parser.add_argument("--dino-device", default="cpu", help="cpu matches the released ARBooth evaluator; cuda is faster")
    parser.add_argument("--torch-hub-dir", default="weights/metrics/torchhub")
    parser.add_argument("--clip-download-root", default="weights/metrics/clip")
    args = parser.parse_args()
    if args.max_prompts < 0:
        parser.error("--max-prompts must be non-negative")
    if args.expected_images_per_prompt < 1 or args.batch_size < 1:
        parser.error("--expected-images-per-prompt and --batch-size must be positive")

    root = Path(__file__).resolve().parents[1]
    results = Path(args.results).resolve()
    report_dir = Path(args.report_dir).resolve()
    manifest = json.loads((root / args.manifest).read_text())
    dataset = (root / args.dataset).resolve()
    subjects = selected_subjects(manifest, args.subjects)
    if not results.is_dir():
        raise SystemExit(f"Generated result directory is missing: {results}")

    try:
        import clip  # OpenAI's official CLIP package, not Hugging Face CLIP.
    except ImportError as error:
        raise SystemExit("OpenAI CLIP is required. Run scripts/setup_vico_vastai.sh first.") from error

    clip_device = torch.device("cuda" if args.clip_device == "auto" and torch.cuda.is_available() else args.clip_device)
    dino_device = torch.device(args.dino_device)
    if clip_device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--clip-device cuda was requested but CUDA is unavailable")
    if dino_device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--dino-device cuda was requested but CUDA is unavailable")

    torch_hub_dir = (root / args.torch_hub_dir).resolve()
    torch_hub_dir.mkdir(parents=True, exist_ok=True)
    torch.hub.set_dir(str(torch_hub_dir))
    # Same official DINO checkpoint as ARBooth's DINOEvaluator.
    dino = torch.hub.load("facebookresearch/dino:main", "dino_vits16", trust_repo=True).to(dino_device)
    clip_root = (root / args.clip_download_root).resolve()
    clip_root.mkdir(parents=True, exist_ok=True)
    clip_model, clip_preprocess = clip.load("ViT-B/32", device=clip_device, download_root=str(clip_root))
    # Verbatim preprocessing logic from ARBooth's CLIPEvaluator:
    # input tensors are already RGB and in [-1, 1], so it intentionally skips
    # CLIP's ConvertRGB and ToTensor transforms.
    clip_preprocess_exact = transforms.Compose(
        [transforms.Normalize(mean=[-1.0, -1.0, -1.0], std=[2.0, 2.0, 2.0])]
        + clip_preprocess.transforms[:2]
        + clip_preprocess.transforms[4:]
    )

    subject_rows: list[dict] = []
    prompt_rows: list[dict] = []
    for item in subjects:
        subject = item["subject"]
        subject_dir = results / subject
        ref_paths = image_paths(dataset / subject)
        if not ref_paths:
            raise SystemExit(f"No ViCo references found for {subject}: {dataset / subject}")
        if not subject_dir.is_dir():
            raise SystemExit(f"Missing generated output for {subject}: {subject_dir}")
        ref_dino = dino_features(ref_paths, dino, dino_device, args.batch_size)
        ref_clip = clip_image_features(ref_paths, clip_model, clip_preprocess_exact, clip_device, args.batch_size)
        timing = load_timing(subject_dir)
        templates = item["prompt_templates"][:args.max_prompts or None]
        concept_scores: dict[str, list[float]] = defaultdict(list)

        for prompt_index, template in enumerate(templates):
            prompt_dir = subject_dir / f"prompt_{prompt_index:02d}"
            generated = output_images(prompt_dir)
            if len(generated) != args.expected_images_per_prompt:
                raise SystemExit(
                    f"{subject}/prompt_{prompt_index:02d}: expected "
                    f"{args.expected_images_per_prompt} generated images, found {len(generated)} in {prompt_dir}"
                )
            metric_prompt = prompt_text(template, item["class_name"], args.text_protocol)
            generated_dino = dino_features(generated, dino, dino_device, args.batch_size)
            generated_clip = clip_image_features(generated, clip_model, clip_preprocess_exact, clip_device, args.batch_size)
            text = clip_text_feature(metric_prompt, clip, clip_model, clip_device).cpu()
            selected_image = ""
            if args.selection == "best-dino":
                selected = select_highest_dino(ref_dino, generated_dino)
                generated_for_score_dino = generated_dino[selected:selected + 1]
                generated_for_score_clip = generated_clip[selected:selected + 1]
                selected_image = generated[selected].name
            else:
                generated_for_score_dino = generated_dino
                generated_for_score_clip = generated_clip
            scores = {
                "i_dino": mean_cross_similarity(ref_dino, generated_for_score_dino),
                "i_clip": mean_cross_similarity(ref_clip, generated_for_score_clip),
                "t_clip": mean_cross_similarity(text, generated_for_score_clip),
            }
            for metric, value in scores.items():
                concept_scores[metric].append(value)
            prompt_rows.append({
                "model": args.model_name, "subject": subject, "prompt_index": prompt_index,
                "template": template, "metric_prompt": metric_prompt,
                "generated_images": len(generated), "selection": args.selection,
                "selected_image": selected_image, **scores,
            })

        subject_rows.append({
            "model": args.model_name,
            "subject": subject,
            "prompts": len(templates),
            "generated_images": len(templates) * args.expected_images_per_prompt,
            "selection": args.selection,
            **{metric: float(np.mean(concept_scores[metric])) for metric in METRIC_FIELDS},
            "preparation_seconds_per_object": timing.get("preparation_seconds"),
            "inference_seconds_per_image": timing.get("inference_seconds_per_image"),
            "end_to_end_seconds_per_image": timing.get("end_to_end_seconds_per_image"),
        })
        print(f"[scored] {subject}: {len(templates)} prompts x {args.expected_images_per_prompt} images", flush=True)

    overall = {
        "model": args.model_name,
        "selection": args.selection,
        "subjects": len(subject_rows),
        "prompts_per_subject": len(subject_rows) and subject_rows[0]["prompts"],
        "inference_images": sum(row["generated_images"] for row in subject_rows),
        **{metric: float(np.mean([row[metric] for row in subject_rows])) for metric in METRIC_FIELDS},
        "preparation_seconds_per_object": finite_mean([row["preparation_seconds_per_object"] for row in subject_rows]),
        "inference_seconds_per_image": finite_mean([row["inference_seconds_per_image"] for row in subject_rows]),
        "end_to_end_seconds_per_image": finite_mean([row["end_to_end_seconds_per_image"] for row in subject_rows]),
        "text_protocol": args.text_protocol,
        "dino_model": "facebookresearch/dino:main:dino_vits16",
        "clip_model": "openai/CLIP:ViT-B/32",
    }
    report_dir.mkdir(parents=True, exist_ok=True)
    write_csv(report_dir / "per_prompt_metrics.csv", prompt_rows, list(prompt_rows[0]))
    write_csv(report_dir / "per_subject_metrics.csv", subject_rows, list(subject_rows[0]))
    write_csv(report_dir / "summary.csv", [overall], list(overall))
    (report_dir / "summary.json").write_text(json.dumps(overall, indent=2) + "\n")
    print("[overall] " + " ".join(f"{metric}={overall[metric]:.5f}" for metric in METRIC_FIELDS), flush=True)
    print(f"[overall] time={overall['inference_seconds_per_image']} s/image", flush=True)
    print(f"[saved] {report_dir}", flush=True)


if __name__ == "__main__":
    main()
