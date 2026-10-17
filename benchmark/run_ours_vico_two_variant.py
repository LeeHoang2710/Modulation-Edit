#!/usr/bin/env python3
"""Run the current Modulation-Edit method on the 16 x 31 ViCo benchmark.

This runner intentionally has two separate phases:

1. Fit and save one null-text scale-0..4 offset for every selected
   ``(subject, offset_seed)`` pair.
2. Reuse those exact saved offsets to generate every requested prompt for both
   late-scale ablations: conditional G_prompt and frozen G_null.

Each offset seed produces one image per prompt.  Thus ``--offset-seeds N``
creates N independently fitted offsets and N images per prompt, rather than
sampling N images from a single offset.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


EARLY_SCALES = "0-4"


def run(command: list[str], root: Path) -> None:
    print("[exec]", " ".join(command), flush=True)
    subprocess.run(command, cwd=root, check=True)


def load_manifest(path: Path) -> list[dict]:
    return json.loads(path.read_text())["subjects"]


def selected_subjects(items: list[dict], requested: str) -> list[dict]:
    wanted = None if requested == "all" else {x.strip() for x in requested.split(",") if x.strip()}
    result = [item for item in items if wanted is None or item["subject"] in wanted]
    unknown = set() if wanted is None else wanted - {item["subject"] for item in result}
    if unknown:
        raise SystemExit(f"Unknown ViCo subject(s): {', '.join(sorted(unknown))}")
    if not result:
        raise SystemExit("No ViCo subjects selected")
    return result


def common_model_args(args: argparse.Namespace) -> list[str]:
    return [
        "--pn", args.pn, "--model_type", "infinity_2b", "--model_path", str(args.model_path),
        "--vae_type", "32", "--vae_path", str(args.vae_path),
        "--add_lvl_embeding_only_first_block", "1", "--use_bit_label", "1",
        "--rope2d_each_sa_layer", "1", "--rope2d_normalized_by_hw", "2",
        "--use_scale_schedule_embedding", "0", "--apply_spatial_patchify", "0",
        "--checkpoint_type", "torch", "--bf16", "1", "--text_channels", "2048",
    ]


def fit_offsets(args: argparse.Namespace, root: Path, subjects: list[dict], work_root: Path) -> dict[str, list[float]]:
    """Fit every X*Y offset before any prompt-conditioned generation begins."""
    preparation: dict[str, list[float]] = {item["subject"]: [] for item in subjects}
    for item in subjects:
        subject = item["subject"]
        references = (root / item["reference_dir"]).resolve()
        for seed in range(args.offset_seeds):
            work = work_root / "offsets" / subject / f"train_seed{seed:03d}"
            offset_path = work / "offset.pth"
            reconstruction = work / "reconstruction.jpg"
            if offset_path.is_file() and not args.force_optimize:
                print(f"[offset exists] {subject} train_seed={seed}: {offset_path}", flush=True)
                preparation[subject].append(0.0)
                continue
            work.mkdir(parents=True, exist_ok=True)
            started = time.perf_counter()
            run([
                sys.executable, "tools/optimize_ada_offset.py", "--mode", "optimize", "--images", str(references),
                *common_model_args(args), "--seed", str(seed), "--hflip", "0", "--grad_checkpointing", "1",
                "--offset_mode", args.offset_mode, "--iters", str(args.steps),
                "--batch_size", str(args.batch_size), "--grad_accum", str(args.grad_accum), "--lr", str(args.lr),
                "--loss_scales", EARLY_SCALES, "--offset_infer_scales", EARLY_SCALES,
                "--num_samples", "1", "--sample_every", "0", "--save_offset", str(offset_path),
                "--save_file", str(reconstruction),
            ], root)
            elapsed = time.perf_counter() - started
            preparation[subject].append(elapsed)
            print(f"[offset saved] {subject} train_seed={seed} elapsed={elapsed:.1f}s", flush=True)
    return preparation


def generate_variant(
    args: argparse.Namespace,
    root: Path,
    subjects: list[dict],
    work_root: Path,
    output_root: Path,
    late_gss: str,
    preparation: dict[str, list[float]],
) -> None:
    method = f"{args.method_prefix}_late_{late_gss}"
    print(f"========== [variant] {method} ==========", flush=True)
    for item in subjects:
        subject = item["subject"]
        destination = output_root / method / subject
        timing_dir = work_root / "timings" / method / subject
        timing_dir.mkdir(parents=True, exist_ok=True)
        pure_generation = 0.0
        images = 0
        for seed in range(args.offset_seeds):
            offset_path = work_root / "offsets" / subject / f"train_seed{seed:03d}" / "offset.pth"
            if not offset_path.is_file():
                raise RuntimeError(f"Missing fitted offset: {offset_path}")
            timing_file = timing_dir / f"train_seed{seed:03d}.json"
            run([
                sys.executable, "tools/infer_offset_vico_scale_schedule.py",
                "--offset-path", str(offset_path), "--manifest", str(args.manifest), "--subject", subject,
                "--out-dir", str(output_root / method), *common_model_args(args),
                "--text_encoder_ckpt", str(args.text_encoder_ckpt), "--alpha-schedule", args.alpha_schedule,
                "--early-cfg", str(args.early_cfg), "--late-cfg", str(args.late_cfg), "--late-gss", late_gss,
                "--num-samples", "1", "--seed", str(args.inference_seed_base + seed),
                "--output-seed-start", str(seed), "--max-prompts", str(args.max_prompts),
                "--timing-file", str(timing_file),
            ], root)
            timing = json.loads(timing_file.read_text())
            pure_generation += float(timing["generation_seconds"])
            images += int(timing["images"])

        prompt_count = min(args.max_prompts, len(item["prompt_templates"])) if args.max_prompts else len(item["prompt_templates"])
        expected = prompt_count * args.offset_seeds
        actual = sum(1 for p in destination.glob("prompt_*/seed_*.png") if p.is_file())
        if actual != expected:
            raise RuntimeError(f"{method}/{subject}: expected {expected} images, found {actual}")
        prep_times = preparation[subject]
        timing = {
            "preparation_seconds": sum(prep_times),
            "preparation_seconds_per_offset": sum(prep_times) / max(len(prep_times), 1),
            "inference_seconds": pure_generation,
            "inference_seconds_per_image": pure_generation / max(images, 1),
            "offset_seeds": args.offset_seeds,
            "images": actual,
            "reference_protocol": "all ViCo subject references; one shared null-text offset for scales 0-4",
            "method_configuration": {
                "early_scales": EARLY_SCALES,
                "alpha_schedule": args.alpha_schedule,
                "early_cfg": args.early_cfg,
                "late_cfg": args.late_cfg,
                "ca_kv": "prompt at all scales",
                "late_modulation": f"G_{late_gss}",
            },
        }
        destination.mkdir(parents=True, exist_ok=True)
        (destination / "timing.json").write_text(json.dumps(timing, indent=2) + "\n")
        print(f"[done] {method}/{subject}: images={actual} inference={timing['inference_seconds_per_image']:.3f}s/image", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="benchmark/manifests/vico.json")
    parser.add_argument("--subjects", default="all", help="all or comma-separated ViCo subject names")
    parser.add_argument("--offset-seeds", type=int, default=1, help="number of independently fitted offsets per subject")
    parser.add_argument("--inference-seed-base", type=int, default=0)
    parser.add_argument("--variants", default="prompt,null", help="comma-separated late modulation variants: prompt,null")
    parser.add_argument("--max-prompts", type=int, default=0, help="0 means all 31; 5 is a fast smoke test")
    parser.add_argument("--output", default="benchmark_outputs_vico")
    parser.add_argument("--work-dir", default="benchmark_work/vico_modulation")
    parser.add_argument("--method-prefix", default="modulation_offset")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument("--offset-mode", choices=["per_scale", "shared"], default="per_scale")
    parser.add_argument("--alpha-schedule", default="0.25,0.20,0.15,0.10,0.05")
    parser.add_argument("--early-cfg", type=float, default=1.2)
    parser.add_argument("--late-cfg", type=float, default=2.0)
    parser.add_argument("--pn", default="1M")
    parser.add_argument("--model-path", default="weights/infinity/infinity_2b_reg.pth")
    parser.add_argument("--vae-path", default="weights/infinity/infinity_vae_d32reg.pth")
    parser.add_argument("--text-encoder-ckpt", default="weights/infinity/flan-t5-xl")
    parser.add_argument("--force-optimize", action="store_true")
    args = parser.parse_args()
    if args.offset_seeds < 1:
        parser.error("--offset-seeds must be at least 1")
    if args.max_prompts < 0:
        parser.error("--max-prompts must be non-negative")
    variants = [x.strip() for x in args.variants.split(",") if x.strip()]
    if not variants or set(variants) - {"prompt", "null"}:
        parser.error("--variants must contain prompt and/or null")

    root = Path(__file__).resolve().parents[1]
    args.manifest = (root / args.manifest).resolve()
    if not args.manifest.exists():
        run([sys.executable, "benchmark/prepare_vico.py", "--output", str(args.manifest)], root)
    for attr in ("model_path", "vae_path", "text_encoder_ckpt"):
        value = (root / getattr(args, attr)).resolve()
        if not value.exists():
            raise SystemExit(f"Missing {attr.replace('_', ' ')}: {value}")
        setattr(args, attr, value)
    subjects = selected_subjects(load_manifest(args.manifest), args.subjects)
    output_root = (root / args.output).resolve()
    work_root = (root / args.work_dir).resolve()

    print(f"[phase 1/2] fitting {len(subjects) * args.offset_seeds} offsets (subjects={len(subjects)}, offset_seeds={args.offset_seeds})", flush=True)
    preparation = fit_offsets(args, root, subjects, work_root)
    print("[phase 2/2] prompt-conditioned ViCo inference", flush=True)
    for late_gss in variants:
        generate_variant(args, root, subjects, work_root, output_root, late_gss, preparation)


if __name__ == "__main__":
    main()
