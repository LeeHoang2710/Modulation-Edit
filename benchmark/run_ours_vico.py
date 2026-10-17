#!/usr/bin/env python3
"""Run Modulation-Edit on the official ARBooth/ViCo input protocol.

Each subject receives one null-text, per-scale AdaLN offset fitted from all of
its ViCo references. The selected prompt-injection variant then produces every
official prompt and normalizes its samples into the shared benchmark layout.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

def run(command: list[str], root: Path) -> None:
    print("[exec]", " ".join(command), flush=True)
    subprocess.run(command, cwd=root, check=True)


def load_manifest(path: Path) -> list[dict]:
    return json.loads(path.read_text())["subjects"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="benchmark/manifests/vico.json")
    parser.add_argument("--subjects", default="all")
    parser.add_argument("--output", default="benchmark_outputs_vico")
    parser.add_argument("--work-dir", default="benchmark_work/vico_modulation")
    parser.add_argument("--method-name", default="modulation_offset")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument("--loss-scales", default="0-4")
    parser.add_argument("--offset-infer-scales", default="0-4")
    parser.add_argument("--offset-mode", choices=["per_scale", "shared"], default="per_scale")
    parser.add_argument("--images-per-prompt", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--alpha-schedule", default="0.25,0.20,0.15,0.10,0.05",
                        help="early-scale alpha values; five values means scales 0-4")
    parser.add_argument("--early-cfg", type=float, default=1.2)
    parser.add_argument("--late-cfg", type=float, default=2.0)
    parser.add_argument("--pn", default="1M")
    parser.add_argument("--model-path", default="weights/infinity/infinity_2b_reg.pth")
    parser.add_argument("--vae-path", default="weights/infinity/infinity_vae_d32reg.pth")
    parser.add_argument("--text-encoder-ckpt", default="weights/infinity/flan-t5-xl")
    parser.add_argument("--force-optimize", action="store_true")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    manifest = (root / args.manifest).resolve()
    if not manifest.exists():
        run([sys.executable, "benchmark/prepare_vico.py", "--output", str(manifest)], root)
    requested = None if args.subjects == "all" else {part.strip() for part in args.subjects.split(",") if part.strip()}
    subjects = [item for item in load_manifest(manifest) if requested is None or item["subject"] in requested]
    if not subjects:
        raise SystemExit("No ViCo subjects selected")

    for key in ("model_path", "vae_path", "text_encoder_ckpt"):
        path = (root / getattr(args, key)).resolve()
        if not path.exists():
            raise SystemExit(f"Missing {key.replace('_', ' ')}: {path}")
        setattr(args, key, path)
    output_root = (root / args.output).resolve()
    work_root = (root / args.work_dir).resolve()

    for item in subjects:
        subject = item["subject"]
        references = (root / item["reference_dir"]).resolve()
        work = work_root / subject
        offset_path = work / "offset.pth"
        reconstruction = work / "reconstruction.jpg"
        if args.force_optimize or not offset_path.is_file():
            started = time.monotonic()
            run([sys.executable, "tools/optimize_ada_offset.py", "--mode", "optimize", "--images", str(references),
                 "--pn", args.pn, "--model_type", "infinity_2b", "--model_path", str(args.model_path),
                 "--vae_type", "32", "--vae_path", str(args.vae_path), "--add_lvl_embeding_only_first_block", "1",
                 "--use_bit_label", "1", "--rope2d_each_sa_layer", "1", "--rope2d_normalized_by_hw", "2",
                 "--use_scale_schedule_embedding", "0", "--apply_spatial_patchify", "0", "--checkpoint_type", "torch",
                 "--bf16", "1", "--text_channels", "2048", "--seed", str(args.seed), "--hflip", "0",
                 "--grad_checkpointing", "1", "--offset_mode", args.offset_mode, "--iters", str(args.steps),
                 "--batch_size", str(args.batch_size), "--grad_accum", str(args.grad_accum), "--lr", str(args.lr),
                 "--loss_scales", args.loss_scales, "--offset_infer_scales", args.offset_infer_scales,
                 "--num_samples", "1", "--sample_every", "0", "--save_offset", str(offset_path),
                 "--save_file", str(reconstruction)], root)
            preparation = time.monotonic() - started
        else:
            preparation = 0.0

        destination = output_root / args.method_name / subject
        started = time.monotonic()
        run([sys.executable, "tools/infer_offset_vico_scale_schedule.py", "--offset-path", str(offset_path), "--subject", subject,
             "--manifest", str(manifest), "--out-dir", str(output_root / args.method_name), "--pn", args.pn,
             "--model_type", "infinity_2b", "--model_path", str(args.model_path), "--vae_type", "32",
             "--vae_path", str(args.vae_path), "--add_lvl_embeding_only_first_block", "1", "--use_bit_label", "1",
             "--rope2d_each_sa_layer", "1", "--rope2d_normalized_by_hw", "2", "--use_scale_schedule_embedding", "0",
             "--apply_spatial_patchify", "0", "--checkpoint_type", "torch", "--bf16", "1", "--text_channels", "2048",
             "--text_encoder_ckpt", str(args.text_encoder_ckpt), "--alpha-schedule", args.alpha_schedule,
             "--early-cfg", str(args.early_cfg), "--late-cfg", str(args.late_cfg),
             "--num-samples", str(args.images_per_prompt), "--seed", str(args.seed)], root)
        inference = time.monotonic() - started
        images = sum(1 for path in destination.glob("prompt_*/seed_*.png") if path.is_file())
        expected_images = len(item["prompt_templates"]) * args.images_per_prompt
        if images != expected_images:
            raise RuntimeError(f"{subject}: expected {expected_images} normalized images, wrote {images}")
        (destination / "timing.json").write_text(json.dumps({
            "preparation_seconds": preparation,
            "inference_seconds": inference,
            "inference_seconds_per_image": inference / max(images, 1),
            "reference_protocol": "all ViCo reference images; null-text per-scale modulation offset",
            "method_configuration": {"alpha_schedule": args.alpha_schedule, "early_cfg": args.early_cfg,
                                     "late_cfg": args.late_cfg, "ca_kv": "prompt at all scales",
                                     "late_modulation": "G_prompt", "loss_scales": args.loss_scales},
        }, indent=2) + "\n")
        print(f"[done] {subject}: {images} images -> {destination}")


if __name__ == "__main__":
    main()
