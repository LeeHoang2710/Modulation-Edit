#!/usr/bin/env python3
"""Learn one style offset per image and compare scale-placement variants.

For each ``styles/*.png`` image this runner:

    1. optimizes a per-scale AdaLN offset using only training scales 0--5;
2. injects the learned style offset using four scale-placement variants:
   no injection, scale 0, scale 1, and scales 0+1;
3. generates ``a dog`` with the normal prompt conditioning (prompt CA and
   prompt shared AdaLN), plus one selected offset.

The other generation scales receive no learned style offset.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np


def run(command: list[str], root: Path) -> None:
    print("[exec]", " ".join(command), flush=True)
    subprocess.run(command, cwd=root, check=True)


def save_comparison(records: list[dict], output: Path) -> None:
    """Save one reference and the four requested scale-placement variants per style."""
    rows = []
    for record in records:
        source = cv2.imread(record["style_path"])
        generated = [cv2.imread(path) for path in record["outputs"].values()]
        if source is None or any(image is None for image in generated):
            continue
        size = 256
        images = [source, *generated]
        tiles = []
        labels = ["style reference", "no style", "offset scale 0", "offset scale 1", "offset scales 0+1"]
        for label, image in zip(labels, images):
            resized = cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)
            header = np.full((40, size, 3), 255, dtype=np.uint8)
            cv2.putText(header, label, (8, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
            tiles.append(np.vstack([header, resized]))
        rows.append(np.hstack(tiles))
    if rows:
        cv2.imwrite(str(output), np.vstack(rows))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--styles-dir", default="styles")
    parser.add_argument("--prompt", default="a dog")
    parser.add_argument("--output-dir", default="outputs/style_endpoint_dog")
    parser.add_argument("--work-dir", default="benchmark_work/style_endpoint_dog")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--lr", type=float, default=4e-3)
    parser.add_argument("--noise-apply-layers", type=int, default=13)
    parser.add_argument("--noise-apply-strength", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--pn", default="1M")
    parser.add_argument("--model-path", default="weights/infinity/infinity_2b_reg.pth")
    parser.add_argument("--vae-path", default="weights/infinity/infinity_vae_d32reg.pth")
    parser.add_argument("--text-encoder-ckpt", default="weights/infinity/flan-t5-xl")
    parser.add_argument("--force-retrain", action="store_true")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    styles_dir = (root / args.styles_dir).resolve()
    style_paths = sorted(styles_dir.glob("*.png"))
    if not style_paths:
        raise SystemExit(f"No PNG style images found in {styles_dir}")
    for required in (args.model_path, args.vae_path, args.text_encoder_ckpt):
        if not (root / required).exists():
            raise SystemExit(f"Missing required asset: {(root / required).resolve()}")

    work_root = (root / args.work_dir).resolve()
    output_root = (root / args.output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    records = []

    common = [
        "--pn", args.pn, "--model_type", "infinity_2b", "--model_path", str((root / args.model_path).resolve()),
        "--vae_type", "32", "--vae_path", str((root / args.vae_path).resolve()),
        "--text_encoder_ckpt", str((root / args.text_encoder_ckpt).resolve()),
        "--add_lvl_embeding_only_first_block", "1", "--use_bit_label", "1",
        "--rope2d_each_sa_layer", "1", "--rope2d_normalized_by_hw", "2",
        "--use_scale_schedule_embedding", "0", "--apply_spatial_patchify", "0",
        "--checkpoint_type", "torch", "--bf16", "1", "--text_channels", "2048",
    ]

    for style_path in style_paths:
        name = style_path.stem
        offset_path = work_root / name / "offset.pth"
        variants = {
            "none": "",
            "scale0": "0",
            "scale1": "1",
            "scale0_1": "0-1",
        }
        result_paths = {
            variant: output_root / f"style_{name}_{variant}.png"
            for variant in variants
        }
        started = time.monotonic()

        if args.force_retrain or not offset_path.exists():
            run([
                sys.executable, "tools/optimize_ada_offset.py", "--mode", "optimize",
                "--images", str(style_path), *common, "--seed", str(args.seed), "--hflip", "0",
                "--grad_checkpointing", "1", "--offset_mode", "per_scale", "--iters", str(args.steps),
                "--batch_size", str(args.batch_size), "--grad_accum", str(args.grad_accum), "--lr", str(args.lr),
                "--noise_apply_layers", str(args.noise_apply_layers),
                "--noise_apply_strength", str(args.noise_apply_strength),
                "--loss_scales", "0-5", "--offset_infer_scales", "0-5", "--num_samples", "1",
                "--sample_every", "0", "--save_offset", str(offset_path),
                "--save_file", str(work_root / name / "reconstruction.jpg"),
            ], root)
        preparation_seconds = time.monotonic() - started

        for variant, result_path in result_paths.items():
            run([
                sys.executable, "tools/infer_style_endpoint_offset.py",
                "--offset-path", str(offset_path), "--prompt", args.prompt, "--output", str(result_path),
                "--offset-scales", variants[variant],
                "--num-samples", str(args.num_samples), "--seed", str(args.seed), *common,
            ], root)
        records.append({
            "style": name, "style_path": str(style_path), "offset_path": str(offset_path),
            "outputs": {variant: str(path) for variant, path in result_paths.items()},
            "preparation_seconds": preparation_seconds,
            "trained_scales": "0-5", "variants": {name: scales for name, scales in variants.items()},
            "prompt": args.prompt,
        })
        print(f"[done] style={name} outputs={list(result_paths.values())}", flush=True)

    (output_root / "run.json").write_text(json.dumps({
        "prompt": args.prompt, "styles": records, "trained_scales": [0, 1, 2, 3, 4, 5],
        "variants": {"none": [], "scale0": [0], "scale1": [1], "scale0_1": [0, 1]},
        "num_styles": len(records),
    }, indent=2) + "\n")
    save_comparison(records, output_root / "style_scale_placement_comparison.png")
    print(f"[complete] generated {len(records)} style-conditioned image(s) in {output_root}")


if __name__ == "__main__":
    main()
