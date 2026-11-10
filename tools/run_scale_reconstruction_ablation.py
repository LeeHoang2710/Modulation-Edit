#!/usr/bin/env python3
"""Re-optimize subject offsets on progressively longer scale prefixes.

For each selected object and each endpoint in 3..12, this script fits a fresh
per-scale offset on scales 0..endpoint, conditioned on the object's subject
prompt, then saves a grid comparing the generated reconstruction across all
prefixes.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np


SUBJECTS = {
    "black_cat": "a black cat",
    "fat_dog": "a fat dog",
}


def run(command: list[str], root: Path, log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        log.write("$ " + " ".join(command) + "\n")
        log.flush()
        subprocess.run(command, cwd=root, stdout=log, stderr=subprocess.STDOUT, check=True)


def add_label(image: np.ndarray, text: str, width: int = 256) -> np.ndarray:
    image = cv2.resize(image, (width, width), interpolation=cv2.INTER_AREA)
    header = np.full((42, width, 3), 255, dtype=np.uint8)
    cv2.putText(header, text, (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (0, 0, 0), 1, cv2.LINE_AA)
    return np.vstack([header, image])


def make_grid(subject: str, endpoint_dirs: list[tuple[int, Path]], references: Path, output: Path) -> None:
    ref_paths = sorted(references.glob("*.jpg")) + sorted(references.glob("*.png"))
    if not ref_paths:
        raise FileNotFoundError(f"No reference images found in {references}")
    ref = cv2.imread(str(ref_paths[0]))
    if ref is None:
        raise OSError(f"Could not read reference image {ref_paths[0]}")

    tiles = [add_label(ref, f"reference: {subject}")]
    for endpoint, work in endpoint_dirs:
        generated = work / "reconstruction_r0_0.jpg"
        if not generated.is_file():
            generated = work / "reconstruction.jpg"
        image = cv2.imread(str(generated))
        if image is None:
            raise FileNotFoundError(f"Missing reconstruction for 0-{endpoint}: {generated}")
        tiles.append(add_label(image, f"optimized scales 0-{endpoint}"))

    # Keep the 13 columns readable by wrapping the variants over two rows.
    columns = 7
    blank = np.full_like(tiles[0], 245)
    rows = []
    for start in range(0, len(tiles), columns):
        row = tiles[start:start + columns]
        row += [blank] * (columns - len(row))
        rows.append(np.hstack(row))
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), np.vstack(rows))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subjects", default="black_cat,fat_dog")
    parser.add_argument("--reference-root", default="ARBooth/inputs")
    parser.add_argument("--work-root", default="benchmark_work/scale_reconstruction_ablation")
    parser.add_argument("--output-root", default="outputs/scale_reconstruction_ablation")
    parser.add_argument("--endpoints", default="3,4,5,6,7,8,9,10,11,12")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--pn", default="1M")
    parser.add_argument("--model-path", default="weights/infinity_2b_reg.pth")
    parser.add_argument("--vae-path", default="weights/infinity_vae_d32reg.pth")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    subjects = [name.strip() for name in args.subjects.split(",") if name.strip()]
    unknown = set(subjects) - set(SUBJECTS)
    if unknown:
        parser.error(f"Unknown subjects: {', '.join(sorted(unknown))}")
    endpoints = [int(value) for value in args.endpoints.split(",") if value.strip()]
    if any(endpoint < 3 or endpoint > 12 for endpoint in endpoints):
        parser.error("endpoints must be between 3 and 12")

    model_path = (root / args.model_path).resolve()
    vae_path = (root / args.vae_path).resolve()
    for path in (model_path, vae_path):
        if not path.is_file():
            parser.error(f"Missing model asset: {path}")

    work_root = (root / args.work_root).resolve()
    output_root = (root / args.output_root).resolve()
    common = [
        "--pn", args.pn, "--model_type", "infinity_2b", "--model_path", str(model_path),
        "--vae_type", "32", "--vae_path", str(vae_path),
        "--add_lvl_embeding_only_first_block", "1", "--use_bit_label", "1",
        "--rope2d_each_sa_layer", "1", "--rope2d_normalized_by_hw", "2",
        "--use_scale_schedule_embedding", "0", "--apply_spatial_patchify", "0",
        "--checkpoint_type", "torch", "--bf16", "1", "--text_channels", "2048",
    ]

    for subject in subjects:
        references = (root / args.reference_root / subject).resolve()
        subject_work = work_root / subject
        endpoint_dirs = []
        for endpoint in endpoints:
            work = subject_work / f"scale_0-{endpoint}"
            offset_path = work / "offset.pth"
            reconstruction = work / "reconstruction.jpg"
            endpoint_dirs.append((endpoint, work))
            if offset_path.is_file() and reconstruction.is_file() and not args.force:
                print(f"[skip] {subject} scales 0-{endpoint}: {work}", flush=True)
                continue
            work.mkdir(parents=True, exist_ok=True)
            command = [
                sys.executable, "tools/optimize_ada_offset.py", "--mode", "optimize",
                "--images", str(references), *common,
                "--seed", str(args.seed), "--hflip", "0", "--grad_checkpointing", "1",
                "--offset_mode", "per_scale", "--iters", str(args.steps),
                "--batch_size", str(args.batch_size), "--grad_accum", str(args.grad_accum),
                "--lr", str(args.lr), "--loss_scales", f"0-{endpoint}",
                "--offset_infer_scales", f"0-{endpoint}", "--train_prompt", SUBJECTS[subject],
                "--num_samples", "1", "--sample_every", "0",
                "--save_offset", str(offset_path), "--save_file", str(reconstruction),
            ]
            print(f"[optimize] {subject}: scales 0-{endpoint}", flush=True)
            run(command, root, work / "optimize.log")

        grid_path = output_root / f"{subject}_scale_reconstruction_grid.png"
        make_grid(subject, endpoint_dirs, references, grid_path)
        print(f"[grid] {grid_path}", flush=True)


if __name__ == "__main__":
    main()
