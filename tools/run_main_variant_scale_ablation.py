#!/usr/bin/env python3
"""Train the friend main variant once, then compare offset scale placement.

Subjects are trained with subject text on scales 0--3. Inference uses
G_subject + 0.9 D_s, prompt cross-attention, CFG 1.5, and the requested
offset-scale variants.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

from tools.infer_offset_modspace import NULL, PROMPT, encode_texts, modspace_infer
from tools.optimize_ada_offset import get_scale_schedule, parse_scales
from tools.run_infinity import add_common_arguments, encode_prompt, load_tokenizer, load_transformer, load_visual_tokenizer


SUBJECT_PROMPTS = {"black_cat": "a grey cat", "fat_dog": "a fat dog"}
VARIANTS = (
    ("inject_0", "0"),
    ("inject_1", "1"),
    ("inject_2", "2"),
    ("inject_3", "3"),
    ("inject_0_1", "0-1"),
    ("inject_0_1_2", "0-2"),
    ("inject_0_1_2_3", "0-3"),
    ("inject_0_2_3", "0,2-3"),
    ("inject_1_2_3", "1-3"),
    ("inject_2_3", "2-3"),
)


def run(command: list[str], root: Path, log: Path) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8") as handle:
        handle.write("$ " + " ".join(command) + "\n")
        handle.flush()
        subprocess.run(command, cwd=root, stdout=handle, stderr=subprocess.STDOUT, check=True)


def tile(image: np.ndarray, label: str, size: int = 256) -> np.ndarray:
    image = cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)
    header = np.full((42, size, 3), 255, dtype=np.uint8)
    cv2.putText(header, label, (7, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 0, 0), 1, cv2.LINE_AA)
    return np.vstack([header, image])


def make_grid(subject: str, references: Path, output_dir: Path, images: dict[str, Path]) -> None:
    ref_paths = sorted(references.glob("*.jpg")) + sorted(references.glob("*.png"))
    ref = cv2.imread(str(ref_paths[0]))
    if ref is None:
        raise OSError(f"Could not read reference image in {references}")
    tiles = [tile(ref, f"{subject}: reference")]
    for name, _ in VARIANTS:
        image = cv2.imread(str(images[name]))
        if image is None:
            raise FileNotFoundError(images[name])
        tiles.append(tile(image, name))
    columns = 6
    blank = np.full_like(tiles[0], 245)
    rows = []
    for start in range(0, len(tiles), columns):
        row = tiles[start:start + columns]
        row += [blank] * (columns - len(row))
        rows.append(np.hstack(row))
    output_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_dir / f"{subject}_main_variant_scale_ablation.png"), np.vstack(rows))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser)
    parser.add_argument("--subjects", default="black_cat,fat_dog")
    parser.add_argument("--reference-root", default="ARBooth/inputs")
    parser.add_argument("--work-root", default="benchmark_work/main_variant_scale_ablation")
    parser.add_argument("--output-root", default="outputs/main_variant_scale_ablation")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument("--bsc-layers", type=int, default=13)
    parser.add_argument("--bsc-strength", type=float, default=0.3)
    parser.add_argument("--beta", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    parser.set_defaults(cfg="1.5", h_div_w_template=-1.)
    args = parser.parse_args()
    args.cfg = float(args.cfg)

    root = Path(__file__).resolve().parents[1]
    subjects = [x.strip() for x in args.subjects.split(",") if x.strip()]
    unknown = set(subjects) - set(SUBJECT_PROMPTS)
    if unknown:
        parser.error(f"Unknown subjects: {', '.join(sorted(unknown))}")
    model_path = (root / "weights/infinity/infinity_2b_reg.pth").resolve()
    vae_path = (root / "weights/infinity/infinity_vae_d32reg.pth").resolve()
    text_path = (root / "weights/infinity/flan-t5-xl").resolve()
    for path in (model_path, vae_path, text_path):
        if not path.exists():
            parser.error(f"Missing model asset: {path}")

    work_root = (root / args.work_root).resolve()
    output_root = (root / args.output_root).resolve()
    common = [
        "--pn", "1M", "--model_type", "infinity_2b", "--model_path", str(model_path),
        "--vae_type", "32", "--vae_path", str(vae_path),
        "--text_encoder_ckpt", str(text_path), "--add_lvl_embeding_only_first_block", "1",
        "--use_bit_label", "1", "--rope2d_each_sa_layer", "1", "--rope2d_normalized_by_hw", "2",
        "--use_scale_schedule_embedding", "0", "--apply_spatial_patchify", "0",
        "--checkpoint_type", "torch", "--bf16", "1", "--text_channels", "2048",
    ]

    for subject in subjects:
        refs = (root / args.reference_root / subject).resolve()
        subject_work = work_root / subject
        offset_path = subject_work / "offset.pth"
        if args.force or not offset_path.is_file():
            command = [
                sys.executable, "tools/optimize_ada_offset.py", "--mode", "optimize",
                "--images", str(refs), *common, "--seed", str(args.seed), "--hflip", "0",
                "--grad_checkpointing", "1", "--offset_mode", "per_scale", "--iters", str(args.steps),
                "--batch_size", str(args.batch_size), "--grad_accum", str(args.grad_accum), "--lr", str(args.lr),
                "--loss_scales", "0-3", "--offset_infer_scales", "0-3",
                "--noise_apply_layers", str(args.bsc_layers), "--noise_apply_strength", str(args.bsc_strength),
                "--train_prompt", SUBJECT_PROMPTS[subject], "--num_samples", "1", "--sample_every", "0",
                "--save_offset", str(offset_path), "--save_file", str(subject_work / "training_reconstruction.jpg"),
            ]
            print(f"[optimize] {subject}: prompt={SUBJECT_PROMPTS[subject]!r}, scales=0-3", flush=True)
            run(command, root, subject_work / "optimize.log")

        vae = load_visual_tokenizer(args)
        vae.requires_grad_(False)
        infinity = load_transformer(vae, args)
        infinity.requires_grad_(False)
        infinity.eval()
        tokenizer, encoder = load_tokenizer(t5_path=str(text_path))
        subject_cond = encode_prompt(tokenizer, encoder, SUBJECT_PROMPTS[subject])
        # Keep prompt CA on the conditional row and null CA on the CFG partner,
        # while replacing only the base SOS/GSS/head condition with G_subject.
        prompt_texts = encode_texts(infinity, 32, subject_cond)
        base_texts = encode_texts(infinity, 32, subject_cond)
        sos, gss, ca = prompt_texts
        base_sos, base_gss, _ = base_texts
        sos, gss = sos.clone(), gss.clone()
        sos[NULL], gss[NULL] = base_sos[PROMPT], base_gss[PROMPT]
        texts = sos, gss, ca
        checkpoint = torch.load(offset_path, map_location="cpu", weights_only=False)
        offset = checkpoint["ada_offset"].reshape(-1, 1, 6, infinity.C).cuda().float() * args.beta
        _, schedule, _ = get_scale_schedule(args, checkpoint.get("h_div_w_template", 1.0))
        infer_args = argparse.Namespace(top_k=900, top_p=0.97)
        output_subject = output_root / subject
        output_subject.mkdir(parents=True, exist_ok=True)
        images = {}
        for name, scale_spec in VARIANTS:
            scales = set(parse_scales(scale_spec, len(schedule)))
            image = modspace_infer(
                infinity, vae, texts, offset,
                [(NULL, PROMPT, True, 0.0), (NULL, NULL, True, 0.0)],
                schedule, scales, set(), args.cfg, 1.0, args.seed, infer_args,
            )
            path = output_subject / f"{name}.png"
            cv2.imwrite(str(path), image)
            images[name] = path
            print(f"[saved] {path} scales={scale_spec}", flush=True)
        make_grid(subject, refs, output_root, images)
        del encoder, tokenizer, infinity, vae
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
