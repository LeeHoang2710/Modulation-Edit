#!/usr/bin/env python3
"""Run the released CSD-VAR Switti pipeline on selected ViCo subjects.

This adapter uses one deterministic reference per subject (as in upstream's
single-image workflow), and maps ViCo templates to CSD-VAR's content token.
It keeps upstream training and inference unchanged and normalizes outputs for
the shared ARBooth-style ViCo evaluator.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


def selected_subjects(manifest: dict, selection: str) -> list[dict]:
    wanted = None if selection == "all" else {s.strip() for s in selection.split(",") if s.strip()}
    subjects = [item for item in manifest["subjects"] if wanted is None or item["subject"] in wanted]
    unknown = set() if wanted is None else wanted - {item["subject"] for item in subjects}
    if unknown or not subjects:
        raise ValueError(f"Invalid subject selection: {sorted(unknown) if unknown else selection}")
    return subjects


def prepare_subject(root: Path, item: dict, parent: Path, max_prompts: int) -> tuple[Path, Path]:
    """Create the upstream single-reference dataset layout and ViCo prompts."""
    cls = item["class_name"]
    data = parent / "data" / f"{cls}+photo"
    images = data / "imgs"
    images.mkdir(parents=True, exist_ok=True)
    ref = (root / item["single_reference"]).resolve()
    if not ref.is_file():
        raise FileNotFoundError(f"Missing reference: {ref}")
    target = images / ("reference" + ref.suffix.lower())
    existing = [p for p in images.iterdir() if p.is_file()]
    if existing and existing != [target]:
        raise RuntimeError(f"Unexpected existing files in {images}; choose a fresh --work-dir")
    if target.exists():
        if target.read_bytes() != ref.read_bytes():
            raise RuntimeError(f"Reference in {images} differs from {ref}; choose a fresh --work-dir")
    else:
        shutil.copy2(ref, target)

    captions = data / "captions.csv"
    with captions.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["", "file_name", "caption_full", "caption_object", "caption_style"])
        writer.writerow([0, target.name, "a photo of a {} in {} style", "a photo of a {}", "a {} style image"])
    # These are generic class-level descriptions, not subject-specific captions.
    # Their CLIP features define CSD-VAR's content projection space.
    variants = [template.format(cls) for template in (
        "a small {}", "a large {}", "a {} indoors", "a {} outdoors", "a front view of a {}",
        "a side view of a {}", "a close-up of a {}", "a red {}", "a white {}", "a photograph of a {}",
    )]
    (data / "variation_content.txt").write_text("\n".join(variants) + "\n")
    # Upstream substitutes {} with its learned token; preserve the ViCo class
    # noun as well, so e.g. "a {} in snow" becomes "a <object> dog in snow".
    templates = [template.replace("{}", "{} " + cls) for template in
                 item["prompt_templates"][:max_prompts or None]]
    prompts = parent / "vico_prompts.json"
    prompts.write_text(json.dumps({"content": {"recontextualization": templates, "stylization": [],
                                                "non-rigid": []}, "style": []}, indent=2) + "\n")
    return data, prompts


def run(command: list[str], cwd: Path) -> None:
    print("[exec]", " ".join(command), flush=True)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(cwd) + os.pathsep + env.get("PYTHONPATH", "")
    subprocess.run(command, cwd=cwd, env=env, check=True)


def checkpoint_files(directory: Path, step: int) -> list[Path]:
    names = [f"learned_embeds-steps-{step}_content.safetensors",
             f"learned_embeds_2-steps-{step}_content.safetensors",
             f"learned_embeds-steps-{step}_style.safetensors",
             f"learned_embeds_2-steps-{step}_style.safetensors",
             f"virtual_token_embed-steps-{step}.pt"]
    return [directory / name for name in names]


def normalize_outputs(raw: Path, target: Path, n_prompts: int, n_images: int) -> None:
    runs = sorted(path for path in raw.iterdir() if path.is_dir())
    if len(runs) != 1:
        raise RuntimeError(f"Expected exactly one upstream inference run in {raw}; found {len(runs)}")
    output = runs[0] / "output" / "ours"
    for index in range(n_prompts):
        source = output / f"prompt_{index}_content_recontextualization"
        files = [source / f"{i}.png" for i in range(n_images)]
        if not all(path.is_file() for path in files):
            raise RuntimeError(f"Missing generated samples in {source}")
        destination = target / f"prompt_{index:02d}"
        destination.mkdir(parents=True, exist_ok=True)
        old = list(destination.glob("seed_*.png"))
        if old:
            raise RuntimeError(f"Outputs already exist in {destination}; select a fresh --output directory")
        for i, file in enumerate(files):
            shutil.copy2(file, destination / f"seed_{i:03d}.png")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="benchmark/manifests/vico.json")
    parser.add_argument("--subjects", default="all")
    parser.add_argument("--max-prompts", type=int, default=0, help="0 means all ViCo prompts")
    parser.add_argument("--images-per-prompt", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42, help="upstream inference RNG seed")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--checkpoint-step", type=int, default=150)
    parser.add_argument("--save-every", type=int, default=50)
    parser.add_argument("--switti", default="yresearch/Switti-AR")
    parser.add_argument("--vae", default="yresearch/VQVAE-Switti", help="must match the VAE hard-coded by upstream inference")
    parser.add_argument("--output", default="benchmark_outputs_vico_csd_var")
    parser.add_argument("--work-dir", default="benchmark_work/vico_csd_var")
    parser.add_argument("--prepare-only", action="store_true", help="Write inputs only; do not run GPU models")
    args = parser.parse_args()
    if args.max_prompts < 0 or args.images_per_prompt < 1 or args.steps < 2 or args.save_every < 1:
        parser.error("Invalid prompt count, image count, steps or save interval")
    if args.checkpoint_step <= 0 or args.checkpoint_step >= args.steps or args.checkpoint_step % args.save_every:
        parser.error("checkpoint-step must be a positive save-every multiple smaller than steps")
    if args.vae != "yresearch/VQVAE-Switti":
        parser.error("upstream inference hard-codes yresearch/VQVAE-Switti; a different training VAE would mismatch")
    root = Path(__file__).resolve().parents[1]
    model = root / "models/CSD-VAR"
    if not (model / "train.py").is_file():
        raise SystemExit("CSD-VAR submodule missing: git submodule update --init --recursive models/CSD-VAR")
    manifest = json.loads((root / args.manifest).read_text())
    items = selected_subjects(manifest, args.subjects)
    work_root = (root / args.work_dir).resolve()
    out_root = (root / args.output).resolve()
    for item in items:
        subject, cls = item["subject"], item["class_name"]
        work = work_root / subject
        raw = work / "raw"
        if not args.prepare_only and (raw.exists() or (out_root / subject).exists()):
            raise RuntimeError(f"Outputs already exist for {subject}; choose fresh --work-dir and --output")
        data, prompts = prepare_subject(root, item, work, args.max_prompts)
        print(f"[subject] {subject}; single reference={item['single_reference']}", flush=True)
        if args.prepare_only:
            continue
        n_prompts = len(item["prompt_templates"][:args.max_prompts or None])
        started = time.monotonic()
        run([sys.executable, "extract_svd.py", "--input_dir_data", str(data.parent)], model)
        proj, proj2 = data / "proj_top_10.pt", data / "proj_2_top_10.pt"
        if not proj.is_file() or not proj2.is_file():
            raise RuntimeError(f"CSD-VAR projection files missing in {data}")
        checkpoint = work / "checkpoint"
        command = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=1", "train.py",
                   f"--pretrained_path={args.switti}", f"--vae_ckpt={args.vae}", f"--data_path={data}",
                   "--use_fsdp=True", "--grad_accum=1", "--dataset_repeats=100", "--pn=512", "--depth=30",
                   "--rope_size=128", "--rope_theta=10000", "--use_ar=True", "--use_crop_cond=True",
                   "--use_swiglu_ffn=True", f"--max_iters={args.steps}", "--bs=1", "--eval_batch_size=16",
                   "--log_iters=50", "--log_images_iters=50", f"--save_iters={args.save_every}",
                   f"--global_save_iters={args.save_every}", "--fp16=2", "--alng=1e-3", "--tblr=1e-2",
                   "--vfast=1", "--wp=50", "--twd=0.05", "--placeholder_token_content=<object>",
                   f"--initializer_token_content={cls}", "--placeholder_token_style=<style>",
                   "--initializer_token_style=photo", "--use_captions=True", "--num_vectors_content=4",
                   "--num_vectors_style=4", "--major_scale_end=3", "--major_type_scale=style",
                   "--enable_orthogonal_loss=False", f"--local_out_dir_path={checkpoint}",
                   "--minor_attend_weight=0.1", "--prompt_config_path=config/config_group_scales.yaml",
                   f"--proj_path={proj}", f"--proj_2_path={proj2}"]
        run(command, model)
        missing = [str(path) for path in checkpoint_files(checkpoint, args.checkpoint_step) if not path.is_file()]
        if missing:
            raise RuntimeError(f"CSD-VAR training did not produce required checkpoint files: {missing}")
        prep_seconds = time.monotonic() - started
        infer_start = time.monotonic()
        run([sys.executable, "inference.py", "--prompts_path", str(prompts), "--step", str(args.checkpoint_step),
             "--content_path_ckpt", str(checkpoint), "--style_path_ckpt", str(checkpoint),
             "--content_name", cls, "--style_name", "photo", "--output_dir", str(raw),
             "--images_per_prompt", str(args.images_per_prompt), "--seed", str(args.seed),
             "--model_path", args.switti, "--prompt_config_path", "config/config_group_scales.yaml",
             "--proj_path", str(proj), "--proj_2_path", str(proj2)], model)
        infer_seconds = time.monotonic() - infer_start
        dest = out_root / subject
        normalize_outputs(raw, dest, n_prompts, args.images_per_prompt)
        timing = {"preparation_seconds": prep_seconds, "inference_seconds": infer_seconds,
                  "inference_seconds_per_image": None,
                  "end_to_end_seconds_per_image": infer_seconds / (n_prompts * args.images_per_prompt),
                  "timing_note": "Inference subprocess wall time includes model loading and file I/O; not pure model inference.",
                  "reference_protocol": "one deterministic first ViCo reference", "reference": item["single_reference"],
                  "sampling": "upstream batched sampling from a single seeded RNG stream", "seed": args.seed,
                  "steps": args.steps, "checkpoint_step": args.checkpoint_step}
        (dest / "timing.json").write_text(json.dumps(timing, indent=2) + "\n")
        print(f"[complete] {subject}: {n_prompts * args.images_per_prompt} images", flush=True)


if __name__ == "__main__":
    main()
