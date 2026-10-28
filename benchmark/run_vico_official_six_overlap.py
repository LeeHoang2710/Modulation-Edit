#!/usr/bin/env python3
"""Reproduce official ViCo inference for the six exact-overlap subjects.

The released ViCo implementation is invoked unchanged.  We only translate
the repository's 31 benchmark templates to ViCo's ``*`` placeholder, choose
the first matching reference image, and normalize its output into a stable
per-prompt/per-seed layout for comparison with Modulation-Edit.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

from PIL import Image


OVERLAPS = {
    "barn": {"current_subject": "wooden_barn", "class_name": "barn"},
    "clock": {"current_subject": "alarm_clock", "class_name": "clock"},
    "dog7": {"current_subject": "brown_dog2", "class_name": "dog"},
    "monster_toy": {"current_subject": "monster_toy", "class_name": "toy"},
    "pink_sunglasses": {"current_subject": "pink_sunglasses", "class_name": "sunglasses"},
    "teddybear": {"current_subject": "brown_teddybear", "class_name": "teddybear"},
}


def parse_subjects(value: str) -> list[str]:
    subjects = list(OVERLAPS) if value == "all" else [part.strip() for part in value.split(",") if part.strip()]
    unknown = sorted(set(subjects) - set(OVERLAPS))
    if unknown:
        raise SystemExit(f"Unknown official ViCo subject(s): {', '.join(unknown)}")
    return subjects


def find_checkpoint(root: Path, official_subject: str, load_step: int) -> Path:
    token = official_subject.replace("_", "").lower()
    candidates = []
    for directory in [path for path in root.rglob("*") if path.is_dir()]:
        cross = directory / "checkpoints" / f"cross_attention-{load_step}.pt"
        embedding = directory / "checkpoints" / f"embeddings_gs-{load_step}.pt"
        if cross.is_file() and embedding.is_file():
            name = directory.name.replace("_", "").replace("-", "").lower()
            score = 0 if token in name else 1
            candidates.append((score, str(directory), directory))
    if not candidates:
        raise SystemExit(
            f"No ViCo checkpoint with step {load_step} found for {official_subject} under {root}. "
            "Run scripts/prepare_vico_official_overlap_assets.sh first."
        )
    candidates.sort(key=lambda item: (item[0], item[1]))
    if candidates[0][0] != 0 and len(candidates) > 1:
        raise SystemExit(f"Could not uniquely match ViCo checkpoint to {official_subject}: {[str(x[2]) for x in candidates]}")
    return candidates[0][2]


def image_files(directory: Path) -> list[Path]:
    return sorted(path for path in directory.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png"})


def run_subject(args: argparse.Namespace, manifest: dict, official_subject: str, root: Path) -> None:
    spec = OVERLAPS[official_subject]
    item = next(entry for entry in manifest["subjects"] if entry["subject"] == spec["current_subject"])
    references = root / item["reference_dir"]
    reference = sorted(image_files(references))[0]
    checkpoint = find_checkpoint((root / args.checkpoint_root).resolve(), official_subject, args.load_step)
    output_subject = (root / args.output) / official_subject
    output_subject.mkdir(parents=True, exist_ok=True)
    shutil.copy2(reference, output_subject / "reference.jpg")

    templates = item["prompt_templates"][:args.max_prompts or None]
    for prompt_index, template in enumerate(templates):
        rendered = template.format(spec["class_name"])
        prompt = rendered.replace(spec["class_name"], "*", 1)
        destination = output_subject / f"prompt_{prompt_index:02d}"
        existing = list(destination.glob("seed_*.jpg")) if destination.is_dir() else []
        if len(existing) == args.num_samples and not args.force:
            continue
        destination.mkdir(parents=True, exist_ok=True)
        temporary = destination / "_official_vico_tmp"
        if temporary.exists():
            shutil.rmtree(temporary)
        temporary.mkdir()
        command = [
            args.python, "scripts/vico_txt2img.py", "--ddim_eta", "0.0",
            "--n_samples", str(args.num_samples), "--n_iter", "1", "--scale", str(args.guidance_scale),
            "--ddim_steps", str(args.ddim_steps), "--ckpt_path", str((root / args.base_checkpoint).resolve()),
            "--image_path", str(reference.resolve()), "--ft_path", str(checkpoint),
            "--load_step", str(args.load_step), "--prompt", prompt, "--outdir", str(temporary),
            "--seed", str(args.seed),
        ]
        started = time.perf_counter()
        subprocess.run(command, cwd=root / args.code_root, check=True)
        elapsed = time.perf_counter() - started
        generated = sorted(path for path in temporary.rglob("*.jpg") if "_official_vico_tmp" not in path.name)
        # The official script writes individual 0000.jpg... files in a prompt
        # subfolder. Ignore its grid and normalize the individual images.
        generated = [path for path in generated if path.stem.isdigit()]
        if len(generated) != args.num_samples:
            raise RuntimeError(f"{official_subject}/prompt_{prompt_index:02d}: expected {args.num_samples}, found {len(generated)}")
        for seed_index, source in enumerate(generated):
            target = destination / f"seed_{seed_index:03d}.jpg"
            shutil.copy2(source, target)
        (destination / "prompt.txt").write_text(prompt + "\n")
        shutil.rmtree(temporary)
        print(f"[vico] {official_subject} prompt={prompt_index:02d} images={len(generated)} time={elapsed:.2f}s", flush=True)

    (output_subject / "metadata.json").write_text(json.dumps({
        "official_subject": official_subject,
        "current_subject": spec["current_subject"],
        "reference": str(reference),
        "checkpoint": str(checkpoint),
        "load_step": args.load_step,
        "seed": args.seed,
        "num_samples_per_prompt": args.num_samples,
        "ddim_steps": args.ddim_steps,
        "guidance_scale": args.guidance_scale,
        "protocol": "official ViCo inference; one matching reference image; * placeholder",
    }, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subjects", default="all", help="official names: barn,clock,dog7,monster_toy,pink_sunglasses,teddybear")
    parser.add_argument("--manifest", default="benchmark/manifests/vico.json")
    parser.add_argument("--code-root", default="models/ViCo")
    parser.add_argument("--checkpoint-root", default="weights/vico")
    parser.add_argument("--base-checkpoint", default="weights/vico/stable-diffusion-v1/sd-v1-4.ckpt")
    parser.add_argument("--output", default="benchmark_outputs_vico_official_six_overlap")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--load-step", type=int, default=399)
    parser.add_argument("--num-samples", type=int, default=4,
                        help="number of inference seeds per prompt; default 4")
    parser.add_argument("--ddim-steps", type=int, default=50)
    parser.add_argument("--guidance-scale", type=float, default=7.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-prompts", type=int, default=0, help="0 means all 31 prompts")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.num_samples < 1 or args.max_prompts < 0:
        parser.error("--num-samples must be positive and --max-prompts non-negative")
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root / args.manifest).read_text())
    code_root = (root / args.code_root).resolve()
    if not (code_root / "scripts/vico_txt2img.py").is_file():
        raise SystemExit("Official ViCo source is missing; run scripts/prepare_vico_official_overlap_assets.sh first.")
    for subject in parse_subjects(args.subjects):
        run_subject(args, manifest, subject, root)


if __name__ == "__main__":
    main()
