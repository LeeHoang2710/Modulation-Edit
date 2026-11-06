#!/usr/bin/env python3
"""Generate the ViCo benchmark with released Custom Diffusion delta weights."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch


def parse_seeds(value: str) -> list[int]:
    seeds = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not seeds or len(seeds) != len(set(seeds)) or any(seed < 0 for seed in seeds):
        raise ValueError("--seeds must contain unique non-negative integers")
    return seeds


def selected_subjects(manifest: dict, selection: str) -> list[dict]:
    wanted = None if selection == "all" else {part.strip() for part in selection.split(",") if part.strip()}
    selected = [item for item in manifest["subjects"] if wanted is None or item["subject"] in wanted]
    unknown = set() if wanted is None else wanted - {item["subject"] for item in selected}
    if unknown or not selected:
        raise ValueError(f"Invalid subject selection: {sorted(unknown) if unknown else selection}")
    return selected


def rendered_prompt(template: str, subject: str) -> str:
    """Match DreamMatcher's released Custom Diffusion prompt convention."""
    return template.format(f"<new1> {subject}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="benchmark/manifests/vico.json")
    parser.add_argument("--subjects", default="all")
    parser.add_argument("--max-prompts", type=int, default=0, help="0 means all 31 prompts")
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--base-model", default="CompVis/stable-diffusion-v1-4")
    parser.add_argument("--weights", default="weights/custom_diffusion")
    parser.add_argument("--output", default="benchmark_outputs_vico_custom_diffusion_5seeds")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--guidance-scale", type=float, default=6.0)
    parser.add_argument("--eta", type=float, default=1.0)
    args = parser.parse_args()
    if args.max_prompts < 0 or args.steps < 1:
        parser.error("--max-prompts must be non-negative and --steps must be positive")
    try:
        seeds = parse_seeds(args.seeds)
    except ValueError as error:
        parser.error(str(error))
    if not torch.cuda.is_available():
        raise SystemExit("Custom Diffusion inference requires CUDA")

    root = Path(__file__).resolve().parents[1]
    custom_root = root / "models/Custom-Diffusion"
    if not (custom_root / "src/diffusers_model_pipeline.py").is_file():
        raise SystemExit("Custom Diffusion submodule is missing; initialize repository submodules")
    sys.path.insert(0, str(custom_root))
    from src.diffusers_model_pipeline import CustomDiffusionPipeline

    manifest = json.loads((root / args.manifest).read_text())
    subjects = selected_subjects(manifest, args.subjects)
    weights_root = (root / args.weights).resolve()
    output_root = (root / args.output).resolve()

    for item in subjects:
        subject = item["subject"]
        delta = weights_root / subject / "delta.bin"
        if not delta.is_file():
            raise FileNotFoundError(f"Missing Custom Diffusion checkpoint: {delta}")
        subject_output = output_root / subject
        if subject_output.exists():
            raise RuntimeError(f"Output already exists for {subject}: {subject_output}")

        load_started = time.monotonic()
        pipe = CustomDiffusionPipeline.from_pretrained(
            args.base_model,
            torch_dtype=torch.float16,
            safety_checker=None,
        ).to("cuda")
        pipe.load_model(str(delta), False)
        load_seconds = time.monotonic() - load_started

        prompts = item["prompt_templates"][: args.max_prompts or None]
        inference_seconds: list[float] = []
        for prompt_index, template in enumerate(prompts):
            prompt = rendered_prompt(template, subject)
            prompt_dir = subject_output / f"prompt_{prompt_index:02d}"
            prompt_dir.mkdir(parents=True, exist_ok=False)
            for seed in seeds:
                generator = torch.Generator(device="cuda").manual_seed(seed)
                started = time.monotonic()
                image = pipe(
                    prompt,
                    num_inference_steps=args.steps,
                    guidance_scale=args.guidance_scale,
                    eta=args.eta,
                    generator=generator,
                ).images[0]
                inference_seconds.append(time.monotonic() - started)
                image.save(prompt_dir / f"seed_{seed:03d}.png")

        timing = {
            "model_load_seconds": load_seconds,
            "preparation_seconds": load_seconds,
            "preparation_seconds_by_offset": [load_seconds],
            "inference_seeds": seeds,
            "inference_seconds_per_image_by_seed": inference_seconds,
            "mean_inference_seconds_per_image": sum(inference_seconds) / len(inference_seconds),
            "inference_seconds_per_image": sum(inference_seconds) / len(inference_seconds),
            "inference_seconds_per_image_by_offset": inference_seconds,
            "end_to_end_seconds_per_image": (load_seconds + sum(inference_seconds)) / len(inference_seconds),
            "prompt_protocol": "DreamMatcher released convention: <new1> plus ViCo subject identifier",
            "base_model": args.base_model,
            "delta_checkpoint": str(delta),
            "steps": args.steps,
            "guidance_scale": args.guidance_scale,
            "eta": args.eta,
        }
        (subject_output / "timing.json").write_text(json.dumps(timing, indent=2) + "\n")
        print(f"[complete] {subject}: {len(prompts) * len(seeds)} images", flush=True)
        del pipe
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
