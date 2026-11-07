#!/usr/bin/env python3
"""Generate the ViCo benchmark with released Custom Diffusion delta weights."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from torchvision.utils import save_image

from dreammatcher_vico_common import CONCEPT_INPUT_NAMES, load_image


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
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--guidance-scale", type=float, default=10.0)
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
    dream_root = root / "models/DreamMatcher"
    if not (dream_root / "dreammatcher/diffuser_utils.py").is_file():
        raise SystemExit("DreamMatcher submodule is missing; initialize repository submodules")
    # DreamMatcher's UNet returns (noise, intermediate_features). Its matching
    # CustomDiffusionPipeline consumes that exact interface; the stock Diffusers
    # StableDiffusionPipeline does not.
    sys.path.insert(0, str(dream_root / "diffusers/src"))
    sys.path.insert(0, str(dream_root))
    from diffusers import DDIMScheduler
    from dreammatcher.diffuser_utils import CustomDiffusionPipeline
    from dreammatcher.dreammatcher_utils import AttentionBase, regiter_attention_editor_diffusers
    from pytorch_lightning import seed_everything

    manifest = json.loads((root / args.manifest).read_text())
    subjects = selected_subjects(manifest, args.subjects)
    weights_root = (root / args.weights).resolve()
    output_root = (root / args.output).resolve()
    source_map = json.loads((dream_root / "inputs/imgs_source.txt").read_text())
    device = torch.device("cuda")

    for item in subjects:
        subject = item["subject"]
        input_name = CONCEPT_INPUT_NAMES[subject]
        source_path = dream_root / "inputs" / input_name / source_map[input_name]
        delta = weights_root / subject / "delta.bin"
        if not source_path.is_file() or not delta.is_file():
            raise FileNotFoundError(f"Missing source image or Custom Diffusion checkpoint: {source_path}, {delta}")
        subject_output = output_root / subject
        if subject_output.exists():
            raise RuntimeError(f"Output already exists for {subject}: {subject_output}")

        load_started = time.monotonic()
        scheduler = DDIMScheduler.from_pretrained(args.base_model, subfolder="scheduler")
        pipe = CustomDiffusionPipeline.from_pretrained(
            args.base_model,
            scheduler=scheduler,
            safety_checker=None,
        ).to("cuda")
        pipe.load_model(str(delta), False)
        source_image = load_image(source_path, device)
        # This is DreamMatcher's released Custom Diffusion baseline protocol:
        # invert the designated reference, but use a no-op attention editor.
        seed_everything(0)
        start_code, _ = pipe.invert(
            source_image,
            "",
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.steps,
            return_intermediates=True,
        )
        load_seconds = time.monotonic() - load_started

        prompts = item["prompt_templates"][: args.max_prompts or None]
        inference_seconds: list[float] = []
        for prompt_index, template in enumerate(prompts):
            ref_prompt = f"a photo of a <new1> {subject}"
            prompt = rendered_prompt(template, subject)
            prompt_dir = subject_output / f"prompt_{prompt_index:02d}"
            prompt_dir.mkdir(parents=True, exist_ok=False)
            for seed in seeds:
                seed_everything(seed)
                end_code = torch.randn((1, 4, 64, 64), device=device)
                regiter_attention_editor_diffusers(pipe, AttentionBase())
                initial_code = torch.cat([start_code, end_code], dim=0)
                started = time.monotonic()
                images = pipe(
                    [ref_prompt, prompt],
                    latents=initial_code,
                    num_inference_steps=args.steps,
                    guidance_scale=args.guidance_scale,
                    outdir=str(prompt_dir),
                    save_pred_x0=False,
                )
                inference_seconds.append(time.monotonic() - started)
                save_image(images[1], prompt_dir / f"seed_{seed:03d}.png")

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
            "source_image": str(source_path),
            "steps": args.steps,
            "guidance_scale": args.guidance_scale,
            "pipeline": "DreamMatcher CustomDiffusion baseline (matching disabled)",
            "scheduler": "DDIMScheduler",
        }
        (subject_output / "timing.json").write_text(json.dumps(timing, indent=2) + "\n")
        print(f"[complete] {subject}: {len(prompts) * len(seeds)} images", flush=True)
        del pipe
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
