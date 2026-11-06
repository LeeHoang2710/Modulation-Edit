#!/usr/bin/env python3
"""Run DreamMatcher on released Custom Diffusion ViCo checkpoints."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torchvision.io import read_image
from torchvision.utils import save_image

CONCEPT_INPUT_NAMES = {
    "alarm_clock": "clock",
    "black_cat": "black_cat",
    "brown_cat": "brown_cat",
    "brown_dog": "brown_dog",
    "brown_dog2": "brown_dog2",
    "brown_teddybear": "teddybear",
    "cat_statue": "cat-toy",
    "duck_toy": "duck_toy",
    "elephant_statue": "elephant_statue",
    "fat_dog": "fat_dog",
    "monster_toy": "monster_toy",
    "pink_sunglasses": "pink_sunglasses",
    "red_teapot": "red_teapot",
    "red_vase": "vase",
    "tortoise_plushy": "tortoise_plushy",
    "wooden_barn": "barn",
}


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


def load_image(path: Path, device: torch.device) -> torch.Tensor:
    image = read_image(str(path))[:3].unsqueeze(0).float() / 127.5 - 1.0
    return F.interpolate(image, (512, 512)).to(device)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="benchmark/manifests/vico.json")
    parser.add_argument("--subjects", default="all")
    parser.add_argument("--max-prompts", type=int, default=0)
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--weights", default="weights/custom_diffusion")
    parser.add_argument("--base-model", default="CompVis/stable-diffusion-v1-4")
    parser.add_argument("--output", default="benchmark_outputs_vico_dreammatcher_custom_diffusion_5seeds")
    parser.add_argument("--guidance-scale", type=float, default=10.0)
    parser.add_argument("--steps", type=int, default=50)
    args = parser.parse_args()
    if args.max_prompts < 0 or args.steps < 1:
        parser.error("invalid prompt count or inference step count")
    try:
        seeds = parse_seeds(args.seeds)
    except ValueError as error:
        parser.error(str(error))
    if not torch.cuda.is_available():
        raise SystemExit("DreamMatcher requires CUDA")

    root = Path(__file__).resolve().parents[1]
    dream_root = root / "models/DreamMatcher"
    if not (dream_root / "run_dreammatcher.py").is_file():
        raise SystemExit("DreamMatcher submodule is missing; initialize repository submodules")
    sys.path.insert(0, str(dream_root / "diffusers/src"))
    sys.path.insert(0, str(dream_root))
    from diffusers import DDIMScheduler
    from dreammatcher.diffuser_utils import CustomDiffusionPipeline
    from dreammatcher.dreammatcher import MutualSelfAttentionControlMaskAuto_Matching
    from dreammatcher.dreammatcher_utils import AttentionBase, regiter_attention_editor_diffusers
    from pytorch_lightning import seed_everything

    manifest = json.loads((root / args.manifest).read_text())
    subjects = selected_subjects(manifest, args.subjects)
    source_map = json.loads((dream_root / "inputs/imgs_source.txt").read_text())
    weights_root = (root / args.weights).resolve()
    output_root = (root / args.output).resolve()
    device = torch.device("cuda")

    for item in subjects:
        subject = item["subject"]
        input_name = CONCEPT_INPUT_NAMES[subject]
        source_path = dream_root / "inputs" / input_name / source_map[input_name]
        delta = weights_root / subject / "delta.bin"
        if not source_path.is_file() or not delta.is_file():
            raise FileNotFoundError(f"Missing source image or delta for {subject}: {source_path}, {delta}")
        subject_output = output_root / subject
        if subject_output.exists():
            raise RuntimeError(f"Output already exists for {subject}: {subject_output}")

        load_started = time.monotonic()
        scheduler = DDIMScheduler.from_pretrained(args.base_model, subfolder="scheduler")
        model = CustomDiffusionPipeline.from_pretrained(
            args.base_model,
            scheduler=scheduler,
            safety_checker=None,
        ).to(device)
        model.load_model(str(delta), False)
        source_image = load_image(source_path, device)
        load_seconds = time.monotonic() - load_started
        inference_seconds: list[float] = []

        for prompt_index, template in enumerate(item["prompt_templates"][: args.max_prompts or None]):
            ref_prompt = f"a photo of a <new1> {subject}"
            gen_prompt = template.format(f"<new1> {subject}")
            ref_token_idx = 6
            cur_token_idx = gen_prompt.split(" ").index(subject) + 1
            params = {
                "seed": 0,
                "dtype": None,
                "ref_prompt": ref_prompt,
                "ref_token_idx": ref_token_idx,
                "gen_prompt": gen_prompt,
                "cur_token_idx": cur_token_idx,
                "guidance_scale": args.guidance_scale,
                "inference_steps": args.steps,
                "initial_step": 4,
                "initial_layer": 10,
                "cut_step": args.steps,
                "cut_layer": 16,
                "thres": 0.05,
                "cc_thres": 0.4,
                "use_inversion_latents": True,
                "key_replace": False,
                "fg_mask": True,
                "matching_guidance": False,
                "mg_initial_step": 4,
                "mg_cut_step": args.steps,
                "mg_cc_thres": 0.4,
                "mg_grad_weight": 50,
                "save_pred_x0": False,
                "save_mask": False,
            }

            regiter_attention_editor_diffusers(model, AttentionBase())
            seed_everything(0)
            start_code, latents_list = model.invert(
                source_image,
                "",
                guidance_scale=args.guidance_scale,
                num_inference_steps=args.steps,
                return_intermediates=True,
            )
            prompt_dir = subject_output / f"prompt_{prompt_index:02d}"
            prompt_dir.mkdir(parents=True, exist_ok=False)
            for seed in seeds:
                params["seed"] = seed
                seed_everything(seed)
                end_code = torch.randn((1, 4, 64, 64), device=device)
                editor = MutualSelfAttentionControlMaskAuto_Matching(
                    params, mask_save_dir=None, save_dir=str(prompt_dir)
                )
                regiter_attention_editor_diffusers(model, editor)
                initial_code = torch.cat([start_code, end_code], dim=0)
                started = time.monotonic()
                model_output = model(
                    [ref_prompt, gen_prompt],
                    latents=initial_code,
                    num_inference_steps=args.steps,
                    guidance_scale=args.guidance_scale,
                    ref_intermediate_latents=latents_list,
                    matching_guidance=None,
                    outdir=str(prompt_dir),
                    save_pred_x0=False,
                )
                inference_seconds.append(time.monotonic() - started)
                save_image(model_output[1], prompt_dir / f"seed_{seed:03d}.png")

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
            "source_image": str(source_path),
            "delta_checkpoint": str(delta),
            "base_model": args.base_model,
            "steps": args.steps,
            "guidance_scale": args.guidance_scale,
            "dreammatcher_parameters": {
                "initial_step": 4, "initial_layer": 10, "cut_step": args.steps,
                "cut_layer": 16, "thres": 0.05, "cc_thres": 0.4,
                "use_inversion_latents": True, "fg_mask": True,
            },
        }
        (subject_output / "timing.json").write_text(json.dumps(timing, indent=2) + "\n")
        print(f"[complete] {subject}: {len(inference_seconds)} images", flush=True)
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
