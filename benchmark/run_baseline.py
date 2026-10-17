#!/usr/bin/env python3
"""Native-command adapters for the five requested personalization baselines.

The adapters deliberately call the checked-out official code rather than
reimplement the methods.  Their output normaliser creates the one image layout
consumed by `benchmark/evaluate_dreambooth.py`.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

from prepare_dreambooth import main as _unused  # validates package-local import when run as a script


def load_manifest(path: Path) -> list[dict]:
    return json.loads(path.read_text())["subjects"]


def shell(command: list[str], cwd: Path, env: dict[str, str]) -> None:
    print("[exec]", " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, env=env, check=True)


def clean_prompt(prompt: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in prompt.lower()).strip("_")[:120]


def copy_images(source: Path, destination: Path, count: int) -> int:
    # Original DreamBooth/Textual-Inversion save true samples in `samples/` and
    # optional grids beside it. Always prefer the sample directory when present.
    search_root = source / "samples" if (source / "samples").is_dir() else source
    candidates = sorted(p for p in search_root.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"})
    if len(candidates) < count:
        raise RuntimeError(f"{source}: expected {count} generated images, found {len(candidates)}")
    destination.mkdir(parents=True, exist_ok=True)
    for index, image in enumerate(candidates[:count]):
        shutil.copy2(image, destination / f"seed_{index:03d}.png")
    return len(candidates[:count])


def image_count(directory: Path) -> int:
    return sum(1 for path in directory.glob("*") if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"})


def ensure_dreambooth_priors(model: Path, cls: str, args, env: dict[str, str]) -> Path:
    """Create class-prior images required by the original DreamBooth code."""
    prior_dir = args.regularization_root / clean_prompt(cls)
    existing = image_count(prior_dir) if prior_dir.is_dir() else 0
    if existing >= args.prior_images:
        return prior_dir
    if not args.sd14_ckpt.is_file():
        raise RuntimeError(
            f"DreamBooth requires {args.sd14_ckpt} to generate class-prior images; "
            "download/point --sd14-ckpt at an accepted Stable Diffusion v1.4 checkpoint."
        )
    raw = prior_dir / "_raw"
    missing = args.prior_images - existing
    # The upstream sampler writes to outdir/samples. Copy its individual images
    # into the flat directory expected by DreamBooth's PersonalizedBase dataset.
    shell(["python", "scripts/stable_txt2img.py", "--skip_grid", "--n_samples", str(min(4, missing)),
           "--n_iter", str((missing + 3) // 4), "--seed", str(args.seed), "--ddim_steps", "50",
           "--scale", "7.5", "--ckpt", str(args.sd14_ckpt), "--outdir", str(raw),
           "--prompt", f"a photo of a {cls}"], model, env)
    produced = sorted((raw / "samples").glob("*"))
    prior_dir.mkdir(parents=True, exist_ok=True)
    for index, path in enumerate(produced[:missing], start=existing):
        if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}:
            shutil.copy2(path, prior_dir / f"prior_{index:04d}{path.suffix.lower()}")
    if image_count(prior_dir) < args.prior_images:
        raise RuntimeError(f"DreamBooth prior generation produced too few images in {prior_dir}")
    return prior_dir


def write_timing(subject_dir: Path, prep: float, infer: float, image_count: int, protocol: str) -> None:
    subject_dir.mkdir(parents=True, exist_ok=True)
    (subject_dir / "timing.json").write_text(json.dumps({
        "preparation_seconds": prep,
        "inference_seconds": infer,
        "inference_seconds_per_image": infer / max(image_count, 1),
        "reference_protocol": protocol,
    }, indent=2) + "\n")


def run_arbooth(root: Path, item: dict, output: Path, args, env: dict[str, str]) -> None:
    model = root / "models" / "ARBooth"
    subject, cls = item["subject"], item["class_name"]
    work = args.work_dir / "arbooth" / subject
    if not args.infinity_model.is_file() or not args.infinity_vae.is_file():
        raise RuntimeError("ARBooth requires the Infinity-2B and VAE checkpoints; run the download script first.")
    if not args.flan_t5.is_dir():
        raise RuntimeError(f"ARBooth requires local FLAN-T5-XL at {args.flan_t5}")
    train_start = time.monotonic()
    shell(["python", "booth_train_final.py", "--instance_prompt", f"<placeholder> {cls}", "--class_prompt", cls,
           "--data_path", str(Path(item["reference_images"][0]).parent), "--local_out_path", str(work),
           "--exp_name", subject, "--val_prompt_path", str(args.validation_prompts),
           "--model_path", str(args.infinity_model), "--vae_path", str(args.infinity_vae),
           "--text_encoder_ckpt", str(args.flan_t5), "--train_iter", str(args.arbooth_steps)], model, env)
    prep = time.monotonic() - train_start
    checkpoints = sorted(work.glob("txt_emb*.pth"))
    if not checkpoints:
        raise RuntimeError(f"ARBooth did not write a text embedding in {work}")
    infer_start, written = time.monotonic(), 0
    subject_out = output / "arbooth" / subject
    for prompt in item["prompts"]:
        raw = work / "raw" / clean_prompt(prompt)
        raw.mkdir(parents=True, exist_ok=True)
        shell(["python", "tools/run_arbooth.py", "--prompt", prompt.replace(cls, f"<placeholder> {cls}"),
               "--model_path", str(checkpoints[-1]), "--save_file", str(raw / "sample.jpg"), "--seed", str(args.seed),
               "--num_samples", str(args.images_per_prompt), "--result_dir", str(raw), "--pn", args.pn,
               "--vae_path", str(args.infinity_vae), "--text_encoder_ckpt", str(args.flan_t5)], model, env)
        target = subject_out / clean_prompt(prompt)
        (target / "prompt.txt").parent.mkdir(parents=True, exist_ok=True)
        (target / "prompt.txt").write_text(prompt + "\n")
        written += copy_images(raw, target, args.images_per_prompt)
    write_timing(subject_out, prep, time.monotonic() - infer_start, written, "all DreamBooth references")


def run_csd_var(root: Path, item: dict, output: Path, args, env: dict[str, str]) -> None:
    model = root / "models" / "CSD-VAR"
    subject, cls = item["subject"], item["class_name"]
    # CSD-VAR Switti is a single-reference method.  It sees the manifest's
    # deterministic first reference; the protocol is surfaced in every report.
    work = args.work_dir / "csd_var" / subject
    data_dir = work / f"{cls}+photo"; data_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(item["single_reference"], data_dir / Path(item["single_reference"]).name)
    prompts_path = work / "prompts.json"; prompts_path.parent.mkdir(parents=True, exist_ok=True)
    # `inference.py` substitutes `{}` with the learned <object> virtual token.
    # The stored prompt.txt below remains the original class prompt so CLIP-T is
    # scored on the same textual intent as the other four baselines.
    prompts_path.write_text(json.dumps({
        "content": {"recontextualization": [p.replace(cls, "{}") for p in item["prompts"]],
                    "stylization": [], "non-rigid": []},
        "style": [],
    }))
    prep_start = time.monotonic()
    shell(["python", "extract_svd.py", "--input_dir_data", str(data_dir)], model, env)
    local = work / "local_output"
    shell(["torchrun", "--standalone", "--nproc_per_node=1", "train.py", "--pretrained_path", str(args.switti),
           "--data_path", str(data_dir), "--vae_ckpt", str(args.switti_vae), "--max_iters", str(args.csd_steps), "--bs=1",
           "--save_iters", str(args.csd_step), "--global_save_iters", str(args.csd_step),
           "--fp16=2", "--placeholder_token_content=<object>", f"--initializer_token_content={cls}",
           "--placeholder_token_style=<style>", "--initializer_token_style=photo", "--use_captions=True",
           "--num_vectors_content=4", "--num_vectors_style=4", "--major_scale_end=3", "--major_type_scale=style",
           "--local_out_dir_path", str(local), "--prompt_config_path=config/config_group_scales.yaml",
           "--proj_path", str(data_dir / "proj_top_10.pt"), "--proj_2_path", str(data_dir / "proj_2_top_10.pt")], model, env)
    prep = time.monotonic() - prep_start
    infer_start = time.monotonic()
    raw = work / "raw"
    shell(["python", "inference.py", "--prompts_path", str(prompts_path), "--step", str(args.csd_step),
           "--content_path_ckpt", str(local), "--style_path_ckpt", str(local), "--content_name", cls, "--style_name", "photo",
           "--output_dir", str(raw), "--images_per_prompt", str(args.images_per_prompt), "--model_path", str(args.switti),
           "--prompt_config_path=config/config_group_scales.yaml", "--proj_path", str(data_dir / "proj_top_10.pt"),
           "--proj_2_path", str(data_dir / "proj_2_top_10.pt")], model, env)
    result_dirs = sorted(raw.glob(f"{cls}+photo_*"))
    if not result_dirs:
        raise RuntimeError(f"CSD-VAR did not write an inference result under {raw}")
    result_dir = result_dirs[-1] / "output" / "ours"
    subject_out, written = output / "csd_var" / subject, 0
    for prompt_index, prompt in enumerate(item["prompts"]):
        target = subject_out / clean_prompt(prompt); target.mkdir(parents=True, exist_ok=True); (target / "prompt.txt").write_text(prompt + "\n")
        source_dirs = sorted(result_dir.glob(f"prompt_{prompt_index}_*"))
        if not source_dirs:
            raise RuntimeError(f"CSD-VAR missing prompt {prompt_index} under {result_dir}")
        written += copy_images(source_dirs[-1], target, args.images_per_prompt)
    write_timing(subject_out, prep, time.monotonic() - infer_start, written, "single deterministic first DreamBooth reference")


def run_unziplora(root: Path, item: dict, output: Path, args, env: dict[str, str]) -> None:
    model = root / "models" / "UnZipLoRA"
    subject, cls = item["subject"], item["class_name"]
    work = args.work_dir / "unziplora" / subject; checkpoint = work / "model"
    prep_start = time.monotonic()
    shell(["accelerate", "launch", "train_unziplora.py", "--pretrained_model_name_or_path", str(args.sdxl),
           "--instance_data_dir", str(Path(item["reference_images"][0]).parent), "--output_dir", str(checkpoint),
           "--instance_prompt", f"a monadikos {cls}", "--content_forward_prompt", f"a monadikos {cls}",
           "--style_forward_prompt", f"a {cls} in photo style", "--rank", "64", "--resolution", "1024",
           "--train_batch_size", "1", "--max_train_steps", str(args.unzip_steps), "--mixed_precision", "fp16",
           "--report_to", "none", "--seed", str(args.seed)], model, env)
    prep = time.monotonic() - prep_start
    infer_start, written, subject_out = time.monotonic(), 0, output / "unziplora" / subject
    for prompt in item["prompts"]:
        raw = work / "raw" / clean_prompt(prompt)
        shell(["python", "infer.py", "--with_unziplora", "--output_dir", str(checkpoint), "--rank", "64",
               "--num", str(args.images_per_prompt), "--save_dir", str(raw), "--validation_prompt", f"a monadikos {prompt}",
               "--validation_prompt_content_forward", f"a monadikos {prompt}",
               "--validation_prompt_style_forward", f"a {cls} in photo style",
               "--validation_prompt_content_recontext", "", "--validation_prompt_style", ""], model, env)
        target = subject_out / clean_prompt(prompt); target.mkdir(parents=True, exist_ok=True); (target / "prompt.txt").write_text(prompt + "\n")
        written += copy_images(raw, target, args.images_per_prompt)
    write_timing(subject_out, prep, time.monotonic() - infer_start, written, "all DreamBooth references")


def legacy_runner(root: Path, item: dict, output: Path, args, env: dict[str, str], name: str) -> None:
    """DreamBooth/TI official LDM command bridge.

    These two official 2022 codebases require an SD-v1.4 `.ckpt` and a Python
    3.8 environment. The launch arguments here match their READMEs exactly.
    """
    model = root / "models" / ("DreamBooth" if name == "dreambooth" else "Textual-Inversion")
    subject, cls = item["subject"], item["class_name"]
    work = args.work_dir / name / subject; work.mkdir(parents=True, exist_ok=True)
    prep_start = time.monotonic()
    if name == "dreambooth":
        prior_dir = ensure_dreambooth_priors(model, cls, args, env)
        shell(["python", "main.py", "--base", "configs/stable-diffusion/v1-finetune_unfrozen.yaml", "-t", "--gpus", "0,",
               "--actual_resume", str(args.sd14_ckpt), "-n", subject, "--data_root", str(Path(item["reference_images"][0]).parent),
               "--reg_data_root", str(prior_dir), "--class_word", cls, "--max_steps", str(args.dreambooth_steps)], model, env)
        checkpoints = sorted((model / "logs" / subject / "checkpoints").glob("*.ckpt"))
    else:
        shell(["python", "main.py", "--base", "configs/latent-diffusion/txt2img-1p4B-finetune.yaml", "-t", "--gpus", "0,",
               "--actual_resume", str(args.ldm_ckpt), "-n", subject, "--data_root", str(Path(item["reference_images"][0]).parent),
               "--init_word", cls, "--placeholder_string", "*"], model, env)
        checkpoints = sorted((model / "logs" / subject / "checkpoints").glob("embeddings*.pt"))
    prep = time.monotonic() - prep_start
    if not checkpoints:
        raise RuntimeError(f"{name}: no checkpoint found after training {subject}")
    infer_start, written, subject_out = time.monotonic(), 0, output / name / subject
    for prompt in item["prompts"]:
        raw = work / "raw" / clean_prompt(prompt)
        if name == "dreambooth":
            personalized_prompt = prompt.replace(cls, f"sks {cls}")
            command = ["python", "scripts/stable_txt2img.py", "--skip_grid", "--n_samples", str(args.images_per_prompt), "--n_iter", "1",
                       "--seed", str(args.seed), "--ckpt", str(checkpoints[-1]), "--outdir", str(raw), "--prompt", personalized_prompt]
        else:
            personalized_prompt = prompt.replace(cls, f"* {cls}")
            command = ["python", "scripts/txt2img.py", "--skip_grid", "--n_samples", str(args.images_per_prompt), "--n_iter", "1",
                       "--seed", str(args.seed), "--ckpt_path", str(args.ldm_ckpt), "--embedding_path", str(checkpoints[-1]),
                       "--outdir", str(raw), "--prompt", personalized_prompt]
        shell(command, model, env)
        target = subject_out / clean_prompt(prompt); target.mkdir(parents=True, exist_ok=True); (target / "prompt.txt").write_text(prompt + "\n")
        written += copy_images(raw, target, args.images_per_prompt)
    write_timing(subject_out, prep, time.monotonic() - infer_start, written, "all DreamBooth references")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["arbooth", "csd_var", "dreambooth", "textual_inversion", "unziplora"])
    parser.add_argument("--manifest", default="benchmark/manifests/dreambooth.json")
    parser.add_argument("--output", default="benchmark_outputs")
    parser.add_argument("--work-dir", default="benchmark_work")
    parser.add_argument("--subjects", default="all")
    parser.add_argument("--images-per-prompt", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--pn", default="1M")
    parser.add_argument("--infinity-model", default="weights/infinity/infinity_2b_reg.pth")
    parser.add_argument("--infinity-vae", default="weights/infinity/infinity_vae_d32reg.pth")
    parser.add_argument("--flan-t5", default="weights/infinity/flan-t5-xl")
    parser.add_argument("--validation-prompts", default="benchmark/prompts_validate.list")
    parser.add_argument("--switti", default="weights/switti/Switti-AR")
    parser.add_argument("--switti-vae", default="weights/switti/VQVAE-Switti")
    parser.add_argument("--csd-steps", type=int, default=200)
    parser.add_argument("--csd-step", type=int, default=150)
    parser.add_argument("--sdxl", default="weights/sdxl-base-1.0")
    parser.add_argument("--unzip-steps", type=int, default=600)
    parser.add_argument("--arbooth-steps", type=int, default=500)
    parser.add_argument("--dreambooth-steps", type=int, default=800)
    parser.add_argument("--prior-images", type=int, default=100)
    parser.add_argument("--sd14-ckpt", default="weights/stable-diffusion/sd-v1-4-full-ema.ckpt")
    parser.add_argument("--ldm-ckpt", default="weights/latent-diffusion/model.ckpt")
    parser.add_argument("--regularization-root", type=Path, default=Path("data/regularization"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    # All paths must stay valid when subprocesses change cwd into an upstream
    # model directory. Resolve once at the benchmark boundary.
    for name in ("work_dir", "infinity_model", "infinity_vae", "flan_t5", "validation_prompts", "switti", "switti_vae", "sdxl", "sd14_ckpt", "ldm_ckpt", "regularization_root"):
        value = Path(getattr(args, name))
        setattr(args, name, value if value.is_absolute() else (root / value).resolve())
    subjects = load_manifest(root / args.manifest)
    selected = subjects if args.subjects == "all" else [x for x in subjects if x["subject"] in set(args.subjects.split(","))]
    if not selected:
        raise SystemExit("No manifest subjects selected.")
    env = os.environ.copy(); env["PYTHONPATH"] = f"{root}:{env.get('PYTHONPATH', '')}"
    runners = {"arbooth": run_arbooth, "csd_var": run_csd_var, "unziplora": run_unziplora}
    for item in selected:
        print(f"[{args.model}] {item['subject']}", flush=True)
        if args.model in runners:
            runners[args.model](root, item, root / args.output, args, env)
        else:
            legacy_runner(root, item, root / args.output, args, env, args.model)


if __name__ == "__main__":
    main()
