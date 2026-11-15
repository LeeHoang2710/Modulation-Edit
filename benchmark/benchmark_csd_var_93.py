#!/usr/bin/env python3
"""Prepare and time CSD-VAR for 3 ViCo subjects x 31 prompts x seed 0."""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path


def execute(command: list[str], cwd: Path, log: Path) -> float:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(cwd) + os.pathsep + env.get("PYTHONPATH", "")
    started = time.perf_counter()
    with log.open("a") as handle:
        handle.write("[exec] " + " ".join(command) + "\n")
        handle.flush()
        subprocess.run(command, cwd=cwd, env=env, stdout=handle, stderr=subprocess.STDOUT, check=True)
    return time.perf_counter() - started


def read_subjects(data_dir: Path, selection: list[str]) -> dict[str, dict]:
    with (data_dir / "subjects.csv").open(newline="") as handle:
        rows = {row["subject"]: row for row in csv.DictReader(handle)}
    missing = set(selection) - rows.keys()
    if missing:
        raise SystemExit(f"Unknown ViCo subjects: {sorted(missing)}")
    for subject in selection:
        row = rows[subject]
        row["templates"] = [
            line.strip() for line in (data_dir / row["prompts"]).read_text().splitlines() if line.strip()
        ]
        if len(row["templates"]) != 31:
            raise SystemExit(f"{subject}: expected 31 prompts, found {len(row['templates'])}")
    return rows


def prepare_inputs(data_dir: Path, item: dict, work: Path) -> tuple[Path, Path]:
    subject, cls = item["subject"], item["class_name"]
    refs = sorted(path for path in (data_dir / f"{subject}+photo" / "imgs").iterdir() if path.is_file())
    if not refs:
        raise FileNotFoundError(f"No references for {subject}")
    data = work / "data" / f"{cls}+photo"
    images = data / "imgs"
    images.mkdir(parents=True, exist_ok=True)
    target = images / ("reference" + refs[0].suffix.lower())
    if not target.exists():
        shutil.copy2(refs[0], target)
    with (data / "captions.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["", "file_name", "caption_full", "caption_object", "caption_style"])
        writer.writerow([0, target.name, "a photo of a {} in {} style", "a photo of a {}", "a {} style image"])
    variants = [template.format(cls) for template in (
        "a small {}", "a large {}", "a {} indoors", "a {} outdoors", "a front view of a {}",
        "a side view of a {}", "a close-up of a {}", "a red {}", "a white {}", "a photograph of a {}",
    )]
    (data / "variation_content.txt").write_text("\n".join(variants) + "\n")
    templates = [template.replace("{}", "{} " + cls) for template in item["templates"]]
    prompts = work / "vico_prompts.json"
    prompts.write_text(json.dumps({"content": {"recontextualization": templates, "stylization": [], "non-rigid": []}, "style": []}, indent=2) + "\n")
    return data, prompts


def checkpoint_files(directory: Path, step: int) -> list[Path]:
    return [directory / name for name in (
        f"learned_embeds-steps-{step}_content.safetensors",
        f"learned_embeds_2-steps-{step}_content.safetensors",
        f"learned_embeds-steps-{step}_style.safetensors",
        f"learned_embeds_2-steps-{step}_style.safetensors",
        f"virtual_token_embed-steps-{step}.pt",
    )]


def create_timed_inference(source: Path, destination: Path) -> None:
    """Create a version-pinned instrumentation copy; upstream stays untouched."""
    code = source.read_text()
    replacements = [
        (
            "import os\nfrom datetime import datetime",
            "import os\nimport time\nfrom datetime import datetime",
        ),
        (
            '    parser.add_argument(\n        "--save_grid",',
            '    parser.add_argument("--timing_file", type=str, default="")\n'
            '    parser.add_argument(\n        "--save_grid",',
        ),
        (
            "    # Generate and save images for each prompt\n    for prompt_idx,",
            "    # CUDA-synchronized generation-only timings. Model/checkpoint loading above and image writes below are excluded.\n"
            "    generation_durations = []\n"
            "    # Generate and save images for each prompt\n    for prompt_idx,",
        ),
        (
            "        # Generate all images at once\n        images = pipe(\n",
            "        # Prompt encoding happens inside pipe and is part of the shared timing contract.\n"
            "        if torch.cuda.is_available():\n            torch.cuda.synchronize()\n"
            "        generation_started = time.perf_counter()\n"
            "        # Generate all images at once\n        images = pipe(\n",
        ),
        (
            "        # Save each generated image\n        for i, image in enumerate(images):",
            "        if torch.cuda.is_available():\n            torch.cuda.synchronize()\n"
            "        generation_durations.append(time.perf_counter() - generation_started)\n\n"
            "        # Save each generated image\n        for i, image in enumerate(images):",
        ),
        (
            '    print(f"Results saved to {result_dir}")',
            '    if args.timing_file:\n'
            '        timing_dir = os.path.dirname(os.path.abspath(args.timing_file))\n'
            '        os.makedirs(timing_dir, exist_ok=True)\n'
            '        with open(args.timing_file, "w") as timing_handle:\n'
            '            json.dump({"images": len(generation_durations) * args.images_per_prompt, '
            '"batch_durations_seconds": generation_durations, "batch_size": args.images_per_prompt, '
            '"timing_contract": "prompt encoding + autoregressive sampling + VAE decoding; excludes model loading and image writes"}, '
            'timing_handle, indent=2)\n'
            '    print(f"Results saved to {result_dir}")',
        ),
    ]
    for old, new in replacements:
        if old not in code:
            raise RuntimeError(f"Unsupported CSD-VAR inference.py; instrumentation anchor missing: {old[:80]!r}")
        code = code.replace(old, new, 1)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(code)


def normalize(raw: Path, target: Path) -> None:
    runs = sorted(path for path in raw.iterdir() if path.is_dir())
    if len(runs) != 1:
        raise RuntimeError(f"Expected one CSD-VAR run in {raw}, found {len(runs)}")
    generated = runs[0] / "output" / "ours"
    for index in range(31):
        source = generated / f"prompt_{index}_content_recontextualization" / "0.png"
        if not source.is_file():
            raise FileNotFoundError(source)
        out = target / f"prompt_{index:02d}"
        out.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, out / "seed_000.png")


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser()
    p.add_argument("--csd-root", default="/workspace/CSD-VAR")
    p.add_argument("--data-dir", default="/workspace/CSD-VAR/benchmark_data/vico_csd")
    p.add_argument("--subjects", default="brown_dog,black_cat,alarm_clock")
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--checkpoint-step", type=int, default=150)
    p.add_argument("--save-every", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--switti", default="yresearch/Switti-AR")
    p.add_argument("--vae", default="yresearch/VQVAE-Switti")
    p.add_argument("--work-dir", default="output/timing/csd_var_3subjects_93/work")
    p.add_argument("--output", default="output/timing/csd_var_3subjects_93/images")
    args = p.parse_args()
    subjects = [part.strip() for part in args.subjects.split(",") if part.strip()]
    if len(subjects) != 3:
        p.error("--subjects must contain exactly three subjects")
    if args.checkpoint_step >= args.steps or args.checkpoint_step % args.save_every:
        p.error("checkpoint-step must be a save-every multiple smaller than steps")
    csd = Path(args.csd_root).resolve()
    data_dir = Path(args.data_dir).resolve()
    if not (csd / "train.py").is_file():
        raise SystemExit(f"Missing released CSD-VAR source: {csd}")
    rows = read_subjects(data_dir, subjects)
    work_root, out_root = Path(args.work_dir).resolve(), Path(args.output).resolve()
    timed_inference = work_root / "_instrumented_inference.py"
    create_timed_inference(csd / "inference.py", timed_inference)
    reports = []
    for subject in subjects:
        item = rows[subject]
        work, target = work_root / subject, out_root / subject
        stale = [path for path in (work / "checkpoint", work / "raw", target) if path.exists()]
        if stale:
            raise RuntimeError(
                f"Timing requires fresh outputs for {subject}; existing paths: {stale}. "
                "Set new WORK_DIR and OUTPUT values."
            )
        log = work / "benchmark.log"
        work.mkdir(parents=True, exist_ok=True)
        data, prompts = prepare_inputs(data_dir, item, work)
        svd_seconds = execute([sys.executable, "extract_svd.py", "--input_dir_data", str(data.parent)], csd, log)
        proj, proj2 = data / "proj_top_10.pt", data / "proj_2_top_10.pt"
        checkpoint = work / "checkpoint"
        cls = item["class_name"]
        train = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=1", "train.py",
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
        train_started_wall = time.time()
        train_seconds = execute(train, csd, log)
        used_checkpoint_files = checkpoint_files(checkpoint, args.checkpoint_step)
        missing = [str(path) for path in used_checkpoint_files if not path.is_file()]
        if missing:
            raise RuntimeError("Missing CSD-VAR checkpoints: " + ", ".join(missing))
        # max(mtime) is when the final component of the selected checkpoint was
        # written.  This measures only the training needed to make the exact
        # checkpoint used below, even when max_iters continues beyond it.
        checkpoint_ready_after_train_start = max(path.stat().st_mtime for path in used_checkpoint_files) - train_started_wall
        checkpoint_ready_after_train_start = max(0.0, min(train_seconds, checkpoint_ready_after_train_start))
        prep_to_used_checkpoint = svd_seconds + checkpoint_ready_after_train_start
        full_preparation_seconds = svd_seconds + train_seconds
        raw = work / "raw"
        generation_timing = work / "generation_timing.json"
        infer_process = execute([sys.executable, str(timed_inference), "--prompts_path", str(prompts),
                         "--step", str(args.checkpoint_step), "--content_path_ckpt", str(checkpoint),
                         "--style_path_ckpt", str(checkpoint), "--content_name", cls, "--style_name", "photo",
                         "--output_dir", str(raw), "--images_per_prompt", "1", "--seed", str(args.seed),
                         "--timing_file", str(generation_timing),
                         "--model_path", args.switti, "--prompt_config_path", "config/config_group_scales.yaml",
                         "--proj_path", str(proj), "--proj_2_path", str(proj2)], csd, log)
        normalize(raw, target)
        measured = json.loads(generation_timing.read_text())
        durations = [float(value) for value in measured["batch_durations_seconds"]]
        if measured["images"] != 31 or len(durations) != 31:
            raise RuntimeError(f"Expected 31 CSD-VAR generation timings, got {measured}")
        report = {"subject": subject, "images": 31,
                  "svd_seconds": svd_seconds, "training_full_seconds": train_seconds,
                  "preparation_seconds_to_used_checkpoint": prep_to_used_checkpoint,
                  "preparation_full_200_step_seconds": full_preparation_seconds,
                  "training_steps": args.steps, "inference_checkpoint_step": args.checkpoint_step,
                  "generation_seconds_total": sum(durations),
                  "generation_seconds_per_image_mean": statistics.mean(durations),
                  "generation_seconds_per_image_std": statistics.stdev(durations),
                  "inference_process_wall_seconds_diagnostic": infer_process,
                  "timing_contract": measured["timing_contract"],
                  "generation_durations_seconds": durations}
        reports.append(report)
        (target / "timing.json").write_text(json.dumps(report, indent=2) + "\n")
        print(f"[complete] {subject}: prep-to-step-{args.checkpoint_step}={prep_to_used_checkpoint:.2f}s "
              f"full-prep={full_preparation_seconds:.2f}s generation={statistics.mean(durations):.4f}s/image", flush=True)
    prep_values = [row["preparation_seconds_to_used_checkpoint"] for row in reports]
    full_prep_values = [row["preparation_full_200_step_seconds"] for row in reports]
    infer_values = [value for row in reports for value in row["generation_durations_seconds"]]
    summary = {"method": "CSD-VAR (Switti)", "protocol": "3 subjects x 31 prompts x seed 0", "images": 93,
               "preparation_mean_seconds_per_subject": statistics.mean(prep_values),
               "preparation_std_seconds_per_subject": statistics.stdev(prep_values),
               "full_200_step_preparation_mean_seconds_per_subject": statistics.mean(full_prep_values),
               "full_200_step_preparation_std_seconds_per_subject": statistics.stdev(full_prep_values),
               "generation_seconds_per_image_mean": statistics.mean(infer_values),
               "generation_seconds_per_image_std": statistics.stdev(infer_values),
               "generation_seconds_total": sum(infer_values),
               "preparation_timing_note": "Primary preparation is SVD plus time until every file of the checkpoint used for inference is written; full 200-step wall time is reported separately. Cached weight downloads and input staging are excluded.",
               "inference_timing_note": "Internally instrumented and CUDA-synchronized; includes prompt encoding, autoregressive sampling, and VQ-VAE decoding; excludes Switti/checkpoint loading and PNG output. Full process wall time is retained only as a diagnostic.",
               "per_subject": reports}
    summary_path = out_root.parent / "timing.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
