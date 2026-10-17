#!/usr/bin/env python3
"""Run ARBooth's released 16 x 31 x 8 evaluation generator without changing it.

The released ``tools/run_arbooth_for_eval.py`` owns prompt construction,
sampling, and output names.  This wrapper only validates the local asset
layout, streams its output to a log, and records the two timing values printed
by the official generator.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path


EXPECTED_SUBJECTS = (
    "alarm_clock", "black_cat", "brown_cat", "brown_dog", "brown_dog2",
    "brown_teddybear", "cat_statue", "duck_toy", "elephant_statue", "fat_dog",
    "monster_toy", "pink_sunglasses", "red_teapot", "red_vase", "tortoise_plushy",
    "wooden_barn",
)
COST = re.compile(r"cost:\s*([0-9.]+),\s*infinity cost=([0-9.]+)")
SAVED = re.compile(r"Save to (.+)$")


def required(path: Path, label: str) -> Path:
    if not path.exists():
        raise SystemExit(f"Missing {label}: {path}")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arbooth-root", default="models/ARBooth")
    parser.add_argument("--checkpoint-root", default="weights/arbooth/by_subject")
    parser.add_argument("--model-path", default="weights/infinity/infinity_2b_reg.pth")
    parser.add_argument("--vae-path", default="weights/infinity/infinity_vae_d32reg.pth")
    parser.add_argument("--text-encoder-ckpt", default="weights/infinity/flan-t5-xl")
    parser.add_argument("--output", default="benchmark_outputs_vico_arbooth")
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cfg", type=int, default=4)
    parser.add_argument("--tau", type=float, default=0.5)
    args = parser.parse_args()
    if args.num_samples != 8:
        parser.error("This paper-reproduction runner intentionally requires --num-samples 8")

    root = Path(__file__).resolve().parents[1]
    arbooth = required((root / args.arbooth_root).resolve(), "ARBooth source directory")
    checkpoints = required((root / args.checkpoint_root).resolve(), "ARBooth checkpoint directory")
    model_path = required((root / args.model_path).resolve(), "Infinity transformer weight")
    vae_path = required((root / args.vae_path).resolve(), "Infinity VAE weight")
    encoder = required((root / args.text_encoder_ckpt).resolve(), "Flan-T5 encoder")
    required(arbooth / "tools/run_arbooth_for_eval.py", "official ARBooth evaluator entry point")
    required(arbooth / "inputs", "ARBooth ViCo inputs")
    for subject in EXPECTED_SUBJECTS:
        required(checkpoints / subject, f"checkpoint folder for {subject}")

    output = (root / args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable, "tools/run_arbooth_for_eval.py",
        "--model_path", str(checkpoints), "--result_dir", str(output),
        "--num_samples", "8", "--seed", str(args.seed), "--cfg", str(args.cfg), "--tau", str(args.tau),
        "--pn", "1M", "--vae_type", "32", "--vae_path", str(vae_path),
        "--text_encoder_ckpt", str(encoder), "--model_type", "infinity_2b",
        "--add_lvl_embeding_only_first_block", "1", "--use_bit_label", "1",
        "--rope2d_each_sa_layer", "1", "--rope2d_normalized_by_hw", "2",
        "--use_scale_schedule_embedding", "0", "--apply_spatial_patchify", "0",
        "--checkpoint_type", "torch", "--bf16", "1", "--text_channels", "2048",
        "--enable_model_cache", "0",
    ]
    print("[exec] " + " ".join(command), flush=True)
    log_path = output / "run_arbooth.log"
    by_subject: dict[str, list[tuple[float, float]]] = defaultdict(list)
    pending_cost: tuple[float, float] | None = None
    with log_path.open("w") as log:
        process = subprocess.Popen(command, cwd=arbooth, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, bufsize=1)
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
            cost = COST.search(line)
            if cost:
                pending_cost = (float(cost.group(1)), float(cost.group(2)))
                continue
            saved = SAVED.search(line)
            if saved and pending_cost is not None:
                saved_path = Path(saved.group(1))
                # .../<subject>/prompt_XX/YY.jpg
                if saved_path.parent.parent.name in EXPECTED_SUBJECTS:
                    by_subject[saved_path.parent.parent.name].append(pending_cost)
                    pending_cost = None
        return_code = process.wait()
    if return_code:
        raise SystemExit(return_code)

    expected_total = len(EXPECTED_SUBJECTS) * 31 * args.num_samples
    actual_total = sum(1 for subject in EXPECTED_SUBJECTS for path in (output / subject).glob("prompt_*/*.jpg") if path.is_file())
    if actual_total != expected_total:
        raise SystemExit(f"Expected {expected_total} ARBooth images, found {actual_total} under {output}")
    if sum(len(values) for values in by_subject.values()) != expected_total:
        raise SystemExit("Could not associate every official timing line with a saved output image")

    for subject in EXPECTED_SUBJECTS:
        full = [value[0] for value in by_subject[subject]]
        inference = [value[1] for value in by_subject[subject]]
        (output / subject / "timing.json").write_text(json.dumps({
            "subject": subject,
            "images": len(full),
            "preparation_seconds": None,
            "end_to_end_seconds_per_image": sum(full) / len(full),
            "inference_seconds_per_image": sum(inference) / len(inference),
            "timing_source": "official ARBooth gen_one_img log: cost / infinity cost",
        }, indent=2) + "\n")
    summary = {
        "subjects": len(EXPECTED_SUBJECTS), "prompts_per_subject": 31,
        "samples_per_prompt": args.num_samples, "images": actual_total,
        "end_to_end_seconds_per_image": sum(value[0] for values in by_subject.values() for value in values) / actual_total,
        "inference_seconds_per_image": sum(value[1] for values in by_subject.values() for value in values) / actual_total,
        "log": str(log_path),
    }
    (output / "timing_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"[complete] {actual_total} images; official infinity time={summary['inference_seconds_per_image']:.3f}s/image", flush=True)


if __name__ == "__main__":
    main()
