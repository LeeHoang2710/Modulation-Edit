#!/usr/bin/env python3
"""Time ARBooth on three ViCo subjects, all 31 prompts, and one fixed seed."""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
from pathlib import Path


def stats(values: list[float]) -> dict[str, float]:
    return {
        "generation_seconds_per_image_mean": statistics.mean(values),
        "generation_seconds_per_image_std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "pure_generation_seconds_per_image_mean": statistics.mean(values),
        "pure_generation_seconds_per_image_std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "generation_seconds_total": sum(values),
    }


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser()
    p.add_argument("--arbooth-root", default="/workspace/ARBooth")
    p.add_argument("--data-dir", default="/workspace/CSD-VAR/benchmark_data/vico_csd")
    p.add_argument("--subjects", default="brown_dog,black_cat,alarm_clock")
    p.add_argument("--checkpoint-root", default="weights/arbooth/by_subject")
    p.add_argument("--vae-path", default="weights/infinity_vae_d32reg.pth")
    p.add_argument("--text-encoder-ckpt", default="weights/flan-t5-xl")
    p.add_argument("--output", default="output/timing/arbooth_3subjects_93")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cfg", type=float, default=4.0)
    p.add_argument("--tau", type=float, default=0.5)
    p.add_argument("--pn", default="1M")
    p.add_argument("--vae-type", type=int, default=32)
    p.add_argument("--warmup-runs", type=int, default=1)
    args = p.parse_args()

    subjects = [value.strip() for value in args.subjects.split(",") if value.strip()]
    if len(subjects) != 3:
        p.error("--subjects must contain exactly three comma-separated subjects")
    data_dir = Path(args.data_dir).resolve()
    subject_csv = data_dir / "subjects.csv"
    if not subject_csv.is_file():
        raise SystemExit(
            f"Missing ViCo manifest: {subject_csv}\n"
            "Run: bash scripts/prepare_vico_beta_alpha.sh"
        )
    entries = {}
    with subject_csv.open(newline="") as handle:
        for row in csv.DictReader(handle):
            prompt_file = data_dir / row["prompts"]
            if not prompt_file.is_file():
                raise SystemExit(f"Missing prompt file: {prompt_file}")
            row["prompt_templates"] = [
                line.strip() for line in prompt_file.read_text().splitlines() if line.strip()
            ]
            entries[row["subject"]] = row
    missing = [subject for subject in subjects if subject not in entries]
    if missing:
        p.error(f"subjects absent from manifest: {missing}")
    for subject in subjects:
        if len(entries[subject]["prompt_templates"]) != 31:
            p.error(f"{subject} does not have exactly 31 prompts")

    arbooth = Path(args.arbooth_root).resolve()
    sys.path.insert(0, str(arbooth))
    from tools.run_arbooth_for_eval import encode_prompt, gen_one_img, load_tokenizer, load_transformer, load_visual_tokenizer
    from infinity.utils.dynamic_resolution import dynamic_resolution_h_w
    import cv2
    import torch

    def resolve(value: str) -> str:
        path = Path(value)
        return str((path if path.is_absolute() else root / path).resolve())

    args.model_path = resolve(args.checkpoint_root)
    args.vae_path = resolve(args.vae_path)
    args.text_encoder_ckpt = resolve(args.text_encoder_ckpt)
    args.cfg_insertion_layer = 0
    args.prefix_embeds_length = 0
    args.text_prefix_length = 4
    args.model_type = "infinity_2b"
    args.sampling_per_bits = 1
    args.h_div_w_template = 1.0
    args.use_flex_attn = 0
    args.add_lvl_embeding_only_first_block = 1
    args.use_bit_label = 1
    args.rope2d_each_sa_layer = 1
    args.rope2d_normalized_by_hw = 2
    args.use_scale_schedule_embedding = 0
    args.apply_spatial_patchify = 0
    args.checkpoint_type = "torch"
    args.bf16 = 1
    args.text_channels = 2048
    args.enable_model_cache = 0

    tokenizer, text_encoder = load_tokenizer(args.text_encoder_ckpt)
    tokenizer.add_special_tokens({"additional_special_tokens": ["<placeholder>"]})
    text_encoder.resize_token_embeddings(len(tokenizer))
    token_id = tokenizer.convert_tokens_to_ids("<placeholder>")
    vae = load_visual_tokenizer(args)
    scales = [(1, h, w) for _, h, w in dynamic_resolution_h_w[1.0][args.pn]["scales"]]
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)

    all_durations: list[float] = []
    all_prompt_durations: list[float] = []
    subject_reports = []
    with torch.no_grad():
        for subject in subjects:
            item = entries[subject]
            # Subject-specific checkpoint loading is intentionally outside inference timing.
            infinity = load_transformer(vae, args, subject)
            subject_dir = output / subject
            subject_dir.mkdir(parents=True, exist_ok=True)
            durations = []
            prompt_durations = []
            if args.warmup_runs:
                warm_prompt = item["prompt_templates"][0].format(f"<placeholder> {item['class_name']}")
                warm_cond = encode_prompt(
                    tokenizer, text_encoder, warm_prompt,
                    special_token_id=token_id, special_token=infinity.special_token,
                )
                for warm_index in range(args.warmup_runs):
                    gen_one_img(
                        infinity, vae, tokenizer, text_encoder, warm_prompt,
                        g_seed=args.seed - 10000 - warm_index, cfg_list=args.cfg, tau_list=args.tau,
                        scale_schedule=scales, cfg_insertion_layer=[0], vae_type=args.vae_type,
                        sampling_per_bits=1, special_token_id=token_id,
                        special_token=infinity.special_token, text_cond_tuple=warm_cond,
                    )
                torch.cuda.synchronize()
            for prompt_index, template in enumerate(item["prompt_templates"]):
                prompt = template.format(f"<placeholder> {item['class_name']}")
                torch.cuda.synchronize()
                prompt_started = time.perf_counter()
                text_cond = encode_prompt(
                    tokenizer, text_encoder, prompt,
                    special_token_id=token_id, special_token=infinity.special_token,
                )
                torch.cuda.synchronize()
                prompt_durations.append(time.perf_counter() - prompt_started)
                torch.cuda.synchronize()
                started = time.perf_counter()
                image = gen_one_img(
                    infinity, vae, tokenizer, text_encoder, prompt,
                    g_seed=args.seed, cfg_list=args.cfg, tau_list=args.tau,
                    scale_schedule=scales, cfg_insertion_layer=[0], vae_type=args.vae_type,
                    sampling_per_bits=1, special_token_id=token_id,
                    special_token=infinity.special_token,
                    text_cond_tuple=text_cond,
                )
                torch.cuda.synchronize()
                durations.append(time.perf_counter() - started)
                cv2.imwrite(str(subject_dir / f"prompt_{prompt_index:02d}_seed_{args.seed}.jpg"), image.cpu().numpy())
            report = {
                "subject": subject, "images": 31, "seed": args.seed,
                "prompt_encoding_seconds_mean": statistics.mean(prompt_durations),
                "prompt_encoding_seconds_std": statistics.stdev(prompt_durations),
                **stats(durations),
            }
            subject_reports.append(report)
            all_durations.extend(durations)
            all_prompt_durations.extend(prompt_durations)
            del infinity
            torch.cuda.empty_cache()

    report = {
        "method": "ARBooth", "protocol": "3 subjects x 31 prompts x 1 fixed seed",
        "subjects": subjects, "images": len(all_durations), "seed": args.seed,
        "model_load_excluded": True, "image_write_excluded": True,
        "preparation_seconds": None,
        "preparation_note": "Released checkpoints were used; personalization/training was not rerun, so preparation time is unavailable rather than replaced by checkpoint loading.",
        "warmup_runs_per_subject": args.warmup_runs,
        "prompt_encoding_seconds_mean": statistics.mean(all_prompt_durations),
        "prompt_encoding_seconds_std": statistics.stdev(all_prompt_durations),
        "timing_contract": "pure generation: autoregressive sampling + VAE decoding; excludes prompt encoding, model/checkpoint loading, device-to-host transfer and image writes; CUDA synchronized",
        **stats(all_durations), "per_subject": subject_reports,
        "per_image_seconds": all_durations,
    }
    if report["images"] != 93:
        raise RuntimeError(f"expected 93 images, generated {report['images']}")
    (output / "timing.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"ARBooth: {report['generation_seconds_per_image_mean']:.4f} +/- {report['generation_seconds_per_image_std']:.4f} s/image (n=93)")


if __name__ == "__main__":
    main()
