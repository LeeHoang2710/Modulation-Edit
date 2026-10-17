#!/usr/bin/env python3
"""Generate a small ARBooth ViCo subset without changing released ARBooth code.

The full protocol continues to use ``tools/run_arbooth_for_eval.py`` exactly
as released.  This helper merely imports its public loading/sampling helpers
to make a one-subject, few-prompt smoke test affordable.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arbooth-root", default="models/ARBooth")
    parser.add_argument("--manifest", default="benchmark/manifests/vico.json")
    parser.add_argument("--subject", default="brown_dog")
    parser.add_argument("--max-prompts", type=int, default=5)
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--output", default="benchmark_outputs_vico_arbooth_smoke")
    parser.add_argument("--model-path", default="weights/arbooth/by_subject")
    parser.add_argument("--vae-path", default="weights/infinity/infinity_vae_d32reg.pth")
    parser.add_argument("--text-encoder-ckpt", default="weights/infinity/flan-t5-xl")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cfg", type=float, default=4.0)
    parser.add_argument("--tau", type=float, default=0.5)
    parser.add_argument("--pn", default="1M")
    parser.add_argument("--vae-type", type=int, default=32)
    parser.add_argument("--add-lvl-embeding-only-first-block", type=int, default=1)
    parser.add_argument("--use-bit-label", type=int, default=1)
    parser.add_argument("--rope2d-each-sa-layer", type=int, default=1)
    parser.add_argument("--rope2d-normalized-by-hw", type=int, default=2)
    parser.add_argument("--use-scale-schedule-embedding", type=int, default=0)
    parser.add_argument("--apply-spatial-patchify", type=int, default=0)
    parser.add_argument("--checkpoint-type", default="torch")
    parser.add_argument("--bf16", type=int, default=1)
    parser.add_argument("--text-channels", type=int, default=2048)
    parser.add_argument("--enable-model-cache", type=int, default=0)
    args = parser.parse_args()
    if args.max_prompts < 1 or args.num_samples < 1:
        parser.error("--max-prompts and --num-samples must be positive")

    arbooth_root = (root / args.arbooth_root).resolve()
    if not (arbooth_root / "tools/run_arbooth_for_eval.py").is_file():
        raise SystemExit(f"Missing released ARBooth source: {arbooth_root}")
    sys.path.insert(0, str(arbooth_root))
    from tools.run_arbooth_for_eval import (  # type: ignore[import-not-found]
        gen_one_img, load_tokenizer, load_transformer, load_visual_tokenizer,
    )
    from infinity.utils.dynamic_resolution import dynamic_resolution_h_w  # type: ignore[import-not-found]
    import cv2
    import torch

    manifest = json.loads((root / args.manifest).read_text())
    item = next((entry for entry in manifest["subjects"] if entry["subject"] == args.subject), None)
    if item is None:
        raise SystemExit(f"Unknown ViCo subject: {args.subject}")
    prompts = item["prompt_templates"][:args.max_prompts]
    if len(prompts) != args.max_prompts:
        raise SystemExit(f"Only {len(prompts)} prompts are available for {args.subject}")

    for attr in ("model_path", "vae_path", "text_encoder_ckpt"):
        value = (root / getattr(args, attr)).resolve()
        if not value.exists():
            raise SystemExit(f"Missing {attr.replace('_', ' ')}: {value}")
        setattr(args, attr, str(value))
    # Match ARBooth's argparse attribute names consumed by its helper functions.
    args.cfg_insertion_layer = 0
    args.prefix_embeds_length = 0
    args.text_prefix_length = 4
    args.model_type = "infinity_2b"
    args.sampling_per_bits = 1
    args.h_div_w_template = 1.0
    args.use_flex_attn = 0

    tokenizer, text_encoder = load_tokenizer(args.text_encoder_ckpt)
    special_token = "<placeholder>"
    tokenizer.add_special_tokens({"additional_special_tokens": [special_token]})
    text_encoder.resize_token_embeddings(len(tokenizer))
    special_token_id = tokenizer.convert_tokens_to_ids(special_token)
    vae = load_visual_tokenizer(args)
    scales = [(1, h, w) for _, h, w in dynamic_resolution_h_w[args.h_div_w_template][args.pn]["scales"]]
    infinity = load_transformer(vae, args, args.subject)

    output = (root / args.output / args.subject).resolve()
    output.mkdir(parents=True, exist_ok=True)
    total_seconds = 0.0
    generated = 0
    with torch.no_grad():
        for prompt_index, template in enumerate(prompts):
            prompt_dir = output / f"prompt_{prompt_index:02d}"
            prompt_dir.mkdir(parents=True, exist_ok=True)
            for sample_index in range(args.num_samples):
                started = time.perf_counter()
                image = gen_one_img(
                    infinity, vae, tokenizer, text_encoder,
                    template.format(f"<placeholder> {item['class_name']}"),
                    g_seed=args.seed + sample_index, cfg_list=args.cfg, tau_list=args.tau,
                    scale_schedule=scales, cfg_insertion_layer=[args.cfg_insertion_layer],
                    vae_type=args.vae_type, sampling_per_bits=args.sampling_per_bits,
                    special_token_id=special_token_id, special_token=infinity.special_token,
                )
                cv2.imwrite(str(prompt_dir / f"{sample_index:02d}.jpg"), image.cpu().numpy())
                total_seconds += time.perf_counter() - started
                generated += 1
    timing = {
        "subject": args.subject, "prompts": len(prompts), "samples_per_prompt": args.num_samples,
        "images": generated, "inference_seconds_per_image": total_seconds / generated,
        "note": "Smoke-only wrapper; imports released ARBooth helpers without modifying released source.",
    }
    (output / "timing.json").write_text(json.dumps(timing, indent=2) + "\n")
    print(f"[complete] {generated} ARBooth smoke images; wall time={timing['inference_seconds_per_image']:.3f}s/image")


if __name__ == "__main__":
    main()
