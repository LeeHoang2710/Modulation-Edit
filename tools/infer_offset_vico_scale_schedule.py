#!/usr/bin/env python3
"""Generate ViCo prompts with the current scale-scheduled modulation method.

For the default five learned early scales:
  s=0..4: (1-a_s) * (G_null + D_s) + a_s * G_prompt,
           a=[.25,.20,.15,.10,.05], CFG=1.2
  s=5..12: G_prompt (or the null-text ablation G_null), CFG=2.0

The conditional cross-attention ``ca_kv`` always uses the prompt. The CFG
partner uses null ``ca_kv`` and its own null modulation; this makes CFG act on
the prompt direction rather than amplifying the learned subject offset.
"""
from __future__ import annotations

import argparse
import json
import os.path as osp
import sys
from pathlib import Path

sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__))))

import cv2
import torch

from tools.infer_offset_modspace import NULL, PROMPT, encode_texts, modspace_infer
from tools.optimize_ada_offset import get_scale_schedule
from tools.run_infinity import add_common_arguments, encode_prompt, load_tokenizer, load_transformer, load_visual_tokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser)
    parser.add_argument("--offset-path", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--subject", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--alpha-schedule", default="0.25,0.20,0.15,0.10,0.05")
    parser.add_argument("--early-cfg", type=float, default=1.2)
    parser.add_argument("--late-cfg", type=float, default=2.0)
    parser.add_argument("--late-gss", choices=["prompt", "null"], default="prompt",
                        help="shared AdaLN modulation at scales after the alpha schedule; ca_kv remains prompt in both variants")
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=900)
    parser.add_argument("--top-p", type=float, default=0.97)
    parser.set_defaults(h_div_w_template=-1.)
    args = parser.parse_args()
    if not (args.use_bit_label and args.vae_type != 0 and not args.apply_spatial_patchify):
        parser.error("Requires bitwise, non-patchified Infinity-2B settings")
    alpha = tuple(float(value) for value in args.alpha_schedule.split(",") if value.strip())
    if not alpha or any(value < 0 or value > 1 for value in alpha):
        parser.error("--alpha-schedule must be comma-separated values in [0, 1]")

    manifest = json.loads(Path(args.manifest).read_text())
    item = next((entry for entry in manifest["subjects"] if entry["subject"] == args.subject), None)
    if item is None:
        parser.error(f"Unknown ViCo subject: {args.subject}")
    checkpoint = torch.load(args.offset_path, map_location="cpu")

    vae = load_visual_tokenizer(args)
    vae.requires_grad_(False)
    infinity = load_transformer(vae, args)
    infinity.requires_grad_(False)
    infinity.eval()
    tokenizer, encoder = load_tokenizer(t5_path=args.text_encoder_ckpt)
    offset = checkpoint["ada_offset"].reshape(-1, 1, 6, infinity.C).cuda()
    null_len = checkpoint.get("null_len", 32)
    h_div_w = args.h_div_w_template if args.h_div_w_template > 0 else checkpoint.get("h_div_w_template", 1.0)
    _, scale_schedule, _ = get_scale_schedule(args, h_div_w)
    early_count = len(alpha)
    if early_count >= len(scale_schedule):
        parser.error(f"alpha schedule has {early_count} scales but generation has only {len(scale_schedule)}")
    learned = set(checkpoint.get("loss_scales", range(offset.shape[0])))
    early_scales = set(range(early_count))
    if not early_scales.issubset(learned):
        parser.error(f"Offset does not contain all required learned early scales: {sorted(early_scales - learned)}")
    if offset.shape[0] != 1 and offset.shape[0] < early_count:
        parser.error(f"Offset contains {offset.shape[0]} scale slots; need at least {early_count}")

    coefficients = {scale: (1.0 - value, value) for scale, value in enumerate(alpha)}
    cfg_by_scale = [args.early_cfg] * early_count + [args.late_cfg] * (len(scale_schedule) - early_count)
    # The CFG partner always remains G_null. This ablation changes only the
    # conditional shared AdaLN modulation after the learned early scales.
    gss_sources = ({scale: PROMPT for scale in range(early_count, len(scale_schedule))}
                   if args.late_gss == "prompt" else {})
    rows = [(NULL, PROMPT, True, 0.0), (NULL, NULL, True, 0.0)]
    out_root = Path(args.out_dir) / args.subject
    out_root.mkdir(parents=True, exist_ok=True)
    prompt_cache = {}
    for index, template in enumerate(item["prompt_templates"]):
        prompt = template.format(item["class_name"])
        if prompt not in prompt_cache:
            prompt_cache[prompt] = encode_prompt(tokenizer, encoder, prompt)
        texts = encode_texts(infinity, null_len, prompt_cache[prompt])
        prompt_dir = out_root / f"prompt_{index:02d}"
        prompt_dir.mkdir(parents=True, exist_ok=True)
        (prompt_dir / "prompt.txt").write_text(prompt + "\n")
        for sample in range(args.num_samples):
            image = modspace_infer(infinity, vae, texts, offset, rows, scale_schedule,
                                   early_scales, early_scales, args.early_cfg, 1.0,
                                   args.seed + sample, args, scale_coefficients=coefficients,
                                   cfg_by_scale=cfg_by_scale,
                                   conditional_gss_source_by_scale=gss_sources)
            path = prompt_dir / f"seed_{sample:03d}.png"
            if not cv2.imwrite(str(path), image):
                raise OSError(f"Could not write {path}")
        print(f"[saved] {prompt_dir} ({args.num_samples} samples)", flush=True)


if __name__ == "__main__":
    main()
