#!/usr/bin/env python3
"""Generate a normal prompt image with a learned style offset at one scale."""
from __future__ import annotations

import argparse
import os.path as osp
import sys
from pathlib import Path

sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__))))

import cv2
import torch

from tools.infer_offset_modspace import NULL, PROMPT, encode_texts, modspace_infer
from tools.optimize_ada_offset import get_scale_schedule, parse_scales
from tools.run_infinity import add_common_arguments, encode_prompt, load_tokenizer, load_transformer, load_visual_tokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser)
    parser.add_argument("--offset-path", required=True)
    parser.add_argument("--prompt", default="a dog")
    parser.add_argument("--output", required=True)
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument(
        "--offset-scales", default="0",
        help="scales receiving the learned style offset: '', '0', '1', or '0-1'",
    )
    parser.add_argument("--top_k", type=int, default=900)
    parser.add_argument("--top_p", type=float, default=0.97)
    args = parser.parse_args()
    if args.num_samples < 1:
        parser.error("--num-samples must be positive")
    offset_scales = set(parse_scales(args.offset_scales, 6)) if args.offset_scales else set()

    checkpoint = torch.load(args.offset_path, map_location="cpu")
    vae = load_visual_tokenizer(args)
    vae.requires_grad_(False)
    infinity = load_transformer(vae, args)
    infinity.requires_grad_(False)
    infinity.eval()
    tokenizer, encoder = load_tokenizer(t5_path=args.text_encoder_ckpt)

    offset = checkpoint["ada_offset"].reshape(-1, 1, 6, infinity.C).cuda().float()
    if offset.shape[0] < 6:
        raise ValueError(f"Expected a 0-5 per-scale checkpoint, got {tuple(offset.shape)}")
    if set(checkpoint.get("loss_scales", [])) != {0, 1, 2, 3, 4, 5}:
        raise ValueError(f"Checkpoint was not trained on exactly scales 0-5: {checkpoint.get('loss_scales')}")
    null_len = int(checkpoint.get("null_len", 32))
    h_div_w = checkpoint.get("h_div_w_template", 1.0)
    _, scale_schedule, _ = get_scale_schedule(args, h_div_w)
    if len(scale_schedule) <= 5:
        raise ValueError("The selected resolution does not contain scale 5")

    prompt_cond = encode_prompt(tokenizer, encoder, args.prompt)
    texts = encode_texts(infinity, null_len, prompt_cond)
    # Both SOS/shared AdaLN and cross-attention use the ordinary prompt.
    # Only the selected learned style offset is added.
    rows = [(PROMPT, PROMPT, True, 0.0), (NULL, NULL, True, 0.0)]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    samples = []
    for sample in range(args.num_samples):
        image = modspace_infer(
            infinity, vae, texts, offset, rows, scale_schedule,
            offset_scales=offset_scales, delta_scales=set(), cfg=1.0, tau=1.0,
            seed=args.seed + sample, args=args,
        )
        sample_path = output.with_name(f"{output.stem}_seed{args.seed + sample:03d}{output.suffix}")
        if not cv2.imwrite(str(sample_path), image):
            raise OSError(f"Could not write {sample_path}")
        samples.append(sample_path)
    # Keep a canonical first-sample path for the cross-scale comparison.
    if not cv2.imwrite(str(output), cv2.imread(str(samples[0]))):
        raise OSError(f"Could not write {output}")
    print(f"[saved] {len(samples)} image(s), prompt={args.prompt!r}, injected scales={sorted(offset_scales)} -> {output}")


if __name__ == "__main__":
    main()
