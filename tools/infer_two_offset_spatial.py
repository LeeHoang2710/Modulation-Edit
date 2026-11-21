#!/usr/bin/env python3
"""Generate one image with two subject offsets assigned to left/right regions.

At each early autoregressive scale, this implements the main-variant transform
independently for the two subjects and concatenates the six SharedAdaLN vectors
along the token grid.  Every transformer block still performs one global
self-attention operation, so tokens on opposite sides can interact naturally.

For region j in {left, right} and early-scale prompt weight a_s:

    T_j,s = (1-a_s) G_j + w_s D_j,s + a_s G_prompt

where w_s is 1-a_s for ``interpolate`` (the paper/main setting) or 1 for
``preserve-offset``.  The matching SA/FFN gates are spatialized together with
the scales and shifts; they are not used as subject mixture coefficients.
"""
from __future__ import annotations

import argparse
import json
import os.path as osp
import re
import sys
from pathlib import Path

sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__))))

import cv2
import torch

from tools.infer_offset_modspace import NULL, PROMPT, encode_texts_with_base, modspace_infer
from tools.optimize_ada_offset import get_scale_schedule, parse_scales
from tools.run_infinity import (
    add_common_arguments,
    encode_prompt,
    load_tokenizer,
    load_transformer,
    load_visual_tokenizer,
)
from tools.spatial_modulation import compose_horizontal_modulation, scale_offset


def _load_offset(path: str, channels: int, multiplier: float) -> tuple[dict, torch.Tensor]:
    checkpoint = torch.load(path, map_location="cpu")
    if "ada_offset" not in checkpoint:
        raise ValueError(f"{path} does not contain ada_offset")
    offset = checkpoint["ada_offset"].reshape(-1, 1, 6, channels).cuda().float() * multiplier
    if not torch.isfinite(offset).all():
        raise ValueError(f"{path} contains non-finite offset values")
    return checkpoint, offset


def _learned_scales(checkpoint: dict, offset: torch.Tensor, num_scales: int) -> set[int]:
    if "loss_scales" in checkpoint:
        return {int(scale) for scale in checkpoint["loss_scales"]}
    if offset.shape[0] == 1:
        return set(range(num_scales)) if offset.abs().sum().item() else set()
    return {index for index in range(offset.shape[0]) if offset[index].abs().sum().item()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser)
    parser.add_argument("--left-offset-path", required=True)
    parser.add_argument("--right-offset-path", required=True)
    parser.add_argument("--prompt", required=True, help="joint generation prompt containing both subjects")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--left-base-prompt", default="",
                        help="defaults to train_prompt stored in the left checkpoint")
    parser.add_argument("--right-base-prompt", default="",
                        help="defaults to train_prompt stored in the right checkpoint")
    parser.add_argument("--joint-base-prompt", default="",
                        help="CFG/base cross-attention prompt; defaults to '<left> and <right>'")
    parser.add_argument("--left-preview-name", default="left_subject",
                        help="filename prefix for the left offset's independent preview")
    parser.add_argument("--right-preview-name", default="right_subject",
                        help="filename prefix for the right offset's independent preview")
    parser.add_argument("--alpha-schedule", default="0.1,0.1,0.1,0.1,0.1",
                        help="main-variant prompt weights for consecutive early scales")
    parser.add_argument("--early-composition", choices=["interpolate", "preserve-offset"],
                        default="interpolate")
    parser.add_argument("--offset-scales", default="auto",
                        help="auto uses each checkpoint's trained scales; otherwise e.g. '0,2-5'")
    parser.add_argument("--split-ratio", type=float, default=0.5,
                        help="fraction of image width assigned to the left subject")
    parser.add_argument("--left-offset-scale", type=float, default=1.0)
    parser.add_argument("--right-offset-scale", type=float, default=1.0)
    parser.add_argument("--early-cfg", type=float, default=1.2)
    parser.add_argument("--late-cfg", type=float, default=2.0)
    parser.add_argument("--late-gss", choices=["prompt", "base"], default="prompt")
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--top-k", type=int, default=900)
    parser.add_argument("--top-p", type=float, default=0.97)
    parser.set_defaults(h_div_w_template=-1.)
    args = parser.parse_args()

    if not (args.use_bit_label and args.vae_type != 0 and not args.apply_spatial_patchify):
        parser.error("Requires bitwise, non-patchified Infinity-2B settings")
    if not 0.0 < args.split_ratio < 1.0:
        parser.error("--split-ratio must be between 0 and 1")
    if args.num_samples < 1:
        parser.error("--num-samples must be positive")
    for option, value in (("--left-preview-name", args.left_preview_name),
                          ("--right-preview-name", args.right_preview_name)):
        if re.fullmatch(r"[A-Za-z0-9_-]+", value) is None:
            parser.error(f"{option} may contain only letters, digits, underscores, and hyphens")
    alpha = tuple(float(value) for value in args.alpha_schedule.split(",") if value.strip())
    if not alpha or any(value < 0 or value > 1 for value in alpha):
        parser.error("--alpha-schedule must contain comma-separated values in [0, 1]")

    vae = load_visual_tokenizer(args)
    vae.requires_grad_(False)
    infinity = load_transformer(vae, args)
    infinity.requires_grad_(False)
    infinity.eval()
    tokenizer, encoder = load_tokenizer(t5_path=args.text_encoder_ckpt)

    left_checkpoint, left_offset = _load_offset(
        args.left_offset_path, infinity.C, args.left_offset_scale
    )
    right_checkpoint, right_offset = _load_offset(
        args.right_offset_path, infinity.C, args.right_offset_scale
    )
    left_base = args.left_base_prompt or left_checkpoint.get("train_prompt", "")
    right_base = args.right_base_prompt or right_checkpoint.get("train_prompt", "")
    if not left_base or not right_base:
        parser.error("Both base prompts are required (pass them explicitly or use class-conditioned checkpoints)")
    joint_base = args.joint_base_prompt or f"{left_base} and {right_base}"

    left_ratio = left_checkpoint.get("h_div_w_template", 1.0)
    right_ratio = right_checkpoint.get("h_div_w_template", 1.0)
    if args.h_div_w_template <= 0 and abs(left_ratio - right_ratio) > 1e-6:
        parser.error("Checkpoint aspect ratios differ; pass --h_div_w_template explicitly")
    h_div_w = args.h_div_w_template if args.h_div_w_template > 0 else left_ratio
    _, scale_schedule, _ = get_scale_schedule(args, h_div_w)
    if len(alpha) >= len(scale_schedule):
        parser.error("The alpha schedule must leave at least one late generation scale")

    requested = (set(range(len(scale_schedule))) if args.offset_scales == "auto"
                 else set(parse_scales(args.offset_scales, len(scale_schedule))))
    left_active = requested & _learned_scales(left_checkpoint, left_offset, len(scale_schedule))
    right_active = requested & _learned_scales(right_checkpoint, right_offset, len(scale_schedule))

    prompt_cond = encode_prompt(tokenizer, encoder, args.prompt)
    left_cond = encode_prompt(tokenizer, encoder, left_base)
    right_cond = encode_prompt(tokenizer, encoder, right_base)
    joint_cond = encode_prompt(tokenizer, encoder, joint_base)
    texts = encode_texts_with_base(infinity, joint_cond, prompt_cond)
    # Reuse the same encoder path to obtain exactly the pretrained G_left/G_right
    # transformations expected by their class-conditioned residuals.
    left_gss = encode_texts_with_base(infinity, left_cond, prompt_cond)[1][NULL:NULL + 1]
    right_gss = encode_texts_with_base(infinity, right_cond, prompt_cond)[1][NULL:NULL + 1]
    prompt_gss = texts[1][PROMPT:PROMPT + 1]

    def spatialize(scale_index, scale_shape, gss):
        if (scale_index >= len(alpha)
                or (scale_index not in left_active and scale_index not in right_active)):
            return gss
        value = alpha[scale_index]
        offset_weight = 1.0 - value if args.early_composition == "interpolate" else 1.0
        left_delta = (scale_offset(left_offset, scale_index)
                      if scale_index in left_active else torch.zeros_like(left_offset[:1]))
        right_delta = (scale_offset(right_offset, scale_index)
                       if scale_index in right_active else torch.zeros_like(right_offset[:1]))

        left_prompt = (1.0 - value) * left_gss + offset_weight * left_delta + value * prompt_gss
        right_prompt = (1.0 - value) * right_gss + offset_weight * right_delta + value * prompt_gss
        left_base_row = left_gss + left_delta
        right_base_row = right_gss + right_delta
        conditional = compose_horizontal_modulation(
            left_prompt, right_prompt, scale_shape, args.split_ratio
        )
        cfg_partner = compose_horizontal_modulation(
            left_base_row, right_base_row, scale_shape, args.split_ratio
        )
        return torch.cat((conditional, cfg_partner), dim=0).to(gss)

    early_count = len(alpha)
    cfg_by_scale = [args.early_cfg] * early_count + [args.late_cfg] * (
        len(scale_schedule) - early_count
    )
    spatial_scales = {
        scale for scale in range(min(early_count, len(scale_schedule)))
        if scale in left_active or scale in right_active
    }
    late_sources = ({scale: PROMPT for scale in range(len(scale_schedule)) if scale not in spatial_scales}
                    if args.late_gss == "prompt" else {})
    rows = [(NULL, PROMPT, False, 0.0), (NULL, NULL, False, 0.0)]
    print("[conditioning] conditional ca_kv=joint prompt at every scale; "
          "CFG partner ca_kv=joint subject base", flush=True)
    dummy_offset = torch.zeros_like(left_offset[:1])
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    def render_single_subject(name, subject_cond, offset, active_scales):
        """Render one learned offset globally before attempting composition."""
        subject_texts = encode_texts_with_base(infinity, subject_cond, subject_cond)
        subject_rows = [(NULL, PROMPT, True, 0.0), (NULL, NULL, True, 0.0)]
        subject_sources = (
            {scale: PROMPT for scale in range(len(scale_schedule))
             if scale not in active_scales}
            if args.late_gss == "prompt" else {}
        )
        for sample in range(args.num_samples):
            seed = args.seed + sample
            image = modspace_infer(
                infinity, vae, subject_texts, offset, subject_rows, scale_schedule,
                active_scales, set(), args.early_cfg, 1.0, seed, args,
                cfg_by_scale=cfg_by_scale,
                conditional_gss_source_by_scale=subject_sources,
            )
            path = output_dir / f"{name}_seed_{seed:03d}.png"
            if not cv2.imwrite(str(path), image):
                raise OSError(f"Could not write {path}")
            print(f"[individual image] {path}", flush=True)

    print("[phase 1/2] rendering each learned offset independently", flush=True)
    render_single_subject(args.left_preview_name, left_cond, left_offset, left_active)
    render_single_subject(args.right_preview_name, right_cond, right_offset, right_active)
    print("[phase 2/2] rendering the spatial two-subject composition", flush=True)
    for sample in range(args.num_samples):
        seed = args.seed + sample
        image = modspace_infer(
            infinity, vae, texts, dummy_offset, rows, scale_schedule,
            set(), set(), args.early_cfg, 1.0, seed, args,
            cfg_by_scale=cfg_by_scale,
            conditional_gss_source_by_scale=late_sources,
            modulation_transform=spatialize,
        )
        path = output_dir / f"seed_{seed:03d}.png"
        if not cv2.imwrite(str(path), image):
            raise OSError(f"Could not write {path}")
        print(f"[image] {path}", flush=True)

    (output_dir / "metadata.json").write_text(json.dumps({
        "prompt": args.prompt,
        "left_offset_path": args.left_offset_path,
        "right_offset_path": args.right_offset_path,
        "left_base_prompt": left_base,
        "right_base_prompt": right_base,
        "joint_base_prompt": joint_base,
        "split_ratio": args.split_ratio,
        "alpha_schedule": alpha,
        "early_composition": args.early_composition,
        "left_active_scales": sorted(left_active),
        "right_active_scales": sorted(right_active),
        "cfg_by_scale": cfg_by_scale,
        "conditional_ca_kv": "joint prompt at every scale",
        "cfg_partner_ca_kv": "joint subject base at every scale",
        "individual_previews": {
            "left": args.left_preview_name,
            "right": args.right_preview_name,
            "prompts": [left_base, right_base],
        },
        "seeds": list(range(args.seed, args.seed + args.num_samples)),
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
