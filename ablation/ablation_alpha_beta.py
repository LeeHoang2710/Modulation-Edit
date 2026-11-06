"""Sweep prompt-modulation alpha and subject-offset strength beta.

The conditional shared-AdaLN modulation is exactly

    G_s = G_subject + beta * D_s + alpha * G_prompt,  s in {0, 2, 3, 4}
          G_subject,                                 otherwise

The full modification prompt still enters conditional cross-attention at every
scale.  The CFG partner uses the same subject modulation/offset and null
cross-attention, matching ``ablation.common``.
"""
import argparse
import os.path as osp

import torch

from common import Variant, add_args, read_vico, run, subject_name


def csv_floats(value):
    values = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("expected at least one comma-separated number")
    return values


def validate_offsets(args):
    """Reject stale null-conditioned or otherwise mismatched checkpoints."""
    subjects = read_vico(args.data_dir, args.subjects, args.num_prompts, args.prompt_name)
    expected_scales = set(range(args.loss_scales_end + 1))
    for item in subjects:
        path = osp.join(args.offset_root, item["subject"], "offset.pth")
        if not osp.isfile(path):
            raise FileNotFoundError(f"missing offset checkpoint: {path}")
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        expected_prompt = subject_name(item["subject"])
        problems = []
        if checkpoint.get("train_prompt") != expected_prompt:
            problems.append(
                f"train_prompt={checkpoint.get('train_prompt')!r}, expected {expected_prompt!r}"
            )
        if set(checkpoint.get("loss_scales", ())) != expected_scales:
            problems.append(
                f"loss_scales={checkpoint.get('loss_scales')!r}, "
                f"expected 0-{args.loss_scales_end}"
            )
        if checkpoint.get("hflip") not in (0, False):
            problems.append(f"hflip={checkpoint.get('hflip')!r}, expected 0")
        expected_values = {
            "iters": 200,
            "batch_size": 2,
            "grad_accum": 2,
            "offset_init": "zeros",
        }
        for key, expected in expected_values.items():
            if checkpoint.get(key) != expected:
                problems.append(f"{key}={checkpoint.get(key)!r}, expected {expected!r}")
        if abs(float(checkpoint.get("lr", -1)) - 5e-3) > 1e-12:
            problems.append(f"lr={checkpoint.get('lr')!r}, expected 0.005")
        if problems:
            raise ValueError(f"{path}: " + "; ".join(problems))


def variant(name, alpha, beta, args):
    return Variant(
        name=name,
        alpha=alpha,
        blend="addp",                 # G_subject + beta*D + alpha*G_prompt
        ca="prompt",                  # full prompt CA at every scale
        offset="learned",
        offset_root=args.offset_root,
        offset_scales=args.offset_scales,
        offset_weight=beta,
        base_text="subject",
        blend_scales=args.offset_scales,
        cfg=float(args.cfg),
        tau=args.tau,
        extra={"sweep": args.sweep, "formula": "G_subject + beta*D + alpha*G_prompt"},
    )


def main():
    parser = add_args(argparse.ArgumentParser(description=__doc__))
    parser.add_argument("--sweep", choices=("alpha", "beta", "grid"), default="beta")
    parser.add_argument("--alphas", type=csv_floats, default=csv_floats("0,0.1,0.2,0.3,0.4"))
    parser.add_argument("--betas", type=csv_floats, default=csv_floats("0.6,0.7,0.8,0.9,1.0"))
    parser.add_argument("--fixed-alpha", type=float, default=0.0,
                        help="alpha used by the beta-only sweep")
    parser.add_argument("--fixed-beta", type=float, default=0.9,
                        help="beta used by the alpha-only sweep")
    parser.add_argument("--offset-scales", default="0,2-4")
    parser.add_argument("--loss-scales-end", type=int, default=4)
    args = parser.parse_args()

    alphas = args.alphas if args.sweep in ("alpha", "grid") else [args.fixed_alpha]
    betas = args.betas if args.sweep in ("beta", "grid") else [args.fixed_beta]
    if any(not 0.0 <= value <= 1.0 for value in alphas):
        parser.error("all alpha values must be in [0, 1]")
    if any(value < 0.0 for value in betas):
        parser.error("all beta values must be non-negative")

    validate_offsets(args)
    variants = [
        variant(f"{args.sweep}_a{alpha:g}_b{beta:g}", alpha, beta, args)
        for alpha in alphas
        for beta in betas
    ]
    print(
        f"[ablation] sweep={args.sweep} variants={len(variants)} "
        f"alphas={alphas} betas={betas} offset_scales={args.offset_scales}",
        flush=True,
    )
    run(args, variants)


if __name__ == "__main__":
    main()
