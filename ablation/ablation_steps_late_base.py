"""Evaluate the step-100 offset with G_subject or G_prompt on scales 1,6-12."""
import argparse

from common import Variant, add_args, run


if __name__ == "__main__":
    parser = add_args(argparse.ArgumentParser())
    parser.add_argument("--offset-base", required=True)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--offset-scales", default="0,2-5")
    parser.add_argument("--prompt-only-scales", default="1,6-12")
    args = parser.parse_args()

    shared = dict(
        ca="prompt",
        offset_root=f"{args.offset_base}/step100",
        offset_scales=args.offset_scales,
        offset_weight=args.beta,
        base_text="subject",
        cfg=float(args.cfg),
        tau=args.tau,
    )
    variants = [
        Variant(
            name="step100_late_gsubject",
            alpha=0.0,
            blend="add",
            blend_scales=args.prompt_only_scales,
            extra={"checkpoint_step": 100, "nonoffset_modulation": "G_subject"},
            **shared,
        ),
        Variant(
            name="step100_late_gprompt",
            alpha=1.0,
            blend="add",
            blend_scales=args.prompt_only_scales,
            extra={"checkpoint_step": 100, "nonoffset_modulation": "G_prompt"},
            **shared,
        ),
    ]
    run(args, variants)
