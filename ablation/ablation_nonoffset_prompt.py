"""Compare the main subject-base modulation with prompt modulation on non-offset scales.

Both variants use the same subject-conditioned offset on scales 0,2,3,4,5 and
prompt CA-KV at every scale.  The only changed variable is the AdaLN modulation
used at scales 1 and 6-12:

  main:           G_subject
  nonoffset_prompt: G_prompt
"""
import argparse

from common import Variant, add_args, run


if __name__ == "__main__":
    parser = add_args(argparse.ArgumentParser())
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--offset-scales", default="0,2-5")
    parser.add_argument("--prompt-only-scales", default="1,6-12")
    args = parser.parse_args()

    shared = dict(
        ca="prompt",
        offset_root=args.offset_root,
        offset_scales=args.offset_scales,
        offset_weight=args.beta,
        base_text="subject",
        cfg=float(args.cfg),
        tau=args.tau,
    )
    variants = [
        Variant(
            name="main_nonoffset_gsubject",
            alpha=0.0,
            blend="add",
            blend_scales=args.prompt_only_scales,
            extra={"nonoffset_modulation": "G_subject"},
            **shared,
        ),
        Variant(
            name="test_nonoffset_gprompt",
            alpha=1.0,
            blend="add",
            blend_scales=args.prompt_only_scales,
            extra={"nonoffset_modulation": "G_prompt"},
            **shared,
        ),
    ]
    run(args, variants)
