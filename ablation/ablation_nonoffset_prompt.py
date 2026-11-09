"""Run prompt modulation on the scales that do not receive a subject offset.

The subject-conditioned offset is used on scales 0,2,3,4,5; prompt AdaLN
modulation is used on scales 1 and 6-12; prompt CA-KV is used at every scale.
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
    variants = [Variant(
        name="test_nonoffset_gprompt",
        alpha=1.0,
        blend="add",
        blend_scales=args.prompt_only_scales,
        extra={"nonoffset_modulation": "G_prompt"},
        **shared,
    )]
    run(args, variants)
