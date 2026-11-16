"""Generate the A-space-one scale-set comparison.

Each variant loads one shared offset per subject and injects that same offset
at the selected scales. The offset is learned only from the selected scales;
all other scales use the prompt modulation in the main-variant protocol.
"""
import argparse

from common import Variant, add_args, run


SCALE_SETS = (
    ("s023", "0,2,3"),
    ("s0123", "0,1,2,3"),
    ("s0234", "0,2,3,4"),
    ("s02345", "0,2,3,4,5"),
)


if __name__ == "__main__":
    parser = add_args(argparse.ArgumentParser(description=__doc__))
    parser.add_argument("--offset-base", required=True,
                        help="directory containing <variant>/<subject>/offset.pth")
    args = parser.parse_args()

    variants = [
        Variant(
            name=name,
            alpha=1.0,
            blend="add",
            ca="prompt",
            offset="learned",
            offset_root=f"{args.offset_base}/{name}",
            offset_scales=scales,
            offset_weight=1.0,
            base_text="subject",
            blend_scales="1,6-12",
            cfg=float(args.cfg),
            tau=args.tau,
            extra={
                "experiment": "A_space_one_shared_offset_scale_set",
                "scale_set": scales,
                "offset_mode": "shared",
                "optimized_scales": scales,
                "applied_scales": scales,
                "formula": "G_subject + D on selected scales; G_prompt on scales 1,6-12",
            },
        )
        for name, scales in SCALE_SETS
    ]
    run(args, variants)
