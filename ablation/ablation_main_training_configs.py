"""Generate the main variant for the two faster offset-training configurations."""
import argparse

from common import Variant, add_args, run


if __name__ == "__main__":
    parser = add_args(argparse.ArgumentParser(description=__doc__))
    parser.add_argument("--offset-base", required=True)
    args = parser.parse_args()
    names = ("bs4_ga1_it200", "bs4_ga2_it100")
    variants = [
        Variant(
            name=name,
            # With base_text="subject", common.py replaces the modulation
            # baseline with G_subject.  alpha=1 on blend_scales therefore
            # yields G_subject + (G_prompt - G_subject) = G_prompt.
            alpha=1.0,
            blend="add",
            ca="prompt",
            offset="learned",
            offset_root=f"{args.offset_base}/{name}",
            offset_scales="0,2-5",
            offset_weight=1.0,
            base_text="subject",
            blend_scales="1,6-12",
            cfg=float(args.cfg),
            tau=args.tau,
            extra={
                "experiment": "main_variant_training_configuration",
                "optimized_scales": "0-5",
                "applied_scales": "0,2-5",
                "optimization_configuration": name,
                "formula": "G_subject + D_s on scales 0,2-5; G_prompt on scales 1,6-12",
            },
        )
        for name in names
    ]
    run(args, variants)
