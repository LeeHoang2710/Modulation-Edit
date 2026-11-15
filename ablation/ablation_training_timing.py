"""Generate 100 matched images for each offset-training timing configuration."""
import argparse
from common import Variant, add_args, run

if __name__ == "__main__":
    parser = add_args(argparse.ArgumentParser())
    parser.add_argument("--offset-base", required=True)
    args = parser.parse_args()
    names = ("bs2_ga2_it200", "bs4_ga1_it200", "bs4_ga2_it100")
    variants = [Variant(
        name=name, ca="prompt", offset_root=f"{args.offset_base}/{name}",
        offset_scales="0,2-5", offset_weight=1.0, base_text="subject",
        alpha=0.0, blend="add", blend_scales="1,6-12",
        cfg=float(args.cfg), tau=args.tau,
        extra={"training_configuration": name},
    ) for name in names]
    run(args, variants)
