#!/usr/bin/env python3
import argparse, csv, json, statistics
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--experiment", required=True)
p.add_argument("--subject", required=True)
args = p.parse_args()
root = Path(args.experiment)
rows = []
for name in ("bs2_ga2_it200", "bs4_ga1_it200", "bs4_ga2_it100"):
    prep = json.loads((root / "offsets" / name / args.subject / "timing.json").read_text())
    infer = json.loads((root / name / args.subject / "timing.json").read_text())
    values = [float(x["inference_seconds"]) for x in infer["images"]]
    rows.append({
        "configuration": name,
        "preparation_seconds": prep["preparation_seconds_excluding_checkpoint_io"],
        "inference_images": len(values),
        "inference_mean_seconds": statistics.mean(values),
        "inference_std_seconds": statistics.stdev(values),
    })
with (root / "timing_summary.csv").open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=rows[0]); w.writeheader(); w.writerows(rows)
for row in rows: print(row)
