"""Write per-image and per-subject inference timing CSVs for experiment variants."""
import argparse
import csv
import json
from pathlib import Path
from statistics import mean, pstdev


parser = argparse.ArgumentParser()
parser.add_argument("--experiment-dir", required=True)
parser.add_argument("--variants", nargs="+", required=True)
args = parser.parse_args()

root = Path(args.experiment_dir)
image_rows = []
for variant in args.variants:
    variant_dir = root / variant
    for timing_path in sorted(variant_dir.glob("*/timing.json")):
        payload = json.loads(timing_path.read_text())
        subject = payload.get("subject", timing_path.parent.name)
        for item in payload.get("images", []):
            image_rows.append({
                "variant": variant,
                "subject": subject,
                "prompt_index": item["prompt_index"],
                "image_index": item["image_index"],
                "seed": item["seed"],
                "inference_seconds": item["inference_seconds"],
                "batch_latency_seconds": item.get("batch_latency_seconds", item["inference_seconds"]),
                "batch_size": item.get("batch_size", 1),
            })

if not image_rows:
    raise SystemExit(f"no per-image timing records under {root}")

raw_path = root / "inference_times_per_image.csv"
with raw_path.open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(image_rows[0]))
    writer.writeheader()
    writer.writerows(image_rows)

groups = {}
for row in image_rows:
    groups.setdefault((row["variant"], row["subject"]), []).append(float(row["inference_seconds"]))

summary_rows = []
for (variant, subject), values in sorted(groups.items()):
    summary_rows.append({
        "variant": variant,
        "subject": subject,
        "images": len(values),
        "inference_seconds_per_image_mean": mean(values),
        "inference_seconds_per_image_std": pstdev(values),
    })
for variant in args.variants:
    values = [float(row["inference_seconds"]) for row in image_rows if row["variant"] == variant]
    if values:
        summary_rows.append({
            "variant": variant,
            "subject": "[overall]",
            "images": len(values),
            "inference_seconds_per_image_mean": mean(values),
            "inference_seconds_per_image_std": pstdev(values),
        })

summary_path = root / "inference_time_summary.csv"
with summary_path.open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
    writer.writeheader()
    writer.writerows(summary_rows)
print(f"[saved] {raw_path}")
print(f"[saved] {summary_path}")
