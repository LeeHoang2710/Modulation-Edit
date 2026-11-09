"""Collect per-subject step-50/100 preparation timing, excluding checkpoint I/O."""
import argparse
import csv
import json
from pathlib import Path
from statistics import mean, pstdev


parser = argparse.ArgumentParser()
parser.add_argument("--offset-base", required=True)
parser.add_argument("--output", required=True)
args = parser.parse_args()

base = Path(args.offset_base)
rows = []
for timing_path in sorted((base / "step100").glob("*/timing.json")):
    payload = json.loads(timing_path.read_text())
    by_step = {int(item["iters"]): item for item in payload.get("checkpoints", [])}
    if 50 not in by_step or 100 not in by_step:
        raise SystemExit(f"missing step-50/100 timing in {timing_path}")
    rows.append({
        "subject": timing_path.parent.name,
        "step50_preparation_seconds": by_step[50]["preparation_seconds_excluding_checkpoint_io"],
        "step100_preparation_seconds": by_step[100]["preparation_seconds_excluding_checkpoint_io"],
        "checkpoint_save_seconds_total": payload["checkpoint_save_seconds"],
    })

if not rows:
    raise SystemExit(f"no timing reports under {base / 'step100'}")

subject_rows = list(rows)
for label, fn in (("[mean]", mean), ("[std]", pstdev)):
    rows.append({
        "subject": label,
        **{
            key: fn(float(row[key]) for row in subject_rows)
            for key in (
                "step50_preparation_seconds",
                "step100_preparation_seconds",
                "checkpoint_save_seconds_total",
            )
        },
    })

output = Path(args.output)
output.parent.mkdir(parents=True, exist_ok=True)
with output.open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
print(f"[saved] {output}")
