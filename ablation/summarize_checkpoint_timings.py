"""Collect per-subject checkpoint preparation timing, excluding checkpoint I/O."""
import argparse
import csv
import json
from pathlib import Path
from statistics import mean, pstdev


parser = argparse.ArgumentParser()
parser.add_argument("--offset-base", required=True)
parser.add_argument("--output", required=True)
parser.add_argument("--steps", default="100", help="comma-separated checkpoint steps")
args = parser.parse_args()
steps = [int(value) for value in args.steps.split(",") if value.strip()]
if not steps:
    parser.error("--steps must contain at least one step")

base = Path(args.offset_base)
rows = []
final_step = max(steps)
for timing_path in sorted((base / f"step{final_step}").glob("*/timing.json")):
    payload = json.loads(timing_path.read_text())
    by_step = {int(item["iters"]): item for item in payload.get("checkpoints", [])}
    missing = [step for step in steps if step not in by_step]
    if missing:
        raise SystemExit(f"missing checkpoint timing {missing} in {timing_path}")
    row = {
        "subject": timing_path.parent.name,
        "checkpoint_save_seconds_total": payload["checkpoint_save_seconds"],
    }
    for step in steps:
        row[f"step{step}_preparation_seconds"] = by_step[step]["preparation_seconds_excluding_checkpoint_io"]
    rows.append(row)

if not rows:
    raise SystemExit(f"no timing reports under {base / f'step{final_step}'}")

subject_rows = list(rows)
for label, fn in (("[mean]", mean), ("[std]", pstdev)):
    rows.append({
        "subject": label,
        **{
            key: fn(float(row[key]) for row in subject_rows)
            for key in [f"step{step}_preparation_seconds" for step in steps]
            + ["checkpoint_save_seconds_total"]
        },
    })

output = Path(args.output)
output.parent.mkdir(parents=True, exist_ok=True)
with output.open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
print(f"[saved] {output}")
