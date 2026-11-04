#!/usr/bin/env python3
"""Select the best beta from the overall rows of an ablation summary."""
import argparse
import csv
import json
from pathlib import Path


TIE_BREAKERS = ("clip_i", "clip_t_cls", "clip_t", "div")


def score(row, metric):
    value = row.get(metric, "")
    return float(value) if value not in ("", None) else float("-inf")


def decode(value):
    if value in ("", None):
        return None
    try:
        return float(value)
    except ValueError:
        return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--metric", default="dino")
    parser.add_argument("--output", required=True)
    parser.add_argument("--value-only", action="store_true")
    args = parser.parse_args()

    with Path(args.summary).open() as handle:
        rows = [row for row in csv.DictReader(handle) if row.get("group") == "overall"]
    if not rows:
        raise SystemExit(f"no overall rows in {args.summary}")
    if args.metric not in rows[0]:
        raise SystemExit(f"selection metric {args.metric!r} is not present in {args.summary}")
    usable = [row for row in rows if row.get(args.metric, "") != ""]
    if not usable:
        raise SystemExit(f"selection metric {args.metric!r} has no values")

    ordering = (args.metric,) + tuple(metric for metric in TIE_BREAKERS if metric != args.metric)
    best = max(usable, key=lambda row: tuple(score(row, metric) for metric in ordering))
    payload = {
        "selection_metric": args.metric,
        "tie_breakers": [metric for metric in ordering[1:]],
        "selected_beta": float(best["beta"]),
        "selected_variant": best["variant"],
        "overall_metrics": {
            key: decode(value)
            for key, value in best.items()
            if key not in {"variant", "group"}
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n")
    if args.value_only:
        print(f"{payload['selected_beta']:g}")
    else:
        print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
