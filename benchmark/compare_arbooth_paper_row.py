#!/usr/bin/env python3
"""Print an observed ARBooth-protocol summary against the paper's Table 1 row."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path


PAPER = {"i_dino": 0.705, "i_clip": 0.824, "t_clip": 0.253, "inference_seconds_per_image": 0.5}


def maybe_float(row: dict[str, str], field: str) -> float | None:
    value = row.get(field, "")
    return float(value) if value not in {"", "None", None} else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", required=True, help="summary.csv from evaluate_vico_arbooth_protocol.py")
    parser.add_argument("--label", default="ARBooth released checkpoints")
    args = parser.parse_args()
    with Path(args.summary).open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 1:
        raise SystemExit(f"Expected exactly one summary row, found {len(rows)}")
    observed = rows[0]
    print(f"{args.label}")
    print("metric                    paper       observed    observed-paper")
    for field in ("i_dino", "i_clip", "t_clip", "inference_seconds_per_image"):
        value = maybe_float(observed, field)
        if value is None:
            print(f"{field:26} {PAPER[field]:8.3f}    unavailable")
        else:
            print(f"{field:26} {PAPER[field]:8.3f}    {value:8.3f}    {value - PAPER[field]:+8.3f}")
    print("Note: time is hardware- and software-stack-dependent; the three similarity metrics are the primary reproduction check.")


if __name__ == "__main__":
    main()
