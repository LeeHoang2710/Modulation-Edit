#!/usr/bin/env python3
"""Print the requested compact ViCo metric table from the evaluator CSV."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", required=True)
    args = parser.parse_args()
    path = Path(args.summary)
    if not path.is_file():
        raise SystemExit(f"Missing metric summary: {path}")
    rows = list(csv.DictReader(path.open()))
    print("\nViCo benchmark summary (higher is better except Time):")
    print(f"{'Method':34} {'I-DINO ↑':>10} {'I-CLIP ↑':>10} {'T-CLIP ↑':>10} {'Time/image ↓':>14}")
    for row in rows:
        print(
            f"{row['model']:34} {float(row['dino']):10.4f} {float(row['clip_i']):10.4f} "
            f"{float(row['clip_t']):10.4f} {float(row['inference_seconds_per_image']):14.3f}"
        )


if __name__ == "__main__":
    main()
