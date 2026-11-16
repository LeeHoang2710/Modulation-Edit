#!/usr/bin/env python3
"""Combine DINO/CLIP quality with pure timing for the three training settings."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path


CONFIGS = ("bs2_ga2_it200", "bs4_ga1_it200", "bs4_ga2_it100")


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--subject", required=True)
    args = parser.parse_args()
    root = Path(args.experiment).resolve()

    metric_rows = []
    for configuration in CONFIGS:
        path = root / configuration / "metrics.csv"
        if not path.is_file():
            raise SystemExit(f"Missing metric report: {path}")
        candidates = read_rows(path)
        row = next((item for item in candidates if item["subject"] == args.subject), None)
        if row is None:
            raise SystemExit(f"Missing subject {args.subject!r} in {path}")
        metric_rows.append({
            "configuration": configuration,
            "subject": args.subject,
            "dino": row["dino"],
            "clip_i": row["clip_i"],
            "clip_t": row["clip_t"],
        })

    metrics_path = root / "metrics_summary.csv"
    with metrics_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metric_rows[0]))
        writer.writeheader()
        writer.writerows(metric_rows)

    timing_path = root / "timing_summary.csv"
    if not timing_path.is_file():
        raise SystemExit(f"Missing timing report: {timing_path}")
    timing = {row["configuration"]: row for row in read_rows(timing_path)}
    combined = []
    for metric in metric_rows:
        configuration = metric["configuration"]
        if configuration not in timing:
            raise SystemExit(f"Missing configuration {configuration!r} in {timing_path}")
        time_row = timing[configuration]
        combined.append({
            **metric,
            "pure_optimization_seconds": time_row["pure_optimization_seconds"],
            "pure_generation_mean_seconds": time_row["pure_generation_mean_seconds"],
            "pure_generation_std_seconds": time_row["pure_generation_std_seconds"],
        })

    combined_path = root / "timing_metrics_summary.csv"
    with combined_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(combined[0]))
        writer.writeheader()
        writer.writerows(combined)

    print("configuration,dino,clip_i,clip_t")
    for row in metric_rows:
        print(
            f"{row['configuration']},{float(row['dino']):.5f},"
            f"{float(row['clip_i']):.5f},{float(row['clip_t']):.5f}"
        )
    print(metrics_path)
    print(combined_path)


if __name__ == "__main__":
    main()
