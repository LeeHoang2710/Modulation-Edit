#!/usr/bin/env python3
"""Combine the validated primary timing fields without changing their meaning."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    args = parser.parse_args()
    root = Path(args.root).resolve()
    rows: list[dict[str, object]] = []

    ours_path = root / "ours" / "timing_summary.csv"
    arbooth_path = root / "arbooth_93" / "timing.json"
    csd_path = root / "csd_var_93" / "timing.json"
    for path in (ours_path, arbooth_path, csd_path):
        if not path.is_file():
            raise SystemExit(f"Missing timing report: {path}")

    with ours_path.open(newline="") as handle:
        for item in csv.DictReader(handle):
            rows.append({
                "method": "Ours",
                "configuration": item["configuration"],
                "pure_optimization_seconds_per_subject": item["pure_optimization_seconds"],
                "pure_generation_seconds_per_image_mean": item["pure_generation_mean_seconds"],
                "pure_generation_seconds_per_image_std": item["pure_generation_std_seconds"],
                "images": item["inference_images"],
                "preparation_note": "learned offset; pure synchronized optimizer steps",
            })

    arbooth = json.loads(arbooth_path.read_text())
    rows.append({
        "method": "ARBooth",
        "configuration": "released checkpoints",
        "pure_optimization_seconds_per_subject": "",
        "pure_generation_seconds_per_image_mean": arbooth["pure_generation_seconds_per_image_mean"],
        "pure_generation_seconds_per_image_std": arbooth["pure_generation_seconds_per_image_std"],
        "images": arbooth["images"],
        "preparation_note": arbooth["preparation_note"],
    })

    csd = json.loads(csd_path.read_text())
    rows.append({
        "method": "CSD-VAR",
        "configuration": "released training protocol",
        "pure_optimization_seconds_per_subject": csd["preparation_mean_seconds_per_subject"],
        "pure_generation_seconds_per_image_mean": csd["pure_generation_seconds_per_image_mean"],
        "pure_generation_seconds_per_image_std": csd["pure_generation_seconds_per_image_std"],
        "images": csd["images"],
        "preparation_note": csd["preparation_timing_note"],
    })

    output = root / "timing_summary.csv"
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(output)


if __name__ == "__main__":
    main()
