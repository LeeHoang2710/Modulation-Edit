#!/usr/bin/env python3
"""Evaluate our normalized outputs against the 16 ViCo reference folders."""
from __future__ import annotations

import sys

from evaluate_vico_metrics import main


if __name__ == "__main__":
    if "--dataset" not in sys.argv:
        sys.argv.extend(["--dataset", "data/vico"])
    if "--results" not in sys.argv:
        sys.argv.extend(["--results", "benchmark_outputs_vico"])
    if "--report-dir" not in sys.argv:
        sys.argv.extend(["--report-dir", "benchmark_reports_vico"])
    main()
