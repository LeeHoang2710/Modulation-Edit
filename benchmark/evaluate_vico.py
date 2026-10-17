#!/usr/bin/env python3
"""Evaluate normalized outputs against the 16 ARBooth/ViCo reference folders."""
from __future__ import annotations

import sys

from evaluate_dreambooth import main


if __name__ == "__main__":
    if "--dataset" not in sys.argv:
        sys.argv.extend(["--dataset", "models/ARBooth/inputs"])
    if "--results" not in sys.argv:
        sys.argv.extend(["--results", "benchmark_outputs_vico"])
    if "--report-dir" not in sys.argv:
        sys.argv.extend(["--report-dir", "benchmark_reports_vico"])
    main()
