#!/usr/bin/env bash
# Run fresh, non-overlapping timing outputs for all three methods.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
bash scripts/preflight_three_model_timing.sh

run_id="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
base="${OUTPUT_ROOT:-output/timing/$run_id}"
mkdir -p "$base"

echo "[1/3] Our method"
OUT="$base/ours" SUBJECT="${OURS_SUBJECT:-black_cat}" \
  bash ablation/08_training_speed_benchmark.sh

echo "[2/3] ARBooth"
OUTPUT="$base/arbooth_93" SUBJECTS="${SUBJECTS:-brown_dog,black_cat,alarm_clock}" \
  bash scripts/run_arbooth_inference_timing_93.sh

echo "[3/3] CSD-VAR"
WORK_DIR="$base/csd_var_93/work" OUTPUT="$base/csd_var_93/images" \
  SUBJECTS="${SUBJECTS:-brown_dog,black_cat,alarm_clock}" \
  bash scripts/run_csd_var_timing_93.sh

echo "[summary] Combining primary timing fields"
.venv/bin/python benchmark/summarize_three_model_timing.py --root "$base"
echo "[done] $base"
