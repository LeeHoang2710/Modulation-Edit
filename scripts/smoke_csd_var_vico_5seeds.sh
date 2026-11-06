#!/usr/bin/env bash
# One ViCo subject × two prompts × five independent seeds = 10 images.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

SUBJECT="${SUBJECT:-brown_dog}"
echo "[smoke:CSD-VAR] subject=$SUBJECT prompts=2 seeds=0,1,2,3,4 expected_images=10"

SUBJECTS="$SUBJECT" \
MAX_PROMPTS=2 \
INFERENCE_SEEDS=0,1,2,3,4 \
IMAGES_PER_PROMPT=5 \
RESULTS="${RESULTS:-benchmark_outputs_smoke_csd_var_${SUBJECT}}" \
WORK_DIR="${WORK_DIR:-benchmark_work/smoke_csd_var_${SUBJECT}}" \
REPORT_DIR="${REPORT_DIR:-benchmark_reports_smoke_csd_var_${SUBJECT}}" \
bash scripts/run_csd_var_vico.sh
