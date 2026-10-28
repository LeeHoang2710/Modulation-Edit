#!/usr/bin/env bash
# One-subject/five-prompt smoke test of eight independently fitted offsets,
# all sampled with the one same g_seed.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
SUBJECTS="${SUBJECTS:-brown_dog}"
MAX_PROMPTS="${MAX_PROMPTS:-5}"
FIXED_INFERENCE_SEED="${FIXED_INFERENCE_SEED:-0}"
VICO_PYTHON="${VICO_PYTHON:-/venv/vico-modulation/bin/python}"
RESULTS="${RESULTS:-benchmark_outputs_vico_ours_multi_offset_smoke}"
WORK_DIR="${WORK_DIR:-benchmark_work/vico_ours_multi_offset_smoke}"
REPORT_DIR="${REPORT_DIR:-benchmark_reports_vico_ours_multi_offset_smoke}"

SUBJECTS="$SUBJECTS" MAX_PROMPTS="$MAX_PROMPTS" FIXED_INFERENCE_SEED="$FIXED_INFERENCE_SEED" \
VICO_PYTHON="$VICO_PYTHON" \
RESULTS="$RESULTS" WORK_DIR="$WORK_DIR" REPORT_DIR="$REPORT_DIR" \
bash scripts/run_ours_vico_multi_offset_fixed_seed.sh
