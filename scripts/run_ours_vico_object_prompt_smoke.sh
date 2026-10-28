#!/usr/bin/env bash
# Smoke test the current method with one cat-statue offset and all 31 ViCo
# prompts. Prompt cross-attention is active at every scale; conditional shared
# AdaLN switches to G_prompt at the late scales through --variants prompt.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

SUBJECTS="${SUBJECTS:-cat_statue}"
MAX_PROMPTS="${MAX_PROMPTS:-0}"
OFFSET_SEEDS="${OFFSET_SEEDS:-1}"
FIXED_INFERENCE_SEED="${FIXED_INFERENCE_SEED:-0}"
STEPS="${STEPS:-50}"
BATCH_SIZE="${BATCH_SIZE:-5}"
GRAD_ACCUM="${GRAD_ACCUM:-1}"
LR="${LR:-4e-3}"
# Five entries map directly to learned scales 0, 1, 2, 3, and 4.
# Override at launch, e.g.:
#   equal alpha:      ALPHA_SCHEDULE='0.15,0.15,0.15,0.15,0.15' bash "$0"
#   linear increasing: ALPHA_SCHEDULE='0.05,0.10,0.15,0.20,0.25' bash "$0"
ALPHA_SCHEDULE="${ALPHA_SCHEDULE:-0.25,0.20,0.15,0.10,0.05}"
EARLY_CFG="${EARLY_CFG:-1.2}"
LATE_CFG="${LATE_CFG:-2.0}"
RESULTS="${RESULTS:-benchmark_outputs_vico_cat_statue_object_prompt_smoke}"
WORK_DIR="${WORK_DIR:-benchmark_work/vico_cat_statue_object_prompt_smoke}"
REPORT_DIR="${REPORT_DIR:-benchmark_reports_vico_cat_statue_object_prompt_smoke}"

echo "[object-prompt smoke] subject=$SUBJECTS offsets=$OFFSET_SEEDS prompts=$MAX_PROMPTS"
echo "[object-prompt smoke] alpha=$ALPHA_SCHEDULE early_cfg=$EARLY_CFG late_cfg=$LATE_CFG"

SUBJECTS="$SUBJECTS" \
MAX_PROMPTS="$MAX_PROMPTS" \
OFFSET_SEEDS="$OFFSET_SEEDS" \
FIXED_INFERENCE_SEED="$FIXED_INFERENCE_SEED" \
STEPS="$STEPS" BATCH_SIZE="$BATCH_SIZE" GRAD_ACCUM="$GRAD_ACCUM" LR="$LR" \
ALPHA_SCHEDULE="$ALPHA_SCHEDULE" EARLY_CFG="$EARLY_CFG" LATE_CFG="$LATE_CFG" \
METHOD_PREFIX="ours_object_prompt_smoke" \
RESULTS="$RESULTS" WORK_DIR="$WORK_DIR" REPORT_DIR="$REPORT_DIR" \
SELECTION=mean8 \
bash scripts/run_ours_vico_multi_offset_fixed_seed.sh
