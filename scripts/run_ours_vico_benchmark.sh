#!/usr/bin/env bash
# Fit one modulation offset per ViCo subject and generate all 31 official prompts.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

ENV_NAME="${ENV_NAME:-benchmark-arbooth}"
SUBJECTS="${SUBJECTS:-all}"
STEPS="${STEPS:-50}"
BATCH_SIZE="${BATCH_SIZE:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-1}"
LR="${LR:-5e-3}"
LOSS_SCALES="${LOSS_SCALES:-all}"
OFFSET_INFER_SCALES="${OFFSET_INFER_SCALES:-all}"
IMAGES_PER_PROMPT="${IMAGES_PER_PROMPT:-8}"
SEED="${SEED:-0}"
DELTA="${DELTA:-0.1}"
CFG="${CFG:-1.0}"
RESULTS="${RESULTS:-benchmark_outputs_vico}"
WORK_DIR="${WORK_DIR:-benchmark_work/vico_modulation}"

python_bin=(conda run --no-capture-output -n "$ENV_NAME" python)
"${python_bin[@]}" benchmark/prepare_vico.py
"${python_bin[@]}" benchmark/run_ours_vico.py \
    --subjects "$SUBJECTS" --steps "$STEPS" --batch-size "$BATCH_SIZE" --grad-accum "$GRAD_ACCUM" --lr "$LR" \
    --loss-scales "$LOSS_SCALES" --offset-infer-scales "$OFFSET_INFER_SCALES" \
    --images-per-prompt "$IMAGES_PER_PROMPT" --seed "$SEED" --delta "$DELTA" --cfg "$CFG" \
    --output "$RESULTS" --work-dir "$WORK_DIR"
"${python_bin[@]}" benchmark/evaluate_vico.py --results "$RESULTS" --models modulation_offset

echo "Completed ViCo modulation benchmark: $RESULTS"
