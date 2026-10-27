#!/usr/bin/env bash
# Eight independently optimized five-scale offset bundles, evaluated with the
# same inference seed for every bundle. This isolates optimization variation:
# output index 0..7 maps to offset-training seed 0..7, while g_seed is fixed.
# It is intentionally separate from the one-offset ARBooth comparison.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

ENV_NAME="${ENV_NAME:-vico-modulation}"
VICO_PYTHON="${VICO_PYTHON:-/venv/vico-modulation/bin/python}"
SUBJECTS="${SUBJECTS:-all}"
MAX_PROMPTS="${MAX_PROMPTS:-0}"
FIXED_INFERENCE_SEED="${FIXED_INFERENCE_SEED:-0}"
OFFSET_SEEDS="${OFFSET_SEEDS:-8}"
STEPS="${STEPS:-50}"
BATCH_SIZE="${BATCH_SIZE:-5}"
GRAD_ACCUM="${GRAD_ACCUM:-1}"
LR="${LR:-4e-3}"

# Default to the current five-scale linear prompt-modulation schedule.
ALPHA_SCHEDULE="${ALPHA_SCHEDULE:-0.25,0.20,0.15,0.10,0.05}"
EARLY_CFG="${EARLY_CFG:-1.2}"
LATE_CFG="${LATE_CFG:-2.0}"
METHOD_PREFIX="${METHOD_PREFIX:-ours_multi_offset${OFFSET_SEEDS}_fixed_seed}"
RESULTS="${RESULTS:-benchmark_outputs_vico_multi_offset8_seed${FIXED_INFERENCE_SEED}}"
WORK_DIR="${WORK_DIR:-benchmark_work/vico_multi_offset8_seed${FIXED_INFERENCE_SEED}}"
REPORT_DIR="${REPORT_DIR:-benchmark_reports_vico_multi_offset8_seed${FIXED_INFERENCE_SEED}}"
SELECTION="${SELECTION:-best-dino}"

if [[ -x "$VICO_PYTHON" ]]; then
    # Vast.ai images commonly provide this already-built virtual environment.
    # Prefer it so the benchmark does not require Conda merely to launch.
    PYTHON_BIN=("$VICO_PYTHON")
elif command -v conda >/dev/null 2>&1; then
    CONDA_BASE="$(conda info --base)"
    # shellcheck disable=SC1091
    source "${CONDA_BASE}/etc/profile.d/conda.sh"
    PYTHON_BIN=(conda run --no-capture-output -n "$ENV_NAME" python)
else
    echo "No usable Python environment found. Set VICO_PYTHON or install the $ENV_NAME Conda environment." >&2
    exit 1
fi

"${PYTHON_BIN[@]}" benchmark/prepare_vico.py
echo "[multi-offset] subjects=$SUBJECTS offsets=$OFFSET_SEEDS; fixed g_seed=$FIXED_INFERENCE_SEED; prompts=$MAX_PROMPTS"

"${PYTHON_BIN[@]}" benchmark/run_ours_vico_two_variant.py \
    --subjects "$SUBJECTS" --offset-seeds "$OFFSET_SEEDS" --samples-per-offset 1 \
    --inference-seed-policy fixed --fixed-inference-seed "$FIXED_INFERENCE_SEED" \
    --max-prompts "$MAX_PROMPTS" --steps "$STEPS" --batch-size "$BATCH_SIZE" \
    --grad-accum "$GRAD_ACCUM" --lr "$LR" \
    --alpha-schedule "$ALPHA_SCHEDULE" --early-cfg "$EARLY_CFG" --late-cfg "$LATE_CFG" \
    --variants prompt --method-prefix "$METHOD_PREFIX" \
    --output "$RESULTS" --work-dir "$WORK_DIR"

# This evaluator matches ARBooth's released DINO/CLIP protocol. Each image is
# produced from an independently optimized offset, not a stochastic re-sample.
"${PYTHON_BIN[@]}" benchmark/evaluate_vico_arbooth_protocol.py \
    --results "$RESULTS/${METHOD_PREFIX}_late_prompt" \
    --model-name "${METHOD_PREFIX}" --subjects "$SUBJECTS" --max-prompts "$MAX_PROMPTS" \
    --report-dir "$REPORT_DIR" --expected-images-per-prompt "$OFFSET_SEEDS" \
    --selection "$SELECTION" \
    --text-protocol arbooth-code

echo "[complete] Images: $RESULTS/${METHOD_PREFIX}_late_prompt"
echo "[complete] Metrics: $REPORT_DIR/summary.csv"
