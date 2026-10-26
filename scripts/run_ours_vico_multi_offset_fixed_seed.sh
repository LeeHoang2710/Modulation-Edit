#!/usr/bin/env bash
# Eight independently optimized five-scale offset bundles, evaluated with the
# same inference seed for every bundle. This isolates optimization variation:
# output index 0..7 maps to offset-training seed 0..7, while g_seed is fixed.
# It is intentionally separate from the one-offset ARBooth comparison.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

ENV_NAME="${ENV_NAME:-vico-modulation}"
SUBJECTS="${SUBJECTS:-all}"
MAX_PROMPTS="${MAX_PROMPTS:-0}"
FIXED_INFERENCE_SEED="${FIXED_INFERENCE_SEED:-0}"
STEPS="${STEPS:-50}"
BATCH_SIZE="${BATCH_SIZE:-5}"
GRAD_ACCUM="${GRAD_ACCUM:-1}"
LR="${LR:-4e-3}"

# Match the requested fixed-alpha G_prompt formulation.
ALPHA_SCHEDULE="${ALPHA_SCHEDULE:-0.10,0.10,0.10,0.10,0.10}"
EARLY_CFG="${EARLY_CFG:-1.0}"
LATE_CFG="${LATE_CFG:-2.0}"
METHOD_PREFIX="${METHOD_PREFIX:-ours_multi_offset8_fixed_seed}"
RESULTS="${RESULTS:-benchmark_outputs_vico_multi_offset8_seed${FIXED_INFERENCE_SEED}}"
WORK_DIR="${WORK_DIR:-benchmark_work/vico_multi_offset8_seed${FIXED_INFERENCE_SEED}}"
REPORT_DIR="${REPORT_DIR:-benchmark_reports_vico_multi_offset8_seed${FIXED_INFERENCE_SEED}}"
SELECTION="${SELECTION:-best-dino}"

command -v conda >/dev/null 2>&1 || {
    echo "conda is required; run scripts/setup_vico_vastai.sh first." >&2
    exit 1
}
CONDA_BASE="$(conda info --base)"
# shellcheck disable=SC1091
source "${CONDA_BASE}/etc/profile.d/conda.sh"
PYTHON_BIN=(conda run --no-capture-output -n "$ENV_NAME" python)

"${PYTHON_BIN[@]}" benchmark/prepare_vico.py
echo "[multi-offset] subjects=$SUBJECTS offsets=8; fixed g_seed=$FIXED_INFERENCE_SEED; prompts=$MAX_PROMPTS"

"${PYTHON_BIN[@]}" benchmark/run_ours_vico_two_variant.py \
    --subjects "$SUBJECTS" --offset-seeds 8 --samples-per-offset 1 \
    --inference-seed-policy fixed --fixed-inference-seed "$FIXED_INFERENCE_SEED" \
    --max-prompts "$MAX_PROMPTS" --steps "$STEPS" --batch-size "$BATCH_SIZE" \
    --grad-accum "$GRAD_ACCUM" --lr "$LR" \
    --alpha-schedule "$ALPHA_SCHEDULE" --early-cfg "$EARLY_CFG" --late-cfg "$LATE_CFG" \
    --variants prompt --method-prefix "$METHOD_PREFIX" \
    --output "$RESULTS" --work-dir "$WORK_DIR"

# This evaluator matches ARBooth's released DINO/CLIP protocol. The eight
# files per prompt are eight optimization runs, not eight stochastic samples.
"${PYTHON_BIN[@]}" benchmark/evaluate_vico_arbooth_protocol.py \
    --results "$RESULTS/${METHOD_PREFIX}_late_prompt" \
    --model-name "${METHOD_PREFIX}" --subjects "$SUBJECTS" --max-prompts "$MAX_PROMPTS" \
    --report-dir "$REPORT_DIR" --expected-images-per-prompt 8 \
    --selection "$SELECTION" \
    --text-protocol arbooth-code

echo "[complete] Images: $RESULTS/${METHOD_PREFIX}_late_prompt"
echo "[complete] Metrics: $REPORT_DIR/summary.csv"
