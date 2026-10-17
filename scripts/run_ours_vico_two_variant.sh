#!/usr/bin/env bash
# Benchmark the latest modulation method on ARBooth's 16 x 31 ViCo protocol.
#
# The runner has two explicit phases:
#   1) fit OFFSET_SEEDS saved offsets for each requested subject;
#   2) use every saved offset for both late-scale variants (G_prompt/G_null).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

ENV_NAME="${ENV_NAME:-benchmark-arbooth}"
EVAL_ENV_NAME="${EVAL_ENV_NAME:-benchmark-eval}"
SUBJECTS="${SUBJECTS:-all}"
OFFSET_SEEDS="${OFFSET_SEEDS:-1}"
MAX_PROMPTS="${MAX_PROMPTS:-0}" # 0=all 31 official ViCo prompts; use 5 for a smoke test.
STEPS="${STEPS:-50}"
BATCH_SIZE="${BATCH_SIZE:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-1}"
LR="${LR:-5e-3}"
ALPHA_SCHEDULE="${ALPHA_SCHEDULE:-0.25,0.20,0.15,0.10,0.05}"
EARLY_CFG="${EARLY_CFG:-1.2}"
LATE_CFG="${LATE_CFG:-2.0}"
VARIANTS="${VARIANTS:-prompt,null}"
RESULTS="${RESULTS:-benchmark_outputs_vico}"
WORK_DIR="${WORK_DIR:-benchmark_work/vico_modulation}"
REPORT_DIR="${REPORT_DIR:-benchmark_reports_vico}"

command -v conda >/dev/null 2>&1 || { echo "conda is required; run scripts/setup_vico_vastai.sh first." >&2; exit 1; }
CONDA_BASE="$(conda info --base)"
# shellcheck disable=SC1091
source "${CONDA_BASE}/etc/profile.d/conda.sh"

TRAIN_PY=(conda run --no-capture-output -n "$ENV_NAME" python)
EVAL_PY=(conda run --no-capture-output -n "$EVAL_ENV_NAME" python)

"${TRAIN_PY[@]}" benchmark/prepare_vico.py
echo "[run] subjects=$SUBJECTS offset_seeds=$OFFSET_SEEDS prompts=$MAX_PROMPTS variants=$VARIANTS"
"${TRAIN_PY[@]}" benchmark/run_ours_vico_two_variant.py \
    --subjects "$SUBJECTS" --offset-seeds "$OFFSET_SEEDS" --max-prompts "$MAX_PROMPTS" \
    --steps "$STEPS" --batch-size "$BATCH_SIZE" --grad-accum "$GRAD_ACCUM" --lr "$LR" \
    --alpha-schedule "$ALPHA_SCHEDULE" --early-cfg "$EARLY_CFG" --late-cfg "$LATE_CFG" \
    --variants "$VARIANTS" --output "$RESULTS" --work-dir "$WORK_DIR"

MODELS=""
if [[ "$VARIANTS" == *prompt* ]]; then MODELS="modulation_offset_late_prompt"; fi
if [[ "$VARIANTS" == *null* ]]; then
    [[ -n "$MODELS" ]] && MODELS+=","
    MODELS+="modulation_offset_late_null"
fi
"${EVAL_PY[@]}" benchmark/evaluate_vico.py \
    --results "$RESULTS" --report-dir "$REPORT_DIR" --models "$MODELS"
"${EVAL_PY[@]}" benchmark/print_vico_summary.py --summary "$REPORT_DIR/summary.csv"

echo "[complete] Images: $RESULTS"
echo "[complete] Metrics: $REPORT_DIR/summary.csv"
