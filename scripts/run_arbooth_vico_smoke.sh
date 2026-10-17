#!/usr/bin/env bash
# One-subject/few-prompt ARBooth smoke test.  Released source remains intact.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
ENV_NAME="${ENV_NAME:-arbooth-vico}"
SUBJECT="${SUBJECT:-brown_dog}"
MAX_PROMPTS="${MAX_PROMPTS:-5}"
NUM_SAMPLES="${NUM_SAMPLES:-1}"
RESULTS="${RESULTS:-benchmark_outputs_vico_arbooth_smoke}"
REPORT_DIR="${REPORT_DIR:-benchmark_reports_vico_arbooth_smoke}"

CONDA_BASE="$(conda info --base)"
# shellcheck disable=SC1091
source "${CONDA_BASE}/etc/profile.d/conda.sh"
PYTHON_BIN=(conda run --no-capture-output -n "$ENV_NAME" python)

"${PYTHON_BIN[@]}" benchmark/prepare_vico.py
"${PYTHON_BIN[@]}" benchmark/run_arbooth_vico_smoke.py \
    --subject "$SUBJECT" --max-prompts "$MAX_PROMPTS" --num-samples "$NUM_SAMPLES" --output "$RESULTS"
"${PYTHON_BIN[@]}" benchmark/evaluate_vico_arbooth_protocol.py \
    --results "$RESULTS" --model-name arbooth_smoke --subjects "$SUBJECT" --max-prompts "$MAX_PROMPTS" \
    --expected-images-per-prompt "$NUM_SAMPLES" --report-dir "$REPORT_DIR" --text-protocol arbooth-code

echo "[complete] ARBooth smoke images: $RESULTS/$SUBJECT; metrics: $REPORT_DIR/summary.csv"
