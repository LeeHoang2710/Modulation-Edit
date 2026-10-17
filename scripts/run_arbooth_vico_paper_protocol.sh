#!/usr/bin/env bash
# Reproduce ARBooth's released 16 concepts x 31 prompts x 8 samples protocol.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

ENV_NAME="${ENV_NAME:-arbooth-vico}"
RESULTS="${RESULTS:-benchmark_outputs_vico_arbooth}"
REPORT_DIR="${REPORT_DIR:-benchmark_reports_vico_arbooth}"
SEED="${SEED:-0}"
CFG="${CFG:-4}"
TAU="${TAU:-0.5}"

command -v conda >/dev/null 2>&1 || {
    echo "conda is required; run scripts/setup_vico_vastai.sh first." >&2
    exit 1
}
CONDA_BASE="$(conda info --base)"
# shellcheck disable=SC1091
source "${CONDA_BASE}/etc/profile.d/conda.sh"
PYTHON_BIN=(conda run --no-capture-output -n "$ENV_NAME" python)

"${PYTHON_BIN[@]}" benchmark/prepare_vico.py
"${PYTHON_BIN[@]}" benchmark/run_arbooth_vico_official.py \
    --output "$RESULTS" --seed "$SEED" --cfg "$CFG" --tau "$TAU"

# ARBooth's code scores the original unrendered prompt templates. Keep this
# strict setting to compare directly against the published Table 1 values.
"${PYTHON_BIN[@]}" benchmark/evaluate_vico_arbooth_protocol.py \
    --results "$RESULTS" --model-name arbooth_released \
    --report-dir "$REPORT_DIR" --expected-images-per-prompt 8 \
    --text-protocol arbooth-code

echo "[complete] ARBooth images: $RESULTS"
echo "[complete] ARBooth metrics: $REPORT_DIR/summary.csv"
