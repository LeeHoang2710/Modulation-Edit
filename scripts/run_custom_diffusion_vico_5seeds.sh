#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
ENV_NAME="${PERSONALIZATION_ENV_NAME:-benchmark-personalization}"
PY=(conda run --no-capture-output -n "$ENV_NAME" python)
SUBJECTS="${SUBJECTS:-all}"
MAX_PROMPTS="${MAX_PROMPTS:-0}"
SEEDS="${SEEDS:-0,1,2,3,4}"
OUTPUT="${OUTPUT:-benchmark_outputs_vico_custom_diffusion_5seeds}"
REPORT_DIR="${REPORT_DIR:-benchmark_reports_vico_custom_diffusion_5seeds}"

"${PY[@]}" benchmark/prepare_vico.py
"${PY[@]}" benchmark/run_custom_diffusion_vico.py \
    --subjects "$SUBJECTS" --max-prompts "$MAX_PROMPTS" --seeds "$SEEDS" --output "$OUTPUT"
"${PY[@]}" benchmark/evaluate_vico_arbooth_protocol.py \
    --results "$OUTPUT" --report-dir "$REPORT_DIR" --model-name custom_diffusion \
    --subjects "$SUBJECTS" --max-prompts "$MAX_PROMPTS" \
    --expected-images-per-prompt 5 --text-protocol arbooth-code
echo "[complete] images=$OUTPUT metrics=$REPORT_DIR/summary.csv"
