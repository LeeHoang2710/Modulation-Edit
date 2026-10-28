#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

VICO_PYTHON="${VICO_PYTHON:-/venv/vico-modulation/bin/python}"
SUBJECTS="${SUBJECTS:-all}"
MAX_PROMPTS="${MAX_PROMPTS:-0}"
NUM_SAMPLES="${NUM_SAMPLES:-8}"
LOAD_STEP="${LOAD_STEP:-399}"
OUTPUT="${OUTPUT:-benchmark_outputs_vico_official_six_overlap}"

"$VICO_PYTHON" benchmark/run_vico_official_six_overlap.py \
    --subjects "$SUBJECTS" --max-prompts "$MAX_PROMPTS" --num-samples "$NUM_SAMPLES" \
    --load-step "$LOAD_STEP" --output "$OUTPUT" \
    --code-root models/ViCo --checkpoint-root weights/vico \
    --base-checkpoint weights/vico/stable-diffusion-v1/sd-v1-4.ckpt

echo "[complete] Official ViCo overlap outputs: $OUTPUT"
