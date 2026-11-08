#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

ENV_NAME="${PERSONALIZATION_ENV_NAME:-benchmark-personalization}"
if [[ -x "/venv/${ENV_NAME}/bin/python" ]]; then
    PYTHON_BIN="/venv/${ENV_NAME}/bin/python"
elif command -v conda >/dev/null 2>&1; then
    PYTHON_BIN="$(conda info --base)/envs/${ENV_NAME}/bin/python"
else
    echo "Cannot find Python for ${ENV_NAME}. Run scripts/setup_csd_custom_dreammatcher_vico.sh first." >&2
    exit 1
fi

SEEDS="${SEEDS:-0,1,2,3,4}"
STEPS="${STEPS:-50}"
GUIDANCE_SCALE="${GUIDANCE_SCALE:-10.0}"
SELECTION="${SELECTION:-benchmark/configs/selected_custom_dreammatcher_prompts.cfg}"
CUSTOM_OUTPUT="${CUSTOM_OUTPUT:-benchmark_outputs_selected_custom_diffusion_5seeds}"
DREAMMATCHER_OUTPUT="${DREAMMATCHER_OUTPUT:-benchmark_outputs_selected_dreammatcher_5seeds}"

SUBJECTS="black_cat,fat_dog,wooden_barn,monster_toy,pink_sunglasses"

"$PYTHON_BIN" benchmark/prepare_vico.py

echo "[1/2] Custom Diffusion: wooden_barn prompts 11,21; five seeds"
"$PYTHON_BIN" benchmark/run_custom_diffusion_vico.py \
    --subjects wooden_barn \
    --prompt-selection "$SELECTION" \
    --seeds "$SEEDS" \
    --steps "$STEPS" \
    --guidance-scale "$GUIDANCE_SCALE" \
    --output "$CUSTOM_OUTPUT"

echo "[2/2] DreamMatcher: 5 subjects × 2 prompts × 5 seeds"
"$PYTHON_BIN" benchmark/run_dreammatcher_custom_diffusion_vico.py \
    --subjects "$SUBJECTS" \
    --prompt-selection "$SELECTION" \
    --seeds "$SEEDS" \
    --steps "$STEPS" \
    --guidance-scale "$GUIDANCE_SCALE" \
    --output "$DREAMMATCHER_OUTPUT"

echo "[complete] Custom Diffusion images: $CUSTOM_OUTPUT"
echo "[complete] DreamMatcher images: $DREAMMATCHER_OUTPUT"
