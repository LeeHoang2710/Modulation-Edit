#!/usr/bin/env bash
# Download ViCo references/prompts, frozen Infinity assets, all 16 released
# ARBooth subject checkpoints, and the exact ARBooth DINO/CLIP metric weights.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

ARBOOTH_ENV="${ARBOOTH_ENV:-arbooth-vico}"
MODULATION_ENV="${MODULATION_ENV:-vico-modulation}"
HF_TOKEN="${HF_TOKEN:-}"

[[ -n "$HF_TOKEN" ]] || { echo "Set HF_TOKEN to a Hugging Face read token before downloading." >&2; exit 1; }
if [[ ! -f models/ARBooth/tools/run_arbooth_for_eval.py ]]; then
    echo "[assets] Initializing the released ARBooth Git submodule..."
    git submodule update --init --recursive models/ARBooth
fi
[[ -f models/ARBooth/tools/run_arbooth_for_eval.py ]] || {
    echo "ARBooth source is missing. Run: git submodule update --init --recursive models/ARBooth" >&2
    exit 1
}

# Download ViCo plus frozen Infinity once.  The asset directory is shared by
# both isolated environments.
ENV_NAME="$MODULATION_ENV" HF_TOKEN="$HF_TOKEN" bash scripts/prepare_vico_assets.sh

# Download all released ARBooth checkpoints and materialize the per-subject
# folder layout expected by ARBooth's unmodified evaluator.
ENV_NAME="$ARBOOTH_ENV" HF_TOKEN="$HF_TOKEN" bash scripts/prepare_arbooth_vico_assets.sh

echo "Ready: data/vico, weights/infinity, weights/arbooth/by_subject, weights/metrics."
