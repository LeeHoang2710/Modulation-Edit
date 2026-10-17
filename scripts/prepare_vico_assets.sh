#!/usr/bin/env bash
# Prepare assets shared by Modulation-Edit and ARBooth on the ViCo protocol.
# ViCo references/prompts are versioned in models/ARBooth/inputs; released
# per-subject checkpoints remain external and are deliberately not committed.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

ENV_NAME="${ENV_NAME:-benchmark-arbooth}"
WEIGHTS_ROOT="${WEIGHTS_ROOT:-weights}"
HF_TOKEN="${HF_TOKEN:-}"
hf=(conda run --no-capture-output -n "$ENV_NAME" huggingface-cli)

git submodule update --init --recursive
[[ -d models/ARBooth/inputs ]] || { echo "Missing models/ARBooth/inputs." >&2; exit 1; }

download() {
    local repository="$1" destination="$2"
    shift 2
    local token_args=()
    [[ -n "$HF_TOKEN" ]] && token_args=(--token "$HF_TOKEN")
    mkdir -p "$destination"
    "${hf[@]}" download "$repository" "$@" --local-dir "$destination" "${token_args[@]}"
}

download FoundationVision/Infinity "$WEIGHTS_ROOT/infinity" infinity_2b_reg.pth infinity_vae_d32reg.pth
download google/flan-t5-xl "$WEIGHTS_ROOT/infinity/flan-t5-xl"

if [[ "${DOWNLOAD_RELEASED_ARBOOTH:-0}" == "1" ]]; then
    download wldn0202/ARBooth "$WEIGHTS_ROOT/arbooth-released"
else
    echo "ARBooth subject checkpoints skipped. Re-run with DOWNLOAD_RELEASED_ARBOOTH=1 to fetch them."
fi

conda run --no-capture-output -n "$ENV_NAME" python benchmark/prepare_vico.py
echo "ViCo assets are ready."
