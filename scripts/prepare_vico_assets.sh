#!/usr/bin/env bash
# Download only the assets required by the current Modulation-Edit ViCo run:
# ViCo reference/prompt data and frozen Infinity/Flan-T5 weights.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

ENV_NAME="${ENV_NAME:-vico-modulation}"
VICO_PYTHON="${VICO_PYTHON:-/venv/vico-modulation/bin/python}"
WEIGHTS_ROOT="${WEIGHTS_ROOT:-weights}"
VICO_DIR="${VICO_DIR:-data/vico}"
if [[ -x "$VICO_PYTHON" ]]; then
    PYTHON_BIN=("$VICO_PYTHON")
elif command -v conda >/dev/null 2>&1; then
    CONDA_BASE="$(conda info --base)"
    # shellcheck disable=SC1091
    source "$CONDA_BASE/etc/profile.d/conda.sh"
    PYTHON_BIN=(conda run --no-capture-output -n "$ENV_NAME" python)
else
    echo "Missing $VICO_PYTHON or Conda environment $ENV_NAME" >&2
    exit 1
fi

if [[ ! -d "$VICO_DIR" ]]; then
    temp_dir="$(mktemp -d)"
    trap 'rm -rf "$temp_dir"' EXIT
    # Sparse checkout avoids downloading ARBooth's training/inference code:
    # only the public ViCo inputs directory is retained locally.
    git clone --depth=1 --filter=blob:none --sparse https://github.com/jiwoogit/ARBooth.git "$temp_dir/arbooth-source"
    git -C "$temp_dir/arbooth-source" sparse-checkout set inputs
    mkdir -p "$(dirname "$VICO_DIR")"
    mv "$temp_dir/arbooth-source/inputs" "$VICO_DIR"
fi
[[ -d "$VICO_DIR" ]] || { echo "Missing ViCo data: $VICO_DIR" >&2; exit 1; }

download() {
    local repository="$1" destination="$2"
    shift 2
    "${PYTHON_BIN[@]}" scripts/download_hf_assets.py \
        --repo-id "$repository" --local-dir "$destination" --include "$@"
}

download FoundationVision/Infinity "$WEIGHTS_ROOT/infinity" infinity_2b_reg.pth infinity_vae_d32reg.pth
download google/flan-t5-xl "$WEIGHTS_ROOT/infinity/flan-t5-xl"

"${PYTHON_BIN[@]}" benchmark/prepare_vico.py --inputs "$VICO_DIR"
echo "ViCo assets are ready."
