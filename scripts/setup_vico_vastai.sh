#!/usr/bin/env bash
# Minimal environments for the active ViCo Modulation-Edit benchmark only.
# Requires a CUDA development Vast.ai image: `nvcc --version` must work because
# ARBooth/Infinity imports FlashAttention.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CUDA_WHL="${CUDA_WHL:-cu121}"
cd "$ROOT"

command -v conda >/dev/null 2>&1 || { echo "Install Miniconda first." >&2; exit 1; }
command -v nvcc >/dev/null 2>&1 || {
    echo "ARBooth needs FlashAttention. Select a Vast.ai CUDA development image that provides nvcc." >&2
    exit 1
}
CONDA_BASE="$(conda info --base)"
# shellcheck disable=SC1091
source "${CONDA_BASE}/etc/profile.d/conda.sh"

create_env() {
    local name="$1"
    if ! conda env list | awk '{print $1}' | grep -qx "$name"; then
        conda create -y -n "$name" python=3.10 pip
    fi
}

pip_install() {
    local env="$1"; shift
    conda run --no-capture-output -n "$env" python -m pip install --upgrade pip setuptools wheel
    conda run --no-capture-output -n "$env" python -m pip install "$@"
}

install_arbooth_requirements() {
    # flash-attn must be built after torch. Its isolated build environment does
    # not expose torch, so install the other requirements first.
    local filtered
    filtered="$(mktemp)"
    sed -E '/^[[:space:]]*flash[_-]attn([[:space:]]|$)/d' models/ARBooth/requirements.txt > "$filtered"
    conda run --no-capture-output -n benchmark-arbooth python -m pip install -r "$filtered"
    command rm -f "$filtered"
    conda run --no-capture-output -n benchmark-arbooth python -m pip install ninja packaging psutil
    if ! conda run --no-capture-output -n benchmark-arbooth python -c 'import flash_attn' >/dev/null 2>&1; then
        conda run --no-capture-output -n benchmark-arbooth python -m pip install --no-build-isolation flash-attn
    fi
}

git submodule update --init --recursive

# Training/inference: PyTorch >=2.5 is required by ARBooth's FlexAttention import.
create_env benchmark-arbooth
pip_install benchmark-arbooth "torch==2.5.1" "torchvision==0.20.1" --index-url "https://download.pytorch.org/whl/${CUDA_WHL}"
install_arbooth_requirements

# Evaluation: DINO ViT-S/16 and CLIP ViT-B/32.
create_env benchmark-eval
pip_install benchmark-eval "torch==2.5.1" "torchvision==0.20.1" --index-url "https://download.pytorch.org/whl/${CUDA_WHL}"
pip_install benchmark-eval "timm==1.0.11" "transformers==4.45.2" safetensors pillow numpy

echo "Ready: benchmark-arbooth (offset fitting/inference), benchmark-eval (DINO/CLIP metrics)."
