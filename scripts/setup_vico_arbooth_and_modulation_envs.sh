#!/usr/bin/env bash
# Build exactly two isolated GPU environments for the ViCo comparison:
#   arbooth-vico      released ARBooth inference + ARBooth-protocol metrics
#   vico-modulation   our multi-offset modulation method + matching metrics
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

CUDA_WHL="${CUDA_WHL:-cu121}"
ARBOOTH_ENV="${ARBOOTH_ENV:-arbooth-vico}"
MODULATION_ENV="${MODULATION_ENV:-vico-modulation}"

command -v conda >/dev/null 2>&1 || { echo "Install Miniconda first." >&2; exit 1; }
if [[ ! -f models/ARBooth/requirements.txt ]]; then
    echo "[setup] Initializing the released ARBooth Git submodule..."
    git submodule update --init --recursive models/ARBooth
fi
[[ -f models/ARBooth/requirements.txt ]] || {
    echo "ARBooth source is still missing. Re-clone with --recurse-submodules or run:" >&2
    echo "  git submodule update --init --recursive models/ARBooth" >&2
    exit 1
}
command -v nvcc >/dev/null 2>&1 || {
    echo "Use a Vast.ai CUDA development image: nvcc is required to build FlashAttention." >&2
    exit 1
}
CONDA_BASE="$(conda info --base)"
# shellcheck disable=SC1091
source "${CONDA_BASE}/etc/profile.d/conda.sh"

create_env() {
    local env="$1"
    if ! conda env list | awk '{print $1}' | grep -Fxq "$env"; then
        conda create --yes --name "$env" python=3.10 pip
    fi
    conda run --no-capture-output -n "$env" python -m pip install --upgrade 'pip<24.1' setuptools wheel
    conda run --no-capture-output -n "$env" python -m pip install \
        "torch==2.5.1" "torchvision==0.20.1" --index-url "https://download.pytorch.org/whl/${CUDA_WHL}"
}

install_without_flash_attention() {
    local env="$1" requirements_file="$2" filtered
    filtered="$(mktemp)"
    sed -E '/^[[:space:]]*flash[_-]attn([[:space:]]|$)/d' "$requirements_file" > "$filtered"
    conda run --no-capture-output -n "$env" python -m pip install -r "$filtered"
    command rm -f "$filtered"
    conda run --no-capture-output -n "$env" python -m pip install ninja packaging psutil
    if ! conda run --no-capture-output -n "$env" python -c 'import flash_attn' >/dev/null 2>&1; then
        conda run --no-capture-output -n "$env" python -m pip install --no-build-isolation flash-attn
    fi
}

install_metrics() {
    local env="$1"
    conda run --no-capture-output -n "$env" python -m pip install pillow numpy pandas 'timm==0.9.6'
    if ! conda run --no-capture-output -n "$env" python -c 'import clip' >/dev/null 2>&1; then
        conda run --no-capture-output -n "$env" python -m pip install 'git+https://github.com/openai/CLIP.git@main#egg=clip'
    fi
}

create_env "$ARBOOTH_ENV"
install_without_flash_attention "$ARBOOTH_ENV" models/ARBooth/requirements.txt
install_metrics "$ARBOOTH_ENV"

create_env "$MODULATION_ENV"
install_without_flash_attention "$MODULATION_ENV" requirements.txt
install_metrics "$MODULATION_ENV"

echo "Ready: $ARBOOTH_ENV (released ARBooth) and $MODULATION_ENV (our strategy)."
