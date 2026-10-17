#!/usr/bin/env bash
# Creates isolated environments because ARBooth/CSD-VAR, the original LDM
# baselines, and UnZipLoRA have incompatible upstream dependency ranges.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CUDA_WHL="${CUDA_WHL:-cu121}" # Use cu118 on older Vast.ai images: CUDA_WHL=cu118 bash "$0"
CONDA_BASE="$(conda info --base)"
# shellcheck disable=SC1091
source "${CONDA_BASE}/etc/profile.d/conda.sh"

create_env() {
    local name="$1" python="$2"
    if ! conda env list | awk '{print $1}' | grep -qx "${name}"; then
        conda create -y -n "${name}" "python=${python}" pip
    fi
}

pip_install() {
    local env="$1"; shift
    conda run --no-capture-output -n "${env}" python -m pip install --upgrade pip setuptools wheel
    conda run --no-capture-output -n "${env}" python -m pip install "$@"
}

cd "$ROOT"
git submodule update --init --recursive

# Shared evaluator: exact DreamBooth metric encoders (DINO ViT-S/16 and CLIP ViT-B/32).
create_env benchmark-eval 3.10
pip_install benchmark-eval "torch==2.4.1" "torchvision==0.19.1" --index-url "https://download.pytorch.org/whl/${CUDA_WHL}"
pip_install benchmark-eval "timm==1.0.11" "transformers==4.45.2" "safetensors" "pillow" "numpy"

# Infinity/ARBooth. flash-attn is optional: omit it if your GPU/CUDA cannot build it.
create_env benchmark-arbooth 3.10
pip_install benchmark-arbooth "torch==2.4.1" "torchvision==0.19.1" --index-url "https://download.pytorch.org/whl/${CUDA_WHL}"
pip_install benchmark-arbooth -r models/ARBooth/requirements.txt

# CSD-VAR official Switti implementation.
create_env benchmark-csdvar 3.10
pip_install benchmark-csdvar "torch==2.3.1" "torchvision==0.18.1" "torchaudio==2.3.1" --index-url "https://download.pytorch.org/whl/${CUDA_WHL}"
pip_install benchmark-csdvar -r models/CSD-VAR/requirements.txt

# The supplied DreamBooth and Textual-Inversion repositories are original LDM
# implementations and must remain on their legacy Python/PyTorch stack.
create_env benchmark-ldm 3.8
pip_install benchmark-ldm "torch==1.10.2+cu113" "torchvision==0.11.3+cu113" --extra-index-url https://download.pytorch.org/whl/cu113
pip_install benchmark-ldm "pytorch-lightning==1.5.9" "omegaconf==2.1.1" "transformers==4.18.0" "einops==0.4.1" "kornia==0.6" "albumentations==1.1.0" "opencv-python-headless==4.2.0.34" "imageio==2.14.1" "imageio-ffmpeg==0.4.7" "test-tube" "streamlit" "torchmetrics==0.6.0" "pillow==9.0.1"
pip_install benchmark-ldm "git+https://github.com/CompVis/taming-transformers.git@master#egg=taming-transformers" "git+https://github.com/openai/CLIP.git@main#egg=clip"

# UnZipLoRA's authors specify Python 3.11 and PyTorch 2.3.
create_env benchmark-unziplora 3.11
pip_install benchmark-unziplora "torch==2.3.0" "torchvision==0.18.0" "torchaudio==2.3.0" --index-url "https://download.pytorch.org/whl/${CUDA_WHL}"
pip_install benchmark-unziplora -r models/UnZipLoRA/requirements.txt "accelerate>=0.34" "bitsandbytes>=0.43"
conda run --no-capture-output -n benchmark-unziplora accelerate config default

echo "Ready. Environments: benchmark-eval, benchmark-arbooth, benchmark-csdvar, benchmark-ldm, benchmark-unziplora"
