#!/usr/bin/env bash
# Create only the CSD-VAR environment; reuse an existing ViCo metrics env.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
ENV_NAME="${CSD_ENV_NAME:-benchmark-csdvar}"
CUDA_WHL="${CUDA_WHL:-cu121}"
command -v conda >/dev/null 2>&1 || { echo "Conda is required; install Miniconda first." >&2; exit 1; }
if [[ ! -f models/CSD-VAR/requirements.txt ]]; then
    git submodule update --init --recursive models/CSD-VAR
fi
[[ -f models/CSD-VAR/requirements.txt ]] || { echo "CSD-VAR source is missing." >&2; exit 1; }
if ! conda env list | awk '{print $1}' | grep -Fxq "$ENV_NAME"; then
    conda create --yes --name "$ENV_NAME" python=3.10 pip
fi
PY=(conda run --no-capture-output -n "$ENV_NAME" python)
"${PY[@]}" -m pip install --upgrade 'pip<24.1' setuptools wheel
"${PY[@]}" -m pip install 'torch==2.3.1' 'torchvision==0.18.1' 'torchaudio==2.3.1' --index-url "https://download.pytorch.org/whl/${CUDA_WHL}"
"${PY[@]}" -m pip install -r models/CSD-VAR/requirements.txt
"${PY[@]}" -m pip install 'numpy<2'
"${PY[@]}" -m pip check
"${PY[@]}" -c 'import torch, torchvision, clip, transformers, diffusers; assert torch.__version__.split("+")[0] == "2.3.1", "Dependencies changed the expected torch version; check the environment"; print("torch:", torch.__version__, "transformers:", transformers.__version__, "CUDA available:", torch.cuda.is_available())'
echo "[ready] $ENV_NAME. Set CSD_PYTHON to its python path if using a custom environment name."
