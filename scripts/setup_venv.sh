#!/bin/bash
# Create .venv with torch 2.5.1 (as pinned in requirements.txt) for this machine: driver 550 / CUDA 12.4, Python 3.10.
# usage: bash scripts/setup_venv.sh
set -euo pipefail
cd "$(dirname "$0")/.."

export PIP_CONFIG_FILE=/dev/null   # the global pip config points at pypi.ngc.nvidia.com, which does not resolve here
FLASH_ATTN_WHL=https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.0.post2/flash_attn-2.7.0.post2+cu12torch2.5cxx11abiFALSE-cp310-cp310-linux_x86_64.whl

if [ ! -x .venv/bin/python ]; then
    # Vast.ai images do not consistently expose /usr/bin/python3.10. Prefer an
    # existing 3.10 interpreter; otherwise create a path-based Conda env.
    PY=""
    for candidate in python3.10 /usr/local/bin/python3.10 /usr/bin/python3.10; do
        if command -v "$candidate" >/dev/null 2>&1; then
            PY="$(command -v "$candidate")"
            break
        fi
    done

    if [ -n "$PY" ] && "$PY" -m venv .venv >/dev/null 2>&1; then
        echo "[setup] Created .venv with $PY"
    else
        CONDA_BIN=""
        for candidate in conda /opt/conda/bin/conda /root/miniconda3/bin/conda /opt/miniforge3/bin/conda; do
            if command -v "$candidate" >/dev/null 2>&1; then
                CONDA_BIN="$(command -v "$candidate")"
                break
            fi
        done
        if [ -z "$CONDA_BIN" ]; then
            installer="${TMPDIR:-/tmp}/miniconda-vico.sh"
            echo "[setup] Python 3.10 and Conda were not found; installing Miniconda"
            if command -v wget >/dev/null 2>&1; then
                wget -q -O "$installer" https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
            else
                curl -fsSL -o "$installer" https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
            fi
            bash "$installer" -b -p /root/miniconda3
            CONDA_BIN=/root/miniconda3/bin/conda
        fi
        rm -rf .venv
        "$CONDA_BIN" create -y -p "$PWD/.venv" python=3.10 pip
        echo "[setup] Created .venv with Conda Python 3.10"
    fi
fi

python_version="$(.venv/bin/python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
if [ "$python_version" != 3.10 ]; then
    echo "Existing .venv uses Python $python_version, but the FlashAttention wheel requires Python 3.10." >&2
    echo "Move or remove .venv, then rerun this script." >&2
    exit 1
fi
pip=(.venv/bin/python -m pip install -q)

"${pip[@]}" setuptools wheel   # triton (used by torch.compile) imports setuptools; a --without-pip venv lacks it
"${pip[@]}" torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
"${pip[@]}" "$FLASH_ATTN_WHL"   # prebuilt (torch2.5, cxx11abi=False like the PyPI torch wheel); `pip install flash_attn` would compile for ~1h
"${pip[@]}" -r requirements.txt huggingface_hub   # torch / flash_attn pins are already satisfied, so nothing is rebuilt
# opencv-python needs libGL, which this container lacks; the headless build provides the same cv2 module
.venv/bin/python -m pip uninstall -q -y opencv-python
"${pip[@]}" --force-reinstall --no-cache-dir opencv-python-headless==4.10.0.84

.venv/bin/python - <<'EOF'
import torch, flash_attn, cv2, timm, tqdm
assert torch.__version__.startswith('2.5.1'), torch.__version__
assert torch.cuda.is_available(), 'CUDA not available'
print(f'torch {torch.__version__} (cuda {torch.version.cuda}), flash_attn {flash_attn.__version__}, timm {timm.__version__}, cv2 {cv2.__version__}, tqdm {tqdm.__version__}: OK')
EOF
