#!/usr/bin/env bash
# Idempotent Vast.ai preparation for our method, ARBooth and CSD-VAR timing.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

WORKSPACE="${WORKSPACE:-/workspace}"
export WORKSPACE

echo "[1/4] Preparing Infinity, weights, ViCo data and CSD-VAR"
bash scripts/prepare_vico_beta_alpha.sh

command -v conda >/dev/null 2>&1 || { echo "Conda should have been installed by the previous step." >&2; exit 1; }
CONDA_BASE="$(conda info --base)"
source "$CONDA_BASE/etc/profile.d/conda.sh"

echo "[2/4] Creating or repairing arbooth-vico"
if [ ! -x "$CONDA_BASE/envs/arbooth-vico/bin/python" ]; then
  conda create -y -n arbooth-vico python=3.10 pip
fi
AR_PY="$CONDA_BASE/envs/arbooth-vico/bin/python"
"$AR_PY" -m pip install --upgrade 'pip<24.1' setuptools wheel
"$AR_PY" -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124

ARBOOTH_ROOT="${ARBOOTH_ROOT:-$WORKSPACE/ARBooth}"
[ -f "$ARBOOTH_ROOT/requirements.txt" ] || { echo "Missing $ARBOOTH_ROOT/requirements.txt" >&2; exit 1; }
filtered="$(mktemp)"
trap 'rm -f "$filtered"' EXIT
sed -E \
  -e '/^[[:space:]]*flash[_-]attn([[:space:]]|$)/d' \
  -e '/^[[:space:]]*httpx([<=>].*)?$/d' \
  -e '/^[[:space:]]*openai([<=>].*)?$/d' \
  -e '/^[[:space:]]*transformers([<=>].*)?$/d' \
  "$ARBOOTH_ROOT/requirements.txt" > "$filtered"
"$AR_PY" -m pip install -r "$filtered"
"$AR_PY" -m pip install ninja packaging psutil 'transformers==4.46.3' 'numpy<2' opencv-python-headless
"$AR_PY" -m pip install 'https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.0.post2/flash_attn-2.7.0.post2+cu12torch2.5cxx11abiFALSE-cp310-cp310-linux_x86_64.whl'

echo "[3/4] Downloading the three ARBooth released checkpoints"
.venv/bin/python - <<'PY'
import os
from pathlib import Path
from huggingface_hub import hf_hub_download
root = Path("weights/arbooth/released")
for subject in ("brown_dog", "black_cat", "alarm_clock"):
    hf_hub_download(
        repo_id="wldn0202/ARBooth", filename=f"ckpts/{subject}.pth",
        local_dir=str(root), token=os.environ.get("HF_TOKEN") or None,
    )
    source = (root / "ckpts" / f"{subject}.pth").resolve()
    target_dir = Path("weights/arbooth/by_subject") / subject
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / f"{subject}.pth"
    if target.exists() or target.is_symlink():
        target.unlink()
    target.symlink_to(source)
PY

echo "[4/4] Running strict preflight"
bash scripts/preflight_three_model_timing.sh
