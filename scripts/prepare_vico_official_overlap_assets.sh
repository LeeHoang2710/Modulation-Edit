#!/usr/bin/env bash
set -euo pipefail

# Download the official ViCo implementation, Stable Diffusion v1-4 base model,
# and the released ViCo checkpoints.  The checkpoint folder is the public
# Google Drive folder linked by the official ViCo repository.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

VICO_CODE_DIR="${VICO_CODE_DIR:-models/ViCo}"
VICO_CHECKPOINT_DIR="${VICO_CHECKPOINT_DIR:-weights/vico}"
VICO_BASE_DIR="${VICO_BASE_DIR:-weights/vico/stable-diffusion-v1}"
VICO_PYTHON="${VICO_PYTHON:-/venv/vico-modulation/bin/python}"
VICO_REPO="${VICO_REPO:-https://github.com/haoosz/ViCo.git}"
CHECKPOINT_FOLDER_URL="${CHECKPOINT_FOLDER_URL:-https://drive.google.com/drive/folders/1I9BJpTLEGueK2hCaR2RKdQrlTrtF24lC?usp=drive_link}"

if [[ ! -x "$VICO_PYTHON" ]]; then
    VICO_PYTHON="${VICO_PYTHON_FALLBACK:-python}"
fi

if [[ ! -d "$VICO_CODE_DIR/.git" ]]; then
    mkdir -p "$(dirname "$VICO_CODE_DIR")"
    git clone --depth 1 "$VICO_REPO" "$VICO_CODE_DIR"
else
    echo "[skip] ViCo source already exists: $VICO_CODE_DIR"
fi

mkdir -p "$VICO_CHECKPOINT_DIR" "$VICO_BASE_DIR"

if ! command -v gdown >/dev/null 2>&1; then
    "$VICO_PYTHON" -m pip install gdown
fi

if [[ ! -f "$VICO_CHECKPOINT_DIR/.download_complete" ]]; then
    echo "[download] official ViCo checkpoints"
    gdown --folder "$CHECKPOINT_FOLDER_URL" -O "$VICO_CHECKPOINT_DIR"
    touch "$VICO_CHECKPOINT_DIR/.download_complete"
else
    echo "[skip] official ViCo checkpoints already downloaded"
fi

if [[ ! -f "$VICO_BASE_DIR/sd-v1-4.ckpt" ]]; then
    echo "[download] Stable Diffusion v1-4 base checkpoint"
    HF_TOKEN="${HF_TOKEN:-}" "$VICO_PYTHON" -c '
import os
from huggingface_hub import hf_hub_download

hf_hub_download(
    repo_id="CompVis/stable-diffusion-v-1-4-original",
    filename="sd-v1-4.ckpt",
    local_dir="weights/vico/stable-diffusion-v1",
    token=os.environ.get("HF_TOKEN") or None,
)
'
else
    echo "[skip] Stable Diffusion v1-4 already exists"
fi

if [[ ! -d data/vico ]]; then
    echo "[data] downloading the ViCo/ARBooth input references"
    bash scripts/prepare_vico_assets.sh
fi

"$VICO_PYTHON" benchmark/prepare_vico.py
echo "[ready] official ViCo source, base model, checkpoints, and references"
