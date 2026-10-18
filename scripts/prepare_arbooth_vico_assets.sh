#!/usr/bin/env bash
# Download every released ARBooth checkpoint and the frozen assets needed to
# reproduce its published 16 x 31 x 8 ViCo evaluation locally.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

ENV_NAME="${ENV_NAME:-vico-modulation}"
ARBOOTH_REPO="${ARBOOTH_REPO:-wldn0202/ARBooth}"
ARBOOTH_DIR="${ARBOOTH_DIR:-weights/arbooth/released}"
PREFETCH_EVALUATORS="${PREFETCH_EVALUATORS:-1}"

command -v conda >/dev/null 2>&1 || {
    echo "conda is required; run scripts/setup_vico_vastai.sh first." >&2
    exit 1
}
CONDA_BASE="$(conda info --base)"
# shellcheck disable=SC1091
source "${CONDA_BASE}/etc/profile.d/conda.sh"
PYTHON_BIN=(conda run --no-capture-output -n "$ENV_NAME" python)

# This fetches the ViCo references/prompts, Infinity-2B, Infinity-VAE, and
# Flan-T5 exactly as the current modulation benchmark already expects.
ENV_NAME="$ENV_NAME" bash scripts/prepare_vico_assets.sh

mkdir -p "$ARBOOTH_DIR"
echo "[download] $ARBOOTH_REPO -> $ARBOOTH_DIR (16 released subject checkpoints; roughly 72 GB)"
"${PYTHON_BIN[@]}" scripts/download_hf_assets.py \
    --repo-id "$ARBOOTH_REPO" --local-dir "$ARBOOTH_DIR" --include 'ckpts/*.pth'

"${PYTHON_BIN[@]}" benchmark/prepare_arbooth_checkpoint_layout.py \
    --flat-dir "$ARBOOTH_DIR/ckpts" --output weights/arbooth/by_subject

if [[ "$PREFETCH_EVALUATORS" == "1" ]]; then
    echo "[prefetch] DINO ViT-S/16 and OpenAI CLIP ViT-B/32 metric weights"
    "${PYTHON_BIN[@]}" -c "
from pathlib import Path
import torch
import clip
root = Path('weights/metrics').resolve()
torch.hub.set_dir(str(root / 'torchhub'))
torch.hub.load('facebookresearch/dino:main', 'dino_vits16', trust_repo=True)
clip.load('ViT-B/32', device='cpu', download_root=str(root / 'clip'))
print('Metric weights are cached under', root)
"
fi

echo "[ready] ARBooth ViCo assets, 16 released checkpoints, and evaluator weights are available locally."
