#!/usr/bin/env bash
# Fetch ViCo and the CSD-VAR/metrics models without downloading Infinity or ARBooth.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
CSD_PYTHON="${CSD_PYTHON:-}"
if [[ -z "$CSD_PYTHON" && -z "${CSD_ENV_NAME:-}" && -x /venv/benchmark-csdvar/bin/python ]]; then CSD_PYTHON=/venv/benchmark-csdvar/bin/python; fi
EVAL_PYTHON="${EVAL_PYTHON:-}"
if [[ -n "$CSD_PYTHON" && -x "$CSD_PYTHON" ]]; then CSD_CMD=("$CSD_PYTHON");
elif [[ -n "$CSD_PYTHON" ]]; then echo "CSD_PYTHON is not executable: $CSD_PYTHON" >&2; exit 1;
elif command -v conda >/dev/null 2>&1; then CSD_CMD=(conda run --no-capture-output -n "${CSD_ENV_NAME:-benchmark-csdvar}" python);
else echo "Set CSD_PYTHON or install the CSD-VAR environment." >&2; exit 1; fi
if [[ -z "$EVAL_PYTHON" && -z "${EVAL_ENV_NAME:-}" ]]; then EVAL_CMD=("${CSD_CMD[@]}");
elif [[ -n "$EVAL_PYTHON" && -x "$EVAL_PYTHON" ]]; then EVAL_CMD=("$EVAL_PYTHON");
elif [[ -n "${EVAL_ENV_NAME:-}" ]] && command -v conda >/dev/null 2>&1; then EVAL_CMD=(conda run --no-capture-output -n "$EVAL_ENV_NAME" python);
else echo "Set EVAL_PYTHON to the existing ViCo evaluator environment." >&2; exit 1; fi

if [[ ! -d data/vico ]]; then
    tmp_dir="$(mktemp -d)"
    trap 'rm -rf "$tmp_dir"' EXIT
    git clone --depth=1 --filter=blob:none --sparse https://github.com/jiwoogit/ARBooth.git "$tmp_dir/arbooth"
    git -C "$tmp_dir/arbooth" sparse-checkout set inputs
    mkdir -p data
    mv "$tmp_dir/arbooth/inputs" data/vico
fi
"${CSD_CMD[@]}" benchmark/prepare_vico.py
for repo in yresearch/Switti-AR yresearch/VQVAE-Switti openai/clip-vit-large-patch14 laion/CLIP-ViT-bigG-14-laion2B-39B-b160k; do
    echo "[download] $repo into Hugging Face cache"
    "${CSD_CMD[@]}" -c 'import sys; from huggingface_hub import snapshot_download; snapshot_download(repo_id=sys.argv[1])' "$repo"
done
echo "[download] ViCo evaluation DINO and CLIP"
"${EVAL_CMD[@]}" -c 'from pathlib import Path; import torch, clip; root=Path("weights/metrics").resolve(); torch.hub.set_dir(str(root / "torchhub")); torch.hub.load("facebookresearch/dino:main", "dino_vits16", trust_repo=True); clip.load("ViT-B/32", device="cpu", download_root=str(root / "clip"))'
echo "[ready] ViCo data, CSD-VAR weights, and metric weights are cached."
