#!/usr/bin/env bash
# Downloads public benchmark assets; none are committed to Git.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
DATA_ROOT="${DATA_ROOT:-data/dreambooth}"
WEIGHTS_ROOT="${WEIGHTS_ROOT:-weights}"
HF_TOKEN="${HF_TOKEN:-}"
HF=(conda run --no-capture-output -n benchmark-eval huggingface-cli)

git submodule update --init --recursive

# Original Google DreamBooth benchmark: 30 subjects and the official prompt list.
if [[ ! -d "$DATA_ROOT/dataset" ]]; then
    mkdir -p "$DATA_ROOT"
    git clone --depth=1 https://github.com/google/dreambooth.git "$DATA_ROOT/source"
    if [[ -d "$DATA_ROOT/source/dataset" ]]; then
        rsync -a "$DATA_ROOT/source/dataset/" "$DATA_ROOT/dataset/"
    else
        echo "Official dataset directory was not found in the cloned source." >&2
        exit 1
    fi
fi

download() {
    local repository="$1" destination="$2"; shift 2
    mkdir -p "$destination"
    local token_args=()
    [[ -n "$HF_TOKEN" ]] && token_args=(--token "$HF_TOKEN")
    "${HF[@]}" download "$repository" "$@" --local-dir "$destination" "${token_args[@]}"
}

# Public backbone weights shared by ARBooth and this project.
download FoundationVision/Infinity "$WEIGHTS_ROOT/infinity" infinity_2b_reg.pth infinity_vae_d32reg.pth
download google/flan-t5-xl "$WEIGHTS_ROOT/infinity/flan-t5-xl"

# CSD-VAR Switti backbone + VQ-VAE; CSD checkpoint is optional for the requested
# DINO/CLIP evaluation but useful for CSD-VAR's own evaluator.
download yresearch/Switti-AR "$WEIGHTS_ROOT/switti/Switti-AR"
download yresearch/VQVAE-Switti "$WEIGHTS_ROOT/switti/VQVAE-Switti"
download tomg-group-umd/CSD-ViT-L "$WEIGHTS_ROOT/csd-vit-l" pytorch_model.bin

# ARBooth released examples are optional: they do not replace per-DreamBooth
# preparation. Set DOWNLOAD_RELEASED_ARBOOTH=1 only if you want them.
if [[ "${DOWNLOAD_RELEASED_ARBOOTH:-0}" == "1" ]]; then
    download wldn0202/ARBooth "$WEIGHTS_ROOT/arbooth-released"
fi

# These official source implementations require checkpoints covered by Hugging
# Face license gates. Accept their licenses then export HF_TOKEN before running.
if [[ -n "$HF_TOKEN" ]]; then
    download stabilityai/stable-diffusion-xl-base-1.0 "$WEIGHTS_ROOT/sdxl-base-1.0"
    echo "For original DreamBooth's SD-1.4 .ckpt and Textual Inversion's LDM .ckpt, place the accepted-license files at:"
else
    echo "HF_TOKEN is unset: SDXL was skipped. After accepting its Hugging Face license, rerun with HF_TOKEN=..."
fi
cat <<'EOF'
Manual licensed checkpoints required by the original legacy repositories:
  weights/stable-diffusion/sd-v1-4-full-ema.ckpt  (DreamBooth SD-v1.4)
  weights/latent-diffusion/model.ckpt             (Textual Inversion LDM)
EOF

python benchmark/prepare_dreambooth.py --dataset "$DATA_ROOT/dataset"
echo "Assets prepared under $DATA_ROOT and $WEIGHTS_ROOT"
