#!/usr/bin/env bash
set -euo pipefail

# Run the original friend's AdaLN-offset optimizer on one subject folder from
# this repository's contents/ directory.
#
# Example:
#   bash scripts/run_contents_offset.sh dog
#   ITERS=400 LR=1e-2 bash scripts/run_contents_offset.sh cat

cd "$(dirname "${BASH_SOURCE[0]}")/.."

SUBJECT="${1:-dog}"
ENV_NAME="${INFINITY_CONDA_ENV:-infinity-modulation}"
PN="${PN:-1M}"
ITERS="${ITERS:-400}"
BATCH_SIZE="${BATCH_SIZE:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-1}"
LR="${LR:-5e-3}"
LOSS_SCALES="${LOSS_SCALES:-all}"
OFFSET_MODE="${OFFSET_MODE:-per_scale}"
NUM_SAMPLES="${NUM_SAMPLES:-4}"
SEED="${SEED:-0}"
OUT_DIR="${OUT_DIR:-output/ada_offset/${SUBJECT}_ls${LOSS_SCALES}}"

IMAGES="contents/${SUBJECT}"
MODEL_PATH="weights/infinity_2b_reg.pth"
VAE_PATH="weights/infinity_vae_d32reg.pth"

if [[ ! -d "${IMAGES}" ]]; then
    echo "Missing subject folder: ${IMAGES}" >&2
    exit 1
fi
if [[ ! -f "${MODEL_PATH}" || ! -f "${VAE_PATH}" ]]; then
    echo "Model weights are missing. Run: bash scripts/download_vastai_weights.sh" >&2
    exit 1
fi

if [[ "${USE_ACTIVE_PYTHON:-0}" == "1" ]]; then
    PYTHON_BIN=("${PYTHON_BIN:-python}")
else
    command -v conda >/dev/null 2>&1 || {
        echo "conda was not found. Run scripts/setup_vastai_env.sh first." >&2
        exit 1
    }
    CONDA_BASE="$(conda info --base)"
    # shellcheck disable=SC1091
    source "${CONDA_BASE}/etc/profile.d/conda.sh"
    PYTHON_BIN=(conda run --no-capture-output -n "${ENV_NAME}" python)
fi

mkdir -p "${OUT_DIR}"

"${PYTHON_BIN[@]}" tools/optimize_ada_offset.py \
    --mode optimize \
    --images "${IMAGES}" \
    --pn "${PN}" \
    --model_type infinity_2b \
    --model_path "${MODEL_PATH}" \
    --vae_type 32 \
    --vae_path "${VAE_PATH}" \
    --add_lvl_embeding_only_first_block 1 \
    --use_bit_label 1 \
    --rope2d_each_sa_layer 1 \
    --rope2d_normalized_by_hw 2 \
    --use_scale_schedule_embedding 0 \
    --apply_spatial_patchify 0 \
    --checkpoint_type torch \
    --bf16 1 \
    --text_channels 2048 \
    --seed "${SEED}" \
    --hflip 1 \
    --grad_checkpointing 1 \
    --offset_mode "${OFFSET_MODE}" \
    --iters "${ITERS}" \
    --batch_size "${BATCH_SIZE}" \
    --grad_accum "${GRAD_ACCUM}" \
    --lr "${LR}" \
    --loss_scales "${LOSS_SCALES}" \
    --offset_infer_scales "${LOSS_SCALES}" \
    --num_samples "${NUM_SAMPLES}" \
    --save_offset "${OUT_DIR}/offset.pth" \
    --save_file "${OUT_DIR}/samples_vs_targets.jpg" \
    2>&1 | tee "${OUT_DIR}/optimize.log"

echo "Completed: ${OUT_DIR}"
