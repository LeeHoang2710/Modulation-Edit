#!/usr/bin/env bash
set -euo pipefail

# Optimize null-conditioned AdaLN offsets on the first scales, then switch to
# a text prompt for the remaining scales during autoregressive inference.
# Example: scales 0-3 use cfg_uncond + offset, scales 4-12 use
# cfg_uncond + the prompt "is sleeping".

cd "$(dirname "${BASH_SOURCE[0]}")/.."

SUBJECT="${1:-dog}"
ENV_NAME="${INFINITY_CONDA_ENV:-infinity-modulation}"
PN="${PN:-1M}"
ITERS="${ITERS:-100}"
BATCH_SIZE="${BATCH_SIZE:-2}"
GRAD_ACCUM="${GRAD_ACCUM:-2}"
LR="${LR:-5e-3}"
LOSS_SCALES="${LOSS_SCALES:-0-3}"
HYBRID_START_SCALE="${HYBRID_START_SCALE:-4}"
HYBRID_ALL_SCALES="${HYBRID_ALL_SCALES:-0}"
HYBRID_PROMPT="${HYBRID_PROMPT:-is sleeping}"
HYBRID_CFG="${HYBRID_CFG:-3}"
OFFSET_MODE="${OFFSET_MODE:-per_scale}"
NUM_SAMPLES="${NUM_SAMPLES:-1}"
SEED="${SEED:-3}"
OUT_DIR="${OUT_DIR:-output/ada_offset_hybrid/${SUBJECT}_ls${LOSS_SCALES}}"

IMAGES="contents/${SUBJECT}"
MODEL_PATH="weights/infinity_2b_reg.pth"
VAE_PATH="weights/infinity_vae_d32reg.pth"
TEXT_ENCODER="weights/flan-t5-xl"

if [[ ! -d "${IMAGES}" ]]; then
    echo "Missing subject folder: ${IMAGES}" >&2
    exit 1
fi

if [[ ! -f "${MODEL_PATH}" || ! -f "${VAE_PATH}" || ! -d "${TEXT_ENCODER}" ]]; then
    echo "Model, VAE, or text-encoder weights are missing." >&2
    echo "Run: bash scripts/download_vastai_weights.sh" >&2
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

COMMON_ARGS=(
    --pn "${PN}"
    --model_type infinity_2b
    --model_path "${MODEL_PATH}"
    --vae_type 32
    --vae_path "${VAE_PATH}"
    --add_lvl_embeding_only_first_block 1
    --use_bit_label 1
    --rope2d_each_sa_layer 1
    --rope2d_normalized_by_hw 2
    --use_scale_schedule_embedding 0
    --apply_spatial_patchify 0
    --checkpoint_type torch
    --bf16 1
    --text_channels 2048
    --seed "${SEED}"
)

echo "Optimizing scales ${LOSS_SCALES} for ${SUBJECT}"
"${PYTHON_BIN[@]}" tools/optimize_ada_offset.py \
    --mode optimize \
    --images "${IMAGES}" \
    "${COMMON_ARGS[@]}" \
    --hflip 1 \
    --grad_checkpointing 1 \
    --offset_mode "${OFFSET_MODE}" \
    --iters "${ITERS}" \
    --batch_size "${BATCH_SIZE}" \
    --grad_accum "${GRAD_ACCUM}" \
    --lr "${LR}" \
    --loss_scales "${LOSS_SCALES}" \
    --offset_infer_scales "${LOSS_SCALES}" \
    --log_every 5 \
    --save_offset "${OUT_DIR}/offset.pth" \
    2>&1 | tee "${OUT_DIR}/optimize.log"

echo "Hybrid inference: first text-conditioned scale=${HYBRID_START_SCALE}, prompt=${HYBRID_PROMPT@Q}"
INFER_ARGS=(
    --mode infer
    "${COMMON_ARGS[@]}"
    --text_encoder_ckpt "${TEXT_ENCODER}"
    --offset_path "${OUT_DIR}/offset.pth"
    --offset_infer_scales "${LOSS_SCALES}"
    --hybrid_prompt "${HYBRID_PROMPT}"
    --hybrid_start_scale "${HYBRID_START_SCALE}"
    --hybrid_cfg "${HYBRID_CFG}"
    --num_samples "${NUM_SAMPLES}"
    --save_file "${OUT_DIR}/hybrid_reconstruction.jpg"
)
if [[ "${HYBRID_ALL_SCALES}" == "1" ]]; then
    INFER_ARGS+=(--hybrid_prompt_all_scales)
fi

"${PYTHON_BIN[@]}" tools/optimize_ada_offset.py "${INFER_ARGS[@]}" \
    2>&1 | tee "${OUT_DIR}/hybrid_infer.log"

echo "Completed: ${OUT_DIR}"
