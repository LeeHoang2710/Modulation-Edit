#!/usr/bin/env bash
set -euo pipefail

# Inference branch 1: replace the null ca_kv with the new prompt at every
# scale, while applying the learned offset only on --offset_infer_scales.

cd "$(dirname "${BASH_SOURCE[0]}")/.."

ENV_NAME="${INFINITY_CONDA_ENV:-infinity-modulation}"
PN="${PN:-1M}"
OFFSET_PATH="${OFFSET_PATH:-output/ada_offset_hybrid/dog_ls0-3/offset.pth}"
OFFSET_SCALES="${OFFSET_SCALES:-0-3}"
HYBRID_PROMPT="${HYBRID_PROMPT:-is sleeping}"
HYBRID_CFG="${HYBRID_CFG:-3}"
NUM_SAMPLES="${NUM_SAMPLES:-1}"
SEED="${SEED:-3}"
OUT_DIR="${OUT_DIR:-output/ada_offset_hybrid/dog_ca_kv_full}"

MODEL_PATH="weights/infinity_2b_reg.pth"
VAE_PATH="weights/infinity_vae_d32reg.pth"
TEXT_ENCODER="weights/flan-t5-xl"

[[ -f "${OFFSET_PATH}" ]] || { echo "Missing learned offset: ${OFFSET_PATH}" >&2; exit 1; }
[[ -f "${MODEL_PATH}" && -f "${VAE_PATH}" && -d "${TEXT_ENCODER}" ]] || {
    echo "Model, VAE, or text-encoder weights are missing." >&2
    exit 1
}

if [[ "${USE_ACTIVE_PYTHON:-0}" == "1" ]]; then
    PYTHON_BIN=("${PYTHON_BIN:-python}")
else
    command -v conda >/dev/null 2>&1 || { echo "conda was not found." >&2; exit 1; }
    CONDA_BASE="$(conda info --base)"
    # shellcheck disable=SC1091
    source "${CONDA_BASE}/etc/profile.d/conda.sh"
    PYTHON_BIN=(conda run --no-capture-output -n "${ENV_NAME}" python)
fi

mkdir -p "${OUT_DIR}"

"${PYTHON_BIN[@]}" tools/optimize_ada_offset.py \
    --mode infer \
    --pn "${PN}" \
    --model_type infinity_2b \
    --model_path "${MODEL_PATH}" \
    --vae_type 32 \
    --vae_path "${VAE_PATH}" \
    --text_encoder_ckpt "${TEXT_ENCODER}" \
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
    --offset_path "${OFFSET_PATH}" \
    --offset_infer_scales "${OFFSET_SCALES}" \
    --hybrid_prompt "${HYBRID_PROMPT}" \
    --hybrid_prompt_all_scales \
    --hybrid_cfg "${HYBRID_CFG}" \
    --num_samples "${NUM_SAMPLES}" \
    --save_file "${OUT_DIR}/ca_kv_full.jpg" \
    2>&1 | tee "${OUT_DIR}/inference.log"

echo "Completed: ${OUT_DIR}/ca_kv_full.jpg"
