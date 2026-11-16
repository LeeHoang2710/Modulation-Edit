#!/usr/bin/env bash
# Compare one shared A-space offset injected at six candidate scale sets.
# The same subject, optimizer, seed, and main inference protocol are used for
# every variant; only the selected scale set changes.
set -euo pipefail
source "$(dirname "$0")/_env.sh"

OUT="${OUT:-${ROOT}/27_a_space_one_scale_sets_seed3}"
OFFSET_BASE="${OFFSET_BASE:-${OUT}/offsets}"
OPT_SEED="${OPT_SEED:-3}"
INFER_SEED="${INFER_SEED:-3}"
LR="${LR:-5e-3}"
FORCE_OPTIMIZE="${FORCE_OPTIMIZE:-0}"
INFER_BATCH="${INFER_BATCH:-1}"
NUM_SAMPLES=1

model=(--pn 1M --model_type infinity_2b --model_path weights/infinity_2b_reg.pth
  --vae_type 32 --vae_path weights/infinity_vae_d32reg.pth
  --add_lvl_embeding_only_first_block 1 --use_bit_label 1
  --rope2d_each_sa_layer 1 --rope2d_normalized_by_hw 2
  --use_scale_schedule_embedding 0 --apply_spatial_patchify 0
  --checkpoint_type torch --bf16 1 --text_channels 2048
  --text_encoder_ckpt weights/flan-t5-xl --null_len 32)

[ -x "${PYTHON}" ] || { echo "Missing experiment Python: ${PYTHON}" >&2; exit 1; }
[ -f "${VICO_DIR}/subjects.csv" ] || { echo "Missing ViCo manifest: ${VICO_DIR}/subjects.csv" >&2; exit 1; }

if [ -n "${SUBJECTS}" ]; then
  IFS=',' read -r -a selected_subjects <<< "${SUBJECTS}"
else
  mapfile -t selected_subjects < <(tail -n +2 "${VICO_DIR}/subjects.csv" | cut -d, -f2)
fi

scale_specs=(
  "s023:0,2,3"
  "s0123:0,1,2,3"
  "s0234:0,2,3,4"
  "s02345:0,2,3,4,5"
)

total=$(( ${#scale_specs[@]} * ${#selected_subjects[@]} ))
current=0
mkdir -p "${OUT}/logs" "${OFFSET_BASE}"

for spec in "${scale_specs[@]}"; do
  IFS=: read -r name scales <<< "${spec}"
  for subject in "${selected_subjects[@]}"; do
    current=$((current + 1))
    offset_dir="${OFFSET_BASE}/${name}/${subject}"
    offset_path="${offset_dir}/offset.pth"
    if [ -f "${offset_path}" ] && [ "${FORCE_OPTIMIZE}" != 1 ]; then
      echo "[optimize ${current}/${total}] reuse ${name}/${subject}"
      continue
    fi
    image_dir="${VICO_DIR}/${subject}+photo/imgs"
    [ -d "${image_dir}" ] || { echo "Missing subject images: ${image_dir}" >&2; exit 1; }
    mkdir -p "${offset_dir}"
    subject_text="$(${PYTHON} -c "import sys; sys.path.insert(0,'ablation'); from common import subject_name; print(subject_name('${subject}'))")"
    echo "[optimize ${current}/${total}] ${name}/${subject}: shared offset on scales=${scales}"
    "${PYTHON}" tools/optimize_ada_offset.py --mode optimize "${model[@]}" \
      --seed "${OPT_SEED}" --images "${image_dir}" --train_prompt "${subject_text}" \
      --hflip 0 --grad_checkpointing 1 --offset_mode shared --iters 200 \
      --batch_size 4 --grad_accum 1 --lr "${LR}" \
      --loss_scales "${scales}" --noise_apply_layers 13 --noise_apply_strength 0.3 \
      --log_every 10 --save_offset_every 0 --save_offset "${offset_path}" \
      --timing_file "${offset_dir}/timing.json" --save_file '' \
      > "${offset_dir}/optimize.log" 2>&1
  done
done

echo "[generate] four shared-offset scale sets; one fixed sample per prompt; inference seed=${INFER_SEED}"
"${PYTHON}" ablation/ablation_a_space_one.py \
  --out_dir "${OUT}" --offset-base "${OFFSET_BASE}" \
  --subjects "${SUBJECTS}" --num_prompts "${NUM_PROMPTS}" --num_samples 1 \
  --batch "${INFER_BATCH}" --cfg "${CFG}" --tau "${TAU}" --seed "${INFER_SEED}" \
  --prompt_name subject --progress --progress_desc "A-space-one scale sets (seed 3)" \
  > "${OUT}/logs/generation.log" 2>&1

echo "[evaluate] ARBooth-compatible DINO, CLIP-I, CLIP-T, CLIP-T class, DIV and CSD metrics"
NUM_PROMPTS="${NUM_PROMPTS}" FORCE_EVAL="${FORCE_EVAL:-0}" \
  bash ablation/eval_vico.sh "${OUT}" \
  > >(tee "${OUT}/logs/evaluation.log") 2>&1

echo "[done] ${OUT}"
echo "[done] comparison table: ${OUT}/summary.csv"
