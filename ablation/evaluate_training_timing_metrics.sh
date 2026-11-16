#!/usr/bin/env bash
# Evaluate the three saved timing variants with ARBooth's DINO/CLIP protocol.
# This never runs optimization or generation; it only reads existing images.
set -euo pipefail
source "$(dirname "$0")/_env.sh"

OUT="${1:-${ROOT}/24_training_speed_${SUBJECT:-black_cat}}"
SUBJECT="${SUBJECT:-black_cat}"
configs=(bs2_ga2_it200 bs4_ga1_it200 bs4_ga2_it100)

[ -x "${EVAL_PYTHON}" ] || {
  echo "Missing evaluator Python. Run: bash scripts/prepare_three_model_timing.sh" >&2
  exit 1
}
[ -f "${VICO_DIR}/subjects.csv" ] || { echo "Missing ViCo data: ${VICO_DIR}" >&2; exit 1; }

for name in "${configs[@]}"; do
  variant="${OUT}/${name}"
  image_dir="${variant}/${SUBJECT}/prompt_00"
  [ -d "${image_dir}" ] || { echo "Missing generated images: ${image_dir}" >&2; exit 1; }
  count="$(find "${image_dir}" -maxdepth 1 -type f \( -name '*.jpg' -o -name '*.png' \) | wc -l | tr -d ' ')"
  [ "${count}" -gt 0 ] || { echo "No generated images: ${image_dir}" >&2; exit 1; }
  echo "[metrics] ${name}: ${count} images"
  "${EVAL_PYTHON}" ablation/evaluate_personalization.py \
    --result_dir "${variant}" --data_dir "${VICO_DIR}" \
    --arbooth_root "${ARBOOTH_ROOT}" --csd_root "${CSD_ROOT}" --basic-only \
    > "${variant}/eval_basic.log" 2>&1
  grep '^\[overall\]' "${variant}/eval_basic.log"
done

"${PYTHON}" ablation/summarize_training_metrics.py \
  --experiment "${OUT}" --subject "${SUBJECT}" \
  | tee "${OUT}/metrics_summary.txt"
