#!/usr/bin/env bash
set -euo pipefail

# Re-optimize black_cat and fat_dog on scale prefixes 0-3 through 0-12,
# then make one wrapped comparison grid per object.
#
# Example:
#   bash scripts/run_scale_reconstruction_ablation.sh
#   STEPS=200 SUBJECTS=black_cat bash scripts/run_scale_reconstruction_ablation.sh

cd "$(dirname "${BASH_SOURCE[0]}")/.."

PYTHON_BIN="${PYTHON_BIN:-python}"
STEPS="${STEPS:-200}"
SUBJECTS="${SUBJECTS:-black_cat,fat_dog}"

extra=()
if [[ "${FORCE:-0}" == "1" ]]; then
  extra+=(--force)
fi

"$PYTHON_BIN" tools/run_main_variant_scale_ablation.py \
  --subjects "$SUBJECTS" \
  --steps "$STEPS" \
  --pn 1M --model_path weights/infinity/infinity_2b_reg.pth \
  --vae_type 32 --vae_path weights/infinity/infinity_vae_d32reg.pth \
  --text_encoder_ckpt weights/infinity/flan-t5-xl \
  --add_lvl_embeding_only_first_block 1 --use_bit_label 1 \
  --rope2d_each_sa_layer 1 --rope2d_normalized_by_hw 2 \
  --use_scale_schedule_embedding 0 --apply_spatial_patchify 0 \
  --checkpoint_type torch --bf16 1 --text_channels 2048 \
  "${extra[@]}"
