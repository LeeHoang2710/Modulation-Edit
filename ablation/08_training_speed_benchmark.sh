#!/usr/bin/env bash
# One subject, three optimization configurations, then 100 matched inference images each.
set -euo pipefail
source "$(dirname "$0")/_env.sh"

SUBJECT="${SUBJECT:-black_cat}"
OUT="${OUT:-${ROOT}/24_training_speed_${SUBJECT}}"
OFFSET_BASE="$OUT/offsets"
OPT_SEED="${OPT_SEED:-3}"
INFER_SEED="${INFER_SEED:-0}"
LR="${LR:-5e-3}"
INFER_BATCH="${INFER_BATCH:-1}"

model=(--pn 1M --model_type infinity_2b --model_path weights/infinity_2b_reg.pth
  --vae_type 32 --vae_path weights/infinity_vae_d32reg.pth
  --add_lvl_embeding_only_first_block 1 --use_bit_label 1
  --rope2d_each_sa_layer 1 --rope2d_normalized_by_hw 2
  --use_scale_schedule_embedding 0 --apply_spatial_patchify 0
  --checkpoint_type torch --bf16 1 --text_channels 2048
  --text_encoder_ckpt weights/flan-t5-xl --null_len 32)

if [ ! -d "$VICO_DIR/$SUBJECT+photo/imgs" ]; then
  echo "Missing subject images: $VICO_DIR/$SUBJECT+photo/imgs" >&2; exit 1
fi
subject_text="$($PYTHON -c "import sys; sys.path.insert(0,'ablation'); from common import subject_name; print(subject_name('$SUBJECT'))")"
mkdir -p "$OUT/logs"

configs=("bs2_ga2_it200:2:2:200" "bs4_ga1_it200:4:1:200" "bs4_ga2_it100:4:2:100")
for spec in "${configs[@]}"; do
  IFS=: read -r name bs ga steps <<< "$spec"
  dir="$OFFSET_BASE/$name/$SUBJECT"; mkdir -p "$dir"
  echo "[optimize] $name (batch=$bs grad_accum=$ga steps=$steps)"
  "$PYTHON" tools/optimize_ada_offset.py --mode optimize "${model[@]}" \
    --seed "$OPT_SEED" --images "$VICO_DIR/$SUBJECT+photo/imgs" --train_prompt "$subject_text" \
    --hflip 0 --grad_checkpointing 1 --offset_mode per_scale --iters "$steps" \
    --batch_size "$bs" --grad_accum "$ga" --lr "$LR" --loss_scales 0-5 \
    --noise_apply_layers 13 --noise_apply_strength 0.3 --log_every 10 \
    --save_offset_every 0 --save_offset "$dir/offset.pth" --timing_file "$dir/timing.json" \
    --save_file '' > "$dir/optimize.log" 2>&1
done

echo "[infer] 3 configurations x 100 images; batch=$INFER_BATCH"
"$PYTHON" ablation/ablation_training_timing.py --out_dir "$OUT" --offset-base "$OFFSET_BASE" \
  --subjects "$SUBJECT" --num_prompts 1 --num_samples 100 --batch "$INFER_BATCH" \
  --cfg "$CFG" --tau "$TAU" --seed "$INFER_SEED" --prompt_name subject --progress \
  --progress_desc "training timing" > "$OUT/logs/inference.log" 2>&1
"$PYTHON" ablation/summarize_training_timing.py --experiment "$OUT" --subject "$SUBJECT" \
  | tee "$OUT/timing_summary.txt"
echo "[done] $OUT/timing_summary.csv"
