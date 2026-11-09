#!/usr/bin/env bash
# Paired one-seed comparison:
#   main: scales 1,6-12 use G_subject
#   test: scales 1,6-12 use G_prompt
# Both use G_subject + D_s on scales 0,2-5, beta=1, prompt CA-KV, CFG=1.5.
set -euo pipefail
source "$(dirname "$0")/_env.sh"

OUT="${OUT:-${ROOT}/21_nonoffset_prompt_reopt_seed3}"
OFFSET_ROOT="${OFFSET_ROOT:-${OUT}/offsets_seed3_ls0-5_subjtext}"
INFER_SEED="${INFER_SEED:-3}"
OPT_SEED="${OPT_SEED:-3}"
ITERS="${ITERS:-200}"
BATCH_SIZE="${BATCH_SIZE:-2}"
GRAD_ACCUM="${GRAD_ACCUM:-2}"
LR="${LR:-5e-3}"
FORCE_OPTIMIZE="${FORCE_OPTIMIZE:-0}"
NUM_SAMPLES=1

model=(--pn 1M --model_type infinity_2b --model_path weights/infinity_2b_reg.pth
  --vae_type 32 --vae_path weights/infinity_vae_d32reg.pth
  --add_lvl_embeding_only_first_block 1 --use_bit_label 1
  --rope2d_each_sa_layer 1 --rope2d_normalized_by_hw 2
  --use_scale_schedule_embedding 0 --apply_spatial_patchify 0
  --checkpoint_type torch --bf16 1 --text_channels 2048
  --text_encoder_ckpt weights/flan-t5-xl --null_len 32)

if [ -n "$SUBJECTS" ]; then
  IFS=',' read -r -a selected_subjects <<< "$SUBJECTS"
else
  mapfile -t selected_subjects < <(tail -n +2 "$VICO_DIR/subjects.csv" | cut -d, -f2)
fi

mkdir -p "$OUT/logs" "$OFFSET_ROOT"

echo "[optimize] ${#selected_subjects[@]} subject offset(s); seed=$OPT_SEED; scales=0-5; iters=$ITERS; batch=$BATCH_SIZE; accum=$GRAD_ACCUM; lr=$LR"
for subject in "${selected_subjects[@]}"; do
  offset_dir="$OFFSET_ROOT/$subject"
  offset_path="$offset_dir/offset.pth"
  if [ -f "$offset_path" ] && [ "$FORCE_OPTIMIZE" != 1 ]; then
    echo "[optimize] reuse $subject: $offset_path"
    continue
  fi
  mkdir -p "$offset_dir"
  subject_text="$($PYTHON -c "import sys; sys.path.insert(0, 'ablation'); from common import subject_name; print(subject_name('$subject'))")"
  echo "[optimize] $subject with subject text '$subject_text'"
  "$PYTHON" tools/optimize_ada_offset.py --mode optimize "${model[@]}" \
    --seed "$OPT_SEED" \
    --images "$VICO_DIR/$subject+photo/imgs" \
    --train_prompt "$subject_text" \
    --hflip 0 \
    --grad_checkpointing 1 \
    --offset_mode per_scale \
    --iters "$ITERS" \
    --batch_size "$BATCH_SIZE" \
    --grad_accum "$GRAD_ACCUM" \
    --lr "$LR" \
    --loss_scales 0-5 \
    --noise_apply_layers 13 \
    --noise_apply_strength 0.3 \
    --log_every 10 \
    --save_offset "$offset_path" \
    --save_file '' \
    > "$offset_dir/optimize.log" 2>&1
done

echo "[generate] paired inference seed=$INFER_SEED; offsets=$OFFSET_ROOT; output=$OUT"
"$PYTHON" ablation/ablation_nonoffset_prompt.py \
  --out_dir "$OUT" \
  --offset_root "$OFFSET_ROOT" \
  --offset-scales "0,2-5" \
  --prompt-only-scales "1,6-12" \
  --beta 1.0 \
  --prompt_name subject \
  --subjects "$SUBJECTS" \
  --num_prompts "$NUM_PROMPTS" \
  --num_samples 1 \
  --batch 1 \
  --cfg "$CFG" \
  --tau "$TAU" \
  --seed "$INFER_SEED" \
  --progress \
  --progress_desc "Gsubject vs Gprompt (seed 3)" \
  > "$OUT/logs/generation.log" 2>&1

echo "[evaluate] evaluating each variant separately"
NUM_SAMPLES=1 NUM_PROMPTS="$NUM_PROMPTS" FORCE_EVAL="${FORCE_EVAL:-0}" \
  bash ablation/eval_vico.sh "$OUT" > "$OUT/logs/evaluation.log" 2>&1

cat "$OUT/summary.csv"
echo "[done] images: $OUT"
echo "[done] metrics: $OUT/summary.csv"
