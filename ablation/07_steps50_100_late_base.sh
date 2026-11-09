#!/usr/bin/env bash
# One 100-step optimization (checkpoint at 50), followed by four variants:
# {step 50, step 100} x {late G_subject, late G_prompt}.
set -euo pipefail
source "$(dirname "$0")/_env.sh"

OUT="${OUT:-${ROOT}/22_steps50_100_late_base_seed3}"
OFFSET_BASE="${OFFSET_BASE:-${OUT}/offsets}"
INFER_SEED="${INFER_SEED:-3}"
OPT_SEED="${OPT_SEED:-3}"
NUM_SAMPLES="${NUM_SAMPLES:-1}"
INFER_BATCH="${INFER_BATCH:-4}"
BATCH_SIZE="${BATCH_SIZE:-4}"
GRAD_ACCUM="${GRAD_ACCUM:-1}"
LR="${LR:-5e-3}"
FORCE_OPTIMIZE="${FORCE_OPTIMIZE:-0}"

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

mkdir -p "$OUT/logs" "$OFFSET_BASE/step50" "$OFFSET_BASE/step100"
echo "[optimize] ${#selected_subjects[@]} subject(s); one 100-step trajectory; checkpoint at 50; batch=$BATCH_SIZE"
for subject in "${selected_subjects[@]}"; do
  step50_dir="$OFFSET_BASE/step50/$subject"
  step100_dir="$OFFSET_BASE/step100/$subject"
  step50_offset="$step50_dir/offset.pth"
  step100_offset="$step100_dir/offset.pth"
  intermediate="$step100_dir/offset_it00050.pth"
  if [ -f "$step50_offset" ] && [ -f "$step100_offset" ] && [ "$FORCE_OPTIMIZE" != 1 ]; then
    echo "[optimize] reuse $subject"
    continue
  fi
  mkdir -p "$step50_dir" "$step100_dir"
  subject_text="$($PYTHON -c "import sys; sys.path.insert(0, 'ablation'); from common import subject_name; print(subject_name('$subject'))")"
  echo "[optimize] $subject with subject text '$subject_text'"
  "$PYTHON" tools/optimize_ada_offset.py --mode optimize "${model[@]}" \
    --seed "$OPT_SEED" \
    --images "$VICO_DIR/$subject+photo/imgs" \
    --train_prompt "$subject_text" \
    --hflip 0 \
    --grad_checkpointing 1 \
    --offset_mode per_scale \
    --iters 100 \
    --batch_size "$BATCH_SIZE" \
    --grad_accum "$GRAD_ACCUM" \
    --lr "$LR" \
    --loss_scales 0-5 \
    --noise_apply_layers 13 \
    --noise_apply_strength 0.3 \
    --log_every 10 \
    --save_offset_every 50 \
    --save_offset "$step100_offset" \
    --timing_file "$step100_dir/timing.json" \
    --save_file '' \
    > "$step100_dir/optimize.log" 2>&1
  if [ ! -f "$intermediate" ]; then
    echo "Missing step-50 checkpoint: $intermediate" >&2
    exit 1
  fi
  cp -f "$intermediate" "$step50_offset"
  cp -f "$step100_dir/timing.json" "$step50_dir/timing.json"
done
"$PYTHON" ablation/summarize_checkpoint_timings.py \
  --offset-base "$OFFSET_BASE" \
  --output "$OUT/preparation_times.csv"

echo "[generate] four variants; inference seed=$INFER_SEED; samples/prompt=$NUM_SAMPLES; batch=$INFER_BATCH"
"$PYTHON" ablation/ablation_steps_late_base.py \
  --out_dir "$OUT" \
  --offset-base "$OFFSET_BASE" \
  --offset-scales "0,2-5" \
  --prompt-only-scales "1,6-12" \
  --beta 1.0 \
  --prompt_name subject \
  --subjects "$SUBJECTS" \
  --num_prompts "$NUM_PROMPTS" \
  --num_samples "$NUM_SAMPLES" \
  --batch "$INFER_BATCH" \
  --cfg "$CFG" \
  --tau "$TAU" \
  --seed "$INFER_SEED" \
  --progress \
  --progress_desc "steps 50/100 x Gsubject/Gprompt" \
  > "$OUT/logs/generation.log" 2>&1

echo "[evaluate] DINO, CLIP-I, and CLIP-T for four variants"
for variant in \
  step050_late_gsubject step050_late_gprompt \
  step100_late_gsubject step100_late_gprompt; do
  VARIANT="$variant" bash ablation/eval_nonoffset_prompt_basic.sh "$OUT" \
    > "$OUT/logs/eval_${variant}.log" 2>&1
done
"$PYTHON" ablation/summarize.py "$OUT"
"$PYTHON" ablation/summarize_inference_timings.py \
  --experiment-dir "$OUT" \
  --variants \
    step050_late_gsubject step050_late_gprompt \
    step100_late_gsubject step100_late_gprompt
echo "[done] $OUT/summary.csv"
echo "[done] $OUT/preparation_times.csv"
echo "[done] $OUT/inference_times_per_image.csv"
echo "[done] $OUT/inference_time_summary.csv"
