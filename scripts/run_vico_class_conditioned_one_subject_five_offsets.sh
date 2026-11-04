#!/usr/bin/env bash
# Fit five independent class-conditioned offsets for one selected ViCo subject,
# then generate the class-only prompt and the first five official prompts.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

SUBJECT="${1:-black_cat}"
PYTHON_BIN="${VICO_PYTHON:-/venv/vico-modulation/bin/python}"
[[ -x "$PYTHON_BIN" ]] || { echo "Missing Python: $PYTHON_BIN" >&2; exit 1; }

OFFSET_SEEDS="${OFFSET_SEEDS:-5}"
INFER_SEED="${INFER_SEED:-0}"
STEPS="${STEPS:-200}"
BATCH_SIZE="${BATCH_SIZE:-2}"
GRAD_ACCUM="${GRAD_ACCUM:-1}"
LR="${LR:-5e-4}"
BSC_LAYERS="${BSC_LAYERS:-4}"
BSC_STRENGTH="${BSC_STRENGTH:-0.3}"
MAX_PROMPTS="${MAX_PROMPTS:-5}"
REUSE_OFFSETS="${REUSE_OFFSETS:-1}"
MANIFEST="${MANIFEST:-benchmark/manifests/vico.json}"
DATASET="${DATASET:-data/vico}"
WORK_ROOT="${WORK_ROOT:-benchmark_work/vico_class_conditioned_one_subject_five_offsets_bsc03}"
OUTPUT_ROOT="${OUTPUT_ROOT:-benchmark_outputs/vico_class_conditioned_one_subject_five_offsets_bsc03_alpha01}"
ALPHA_SCHEDULE="0.1,0.1,0.1,0.1,0.1"

[[ "$OFFSET_SEEDS" =~ ^[1-9][0-9]*$ ]] || { echo "OFFSET_SEEDS must be positive" >&2; exit 1; }
[[ -f "$MANIFEST" ]] || { echo "Missing $MANIFEST; run scripts/prepare_vico_assets.sh" >&2; exit 1; }
[[ -d "$DATASET/$SUBJECT" ]] || { echo "Missing subject data: $DATASET/$SUBJECT" >&2; exit 1; }
"$PYTHON_BIN" -c 'import json,sys; m=json.load(open(sys.argv[1])); assert any(x["subject"]==sys.argv[2] for x in m["subjects"]), f"Unknown subject: {sys.argv[2]}"' "$MANIFEST" "$SUBJECT"

CLASS_NAME="${SUBJECT//_/ }"
TRAIN_PROMPT="a $CLASS_NAME"
MODEL=(--pn 1M --model_type infinity_2b --model_path weights/infinity/infinity_2b_reg.pth
  --vae_type 32 --vae_path weights/infinity/infinity_vae_d32reg.pth
  --add_lvl_embeding_only_first_block 1 --use_bit_label 1
  --rope2d_each_sa_layer 1 --rope2d_normalized_by_hw 2
  --use_scale_schedule_embedding 0 --apply_spatial_patchify 0
  --checkpoint_type torch --bf16 1 --text_channels 2048
  --text_encoder_ckpt weights/infinity/flan-t5-xl)

for required in weights/infinity/infinity_2b_reg.pth weights/infinity/infinity_vae_d32reg.pth; do
  [[ -f "$required" ]] || { echo "Missing $required; run scripts/prepare_vico_assets.sh" >&2; exit 1; }
done
[[ -d weights/infinity/flan-t5-xl ]] || { echo "Missing weights/infinity/flan-t5-xl" >&2; exit 1; }

for ((offset_seed=0; offset_seed<OFFSET_SEEDS; offset_seed++)); do
  printf -v seed_tag '%03d' "$offset_seed"
  offset_dir="$WORK_ROOT/$SUBJECT/train_seed${seed_tag}"
  offset_path="$offset_dir/offset.pth"
  mkdir -p "$offset_dir" "$WORK_ROOT/logs/$SUBJECT" "$WORK_ROOT/timings/$SUBJECT"

  reuse=0
  if [[ "$REUSE_OFFSETS" == "1" && -f "$offset_path" ]]; then
    if "$PYTHON_BIN" -c 'import sys,torch; c=torch.load(sys.argv[1],map_location="cpu"); assert c.get("train_prompt")==sys.argv[2] and set(c.get("loss_scales",[]))==set(range(5)) and c.get("seed")==int(sys.argv[3]) and c.get("iters")==int(sys.argv[4]) and c.get("batch_size")==int(sys.argv[5]) and c.get("grad_accum")==int(sys.argv[6]) and abs(c.get("lr")-float(sys.argv[7]))<1e-12 and c.get("noise_apply_layers")==int(sys.argv[8]) and abs(c.get("noise_apply_strength")-float(sys.argv[9]))<1e-12' "$offset_path" "$TRAIN_PROMPT" "$offset_seed" "$STEPS" "$BATCH_SIZE" "$GRAD_ACCUM" "$LR" "$BSC_LAYERS" "$BSC_STRENGTH"; then
      reuse=1
    fi
  fi

  if [[ "$reuse" == "1" ]]; then
    echo "[reuse] subject=$SUBJECT offset_seed=$offset_seed path=$offset_path"
  else
    echo "[fit] subject=$SUBJECT conditioning='$TRAIN_PROMPT' offset_seed=$offset_seed scales=0-4 steps=$STEPS batch=$BATCH_SIZE lr=$LR bsc_layers=$BSC_LAYERS bsc_strength=$BSC_STRENGTH"
    "$PYTHON_BIN" tools/optimize_ada_offset.py --mode optimize --images "$DATASET/$SUBJECT" \
      "${MODEL[@]}" --train_prompt "$TRAIN_PROMPT" --seed "$offset_seed" --hflip 0 \
      --grad_checkpointing 1 --offset_mode per_scale --iters "$STEPS" \
      --batch_size "$BATCH_SIZE" --grad_accum "$GRAD_ACCUM" --lr "$LR" \
      --noise_apply_layers "$BSC_LAYERS" --noise_apply_strength "$BSC_STRENGTH" \
      --loss_scales 0-4 --offset_infer_scales 0-4 --num_samples 1 --sample_every 0 \
      --save_offset "$offset_path" --save_file "$offset_dir/reconstruction.jpg" \
      2>&1 | tee "$WORK_ROOT/logs/$SUBJECT/fit_seed${seed_tag}.log"
  fi

  echo "[infer] subject=$SUBJECT offset_seed=$offset_seed inference_seed=$INFER_SEED prompts=$((MAX_PROMPTS + 1))"
  "$PYTHON_BIN" tools/infer_offset_vico_scale_schedule.py --offset-path "$offset_path" \
    --manifest "$MANIFEST" --subject "$SUBJECT" --out-dir "$OUTPUT_ROOT/method" \
    "${MODEL[@]}" --class-conditioned-base --subject-as-class --prompt-mode class \
    --offset-scales 0-4 --alpha-schedule "$ALPHA_SCHEDULE" --early-composition preserve-offset \
    --early-cfg "${EARLY_CFG:-1.2}" --late-cfg "${LATE_CFG:-2.0}" --late-gss prompt \
    --prepend-class-prompt --max-prompts "$MAX_PROMPTS" --num-samples 1 \
    --seed "$INFER_SEED" --output-seed-start "$offset_seed" \
    --timing-file "$WORK_ROOT/timings/$SUBJECT/train_seed${seed_tag}.json" \
    2>&1 | tee "$WORK_ROOT/logs/$SUBJECT/infer_seed${seed_tag}.log"
done

expected_images=$((OFFSET_SEEDS * (MAX_PROMPTS + 1)))
verified_images=0
for ((prompt_index=0; prompt_index<=MAX_PROMPTS; prompt_index++)); do
  printf -v prompt_tag '%02d' "$prompt_index"
  for ((offset_seed=0; offset_seed<OFFSET_SEEDS; offset_seed++)); do
    printf -v seed_tag '%03d' "$offset_seed"
    image_path="$OUTPUT_ROOT/method/$SUBJECT/prompt_${prompt_tag}/seed_${seed_tag}.png"
    [[ -f "$image_path" ]] || { echo "Missing expected image: $image_path" >&2; exit 1; }
    verified_images=$((verified_images + 1))
  done
done
[[ "$verified_images" == "$expected_images" ]] || { echo "Expected $expected_images images, verified $verified_images" >&2; exit 1; }

echo "[complete] offsets=$WORK_ROOT/$SUBJECT"
echo "[complete] images=$OUTPUT_ROOT/method/$SUBJECT"
echo "[complete] verified_images=$verified_images ($OFFSET_SEEDS offsets x $((MAX_PROMPTS + 1)) prompts)"
