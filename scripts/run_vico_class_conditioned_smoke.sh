#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

SUBJECT="${1:-cat_statue}"
PYTHON_BIN="${VICO_PYTHON:-/venv/vico-modulation/bin/python}"
[[ -x "$PYTHON_BIN" ]] || { echo "Missing Python: $PYTHON_BIN" >&2; exit 1; }
MANIFEST="${MANIFEST:-benchmark/manifests/vico.json}"
DATASET="${DATASET:-data/vico}"
MAX_PROMPTS="${MAX_PROMPTS:-5}"
STEPS="${STEPS:-50}"
OUTPUT="${OUTPUT:-benchmark_outputs/vico_class_conditioned/$SUBJECT}"
WORK="${WORK:-benchmark_work/vico_class_conditioned/$SUBJECT}"
REPORT="${REPORT:-benchmark_reports/vico_class_conditioned/$SUBJECT}"
[[ -f "$MANIFEST" && -d "$DATASET/$SUBJECT" ]] || { echo "Prepare ViCo data/manifest first" >&2; exit 1; }
"$PYTHON_BIN" -c 'import json,sys; m=json.load(open(sys.argv[1])); assert any(x["subject"]==sys.argv[2] for x in m["subjects"]), "Unknown ViCo subject"' "$MANIFEST" "$SUBJECT"
CLASS_NAME="${SUBJECT//_/ }"
TRAIN_PROMPT="a $CLASS_NAME"
MODEL=(--pn 1M --model_type infinity_2b --model_path weights/infinity/infinity_2b_reg.pth
  --vae_type 32 --vae_path weights/infinity/infinity_vae_d32reg.pth
  --add_lvl_embeding_only_first_block 1 --use_bit_label 1
  --rope2d_each_sa_layer 1 --rope2d_normalized_by_hw 2
  --use_scale_schedule_embedding 0 --apply_spatial_patchify 0
  --checkpoint_type torch --bf16 1 --text_channels 2048
  --text_encoder_ckpt weights/infinity/flan-t5-xl)
mkdir -p "$WORK" "$OUTPUT"
OFFSET="$WORK/offset.pth"
echo "[fit] $SUBJECT prompt='$TRAIN_PROMPT' scales=0-4"
"$PYTHON_BIN" tools/optimize_ada_offset.py --mode optimize --images "$DATASET/$SUBJECT" \
  "${MODEL[@]}" --train_prompt "$TRAIN_PROMPT" --seed 0 --hflip 0 \
  --grad_checkpointing 1 --offset_mode per_scale --iters "$STEPS" \
  --batch_size "${BATCH_SIZE:-5}" --grad_accum "${GRAD_ACCUM:-1}" --lr "${LR:-4e-3}" \
  --loss_scales 0-4 --offset_infer_scales 0-4 --num_samples 1 --sample_every 0 \
  --save_offset "$OFFSET" --save_file "$WORK/reconstruction.jpg" \
  2>&1 | tee "$WORK/fit.log"
echo "[infer] full class prompts, ca_kv and class-base modulation"
"$PYTHON_BIN" tools/infer_offset_vico_scale_schedule.py --offset-path "$OFFSET" \
  --manifest "$MANIFEST" --subject "$SUBJECT" --out-dir "$OUTPUT/method" \
  "${MODEL[@]}" --class-conditioned-base --subject-as-class --prompt-mode class \
  --alpha-schedule "${ALPHA_SCHEDULE:-0.25,0.20,0.15,0.10,0.05}" \
  --early-cfg "${EARLY_CFG:-1.2}" --late-cfg "${LATE_CFG:-2.0}" \
  --late-gss prompt --num-samples 1 --seed "${INFER_SEED:-0}" \
  --max-prompts "$MAX_PROMPTS" --timing-file "$WORK/timing.json" \
  2>&1 | tee "$WORK/infer.log"
echo "[done] offset=$OFFSET images=$OUTPUT/method/$SUBJECT"
