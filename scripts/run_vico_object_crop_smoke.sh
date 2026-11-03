#!/usr/bin/env bash
# Detect/crop one ViCo subject, fit one scale-0..4 offset through 100 steps,
# then generate and score the same five prompts at steps 50, 75, and 100.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

SUBJECT="${1:?Usage: bash scripts/run_vico_object_crop_smoke.sh SUBJECT (e.g. cat_statue)}"
VICO_PYTHON="${VICO_PYTHON:-/venv/vico-modulation/bin/python}"
ENV_NAME="${ENV_NAME:-vico-modulation}"
if [[ -x "$VICO_PYTHON" ]]; then
    PYTHON_BIN=("$VICO_PYTHON")
elif command -v conda >/dev/null 2>&1; then
    CONDA_BASE="$(conda info --base)"
    # shellcheck disable=SC1091
    source "$CONDA_BASE/etc/profile.d/conda.sh"
    PYTHON_BIN=(conda run --no-capture-output -n "$ENV_NAME" python)
else
    echo "Missing $VICO_PYTHON or Conda environment $ENV_NAME" >&2
    exit 1
fi

DATASET="${DATASET:-data/vico}"
CROP_ROOT="${CROP_ROOT:-data/vico_auto_cropped}"
MANIFEST_ROOT="${MANIFEST_ROOT:-benchmark/manifests}"
MANIFEST="$MANIFEST_ROOT/${SUBJECT}_auto_cropped.json"
WORK_DIR="${WORK_DIR:-benchmark_work/vico_crop_smoke/${SUBJECT}}"
RESULTS="${RESULTS:-benchmark_outputs/vico_crop_smoke_modifier_only/${SUBJECT}}"
REPORT_DIR="${REPORT_DIR:-benchmark_reports/vico_crop_smoke_modifier_only/${SUBJECT}}"
METHOD="${METHOD:-ours_${SUBJECT}_modifier_only_smoke}"
BATCH_SIZE="${BATCH_SIZE:-5}"
GRAD_ACCUM="${GRAD_ACCUM:-1}"
LR="${LR:-4e-3}"
INFER_SEED="${INFER_SEED:-0}"
ALPHA_SCHEDULE="${ALPHA_SCHEDULE:-0.25,0.20,0.15,0.10,0.05}"
EARLY_CFG="${EARLY_CFG:-1.2}"
LATE_CFG="${LATE_CFG:-2.0}"
DETECTOR="${DETECTOR:-IDEA-Research/grounding-dino-tiny}"
BOX_THRESHOLD="${BOX_THRESHOLD:-0.25}"
MARGIN="${MARGIN:-0.12}"

if [[ ! -d "$DATASET/$SUBJECT" ]]; then
    echo "Missing $DATASET/$SUBJECT. Run: bash scripts/prepare_vico_assets.sh" >&2
    exit 1
fi

crop_args=("$SUBJECT" --inputs "$DATASET" --output-root "$CROP_ROOT" \
    --manifest-root "$MANIFEST_ROOT" --detector "$DETECTOR" \
    --box-threshold "$BOX_THRESHOLD" --margin "$MARGIN")
if [[ -n "${CROP_LABEL:-}" ]]; then
    crop_args+=(--label "$CROP_LABEL")
fi
"${PYTHON_BIN[@]}" benchmark/prepare_vico_object_crops.py "${crop_args[@]}"
echo "[review] Crop preview: $CROP_ROOT/${SUBJECT}_crop_preview.jpg"

if [[ "${CROP_ONLY:-0}" == "1" ]]; then
    echo "[complete] Crops prepared; CROP_ONLY=1 skipped training and inference."
    exit 0
fi

model_args=(--pn 1M --model_type infinity_2b --model_path weights/infinity/infinity_2b_reg.pth \
    --vae_type 32 --vae_path weights/infinity/infinity_vae_d32reg.pth \
    --add_lvl_embeding_only_first_block 1 --use_bit_label 1 \
    --rope2d_each_sa_layer 1 --rope2d_normalized_by_hw 2 \
    --use_scale_schedule_embedding 0 --apply_spatial_patchify 0 \
    --checkpoint_type torch --bf16 1 --text_channels 2048)
for required in weights/infinity/infinity_2b_reg.pth weights/infinity/infinity_vae_d32reg.pth; do
    [[ -f "$required" ]] || { echo "Missing $required; run scripts/prepare_vico_assets.sh" >&2; exit 1; }
done
[[ -d weights/infinity/flan-t5-xl ]] || { echo "Missing Flan-T5-XL; run scripts/prepare_vico_assets.sh" >&2; exit 1; }

OFFSET_DIR="$WORK_DIR/offsets/$SUBJECT/train_seed000"
mkdir -p "$OFFSET_DIR" "$WORK_DIR/logs"
if [[ "${REUSE_OFFSETS:-1}" == "1" && -f "$OFFSET_DIR/offset_it00050.pth" \
      && -f "$OFFSET_DIR/offset_it00075.pth" && -f "$OFFSET_DIR/offset.pth" ]]; then
    echo "[reuse] using existing 50/75/100 checkpoints from $OFFSET_DIR"
else
    echo "[fit] subject=$SUBJECT; one 100-step run; checkpoints 50, 75, 100; scales 0-4"
    "${PYTHON_BIN[@]}" tools/optimize_ada_offset.py \
        --mode optimize --images "$CROP_ROOT/$SUBJECT" "${model_args[@]}" \
        --seed 0 --hflip 0 --grad_checkpointing 1 --offset_mode per_scale \
        --iters 100 --batch_size "$BATCH_SIZE" --grad_accum "$GRAD_ACCUM" --lr "$LR" \
        --loss_scales 0-4 --offset_infer_scales 0-4 \
        --num_samples 1 --sample_every 0 --save_offset_at 50,75 \
        --save_offset "$OFFSET_DIR/offset.pth" --save_file "$OFFSET_DIR/reconstruction.jpg" \
        2>&1 | tee "$WORK_DIR/logs/fit_100_steps.log"
fi

for step in 50 75 100; do
    printf -v step_tag '%03d' "$step"
    if [[ "$step" == 100 ]]; then
        offset_path="$OFFSET_DIR/offset.pth"
    else
        printf -v step_tag_5 '%05d' "$step"
        offset_path="$OFFSET_DIR/offset_it${step_tag_5}.pth"
    fi
    [[ -f "$offset_path" ]] || { echo "Missing checkpoint: $offset_path" >&2; exit 1; }
    variant="$RESULTS/step_${step_tag}/${METHOD}_late_prompt"
    timing_file="$WORK_DIR/timings/step_${step_tag}.json"
    mkdir -p "$(dirname "$timing_file")" "$variant"
    echo "[infer] step=$step; five modifier-only prompts; seed=$INFER_SEED; alpha=$ALPHA_SCHEDULE"
    "${PYTHON_BIN[@]}" tools/infer_offset_vico_scale_schedule.py \
        --offset-path "$offset_path" --manifest "$MANIFEST" --subject "$SUBJECT" \
        --out-dir "$variant" "${model_args[@]}" \
        --text_encoder_ckpt weights/infinity/flan-t5-xl \
        --alpha-schedule "$ALPHA_SCHEDULE" --early-cfg "$EARLY_CFG" \
        --late-cfg "$LATE_CFG" --late-gss prompt --prompt-mode modifier-only \
        --num-samples 1 --seed "$INFER_SEED" --max-prompts 5 \
        --timing-file "$timing_file" \
        2>&1 | tee "$WORK_DIR/logs/infer_step_${step_tag}.log"

    # The existing evaluator reads inference time from <subject>/timing.json.
    "${PYTHON_BIN[@]}" -c 'import json,sys; from pathlib import Path; t=json.loads(Path(sys.argv[1]).read_text()); d=Path(sys.argv[2]); d.parent.mkdir(parents=True,exist_ok=True); d.write_text(json.dumps({"inference_seconds":t["generation_seconds"],"inference_seconds_per_image":t["generation_seconds_per_image"],"images":t["images"]},indent=2)+"\n")' \
        "$timing_file" "$variant/$SUBJECT/timing.json"

    # Evaluate against the unchanged source references, not the training crops.
    "${PYTHON_BIN[@]}" benchmark/evaluate_vico_arbooth_protocol.py \
        --manifest "$MANIFEST" --dataset "$DATASET" \
        --results "$variant" --model-name "${METHOD}_step_${step_tag}" \
        --subjects "$SUBJECT" --max-prompts 5 --expected-images-per-prompt 1 \
        --selection mean8 --text-protocol modifier-only \
        --report-dir "$REPORT_DIR/step_${step_tag}"
done

echo "[complete] Crops: $CROP_ROOT/$SUBJECT"
echo "[complete] Checkpoints: $OFFSET_DIR/offset_it00050.pth, offset_it00075.pth, offset.pth"
echo "[complete] Images: $RESULTS/step_{050,075,100}/${METHOD}_late_prompt/$SUBJECT"
echo "[complete] Metrics: $REPORT_DIR/step_{050,075,100}/summary.csv"
echo "[complete] Logs: $WORK_DIR/logs"
