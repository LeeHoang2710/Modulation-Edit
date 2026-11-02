#!/usr/bin/env bash
# Detect/crop every reference for one ViCo subject, fit one scale-0..4 offset,
# then generate and score the first five official prompts.
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
RESULTS="${RESULTS:-benchmark_outputs/vico_crop_smoke/${SUBJECT}}"
REPORT_DIR="${REPORT_DIR:-benchmark_reports/vico_crop_smoke/${SUBJECT}}"
METHOD="${METHOD:-ours_${SUBJECT}_auto_crop_smoke}"
STEPS="${STEPS:-50}"
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

echo "[smoke] subject=$SUBJECT; one offset; five prompts; scales 0-4; alpha=$ALPHA_SCHEDULE"
"${PYTHON_BIN[@]}" benchmark/run_ours_vico_two_variant.py \
    --manifest "$MANIFEST" --subjects "$SUBJECT" \
    --offset-seeds 1 --samples-per-offset 1 \
    --inference-seed-policy fixed --fixed-inference-seed "$INFER_SEED" \
    --max-prompts 5 --steps "$STEPS" --batch-size "$BATCH_SIZE" \
    --grad-accum "$GRAD_ACCUM" --lr "$LR" \
    --alpha-schedule "$ALPHA_SCHEDULE" --early-cfg "$EARLY_CFG" --late-cfg "$LATE_CFG" \
    --variants prompt --method-prefix "$METHOD" \
    --output "$RESULTS" --work-dir "$WORK_DIR" --force-optimize

# Original references, rather than cropped training images, are used for I-DINO and I-CLIP.
"${PYTHON_BIN[@]}" benchmark/evaluate_vico_arbooth_protocol.py \
    --manifest "$MANIFEST" --dataset "$DATASET" \
    --results "$RESULTS/${METHOD}_late_prompt" \
    --model-name "$METHOD" --subjects "$SUBJECT" --max-prompts 5 \
    --expected-images-per-prompt 1 --selection mean8 \
    --text-protocol arbooth-code --report-dir "$REPORT_DIR"

echo "[complete] Crops: $CROP_ROOT/$SUBJECT"
echo "[complete] Offset: $WORK_DIR/offsets/$SUBJECT/train_seed000/offset.pth"
echo "[complete] Images: $RESULTS/${METHOD}_late_prompt/$SUBJECT"
echo "[complete] Metrics: $REPORT_DIR/summary.csv"
echo "[complete] Logs: $WORK_DIR/logs"
