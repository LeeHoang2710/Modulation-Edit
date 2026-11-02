#!/usr/bin/env bash
# Train one new scale-0..4 offset from five tightly cropped alarm-clock photos,
# then generate and evaluate the first five official ViCo prompts.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

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
    echo "Missing Python environment: $VICO_PYTHON (or Conda environment $ENV_NAME)" >&2
    exit 1
fi

MANIFEST="${MANIFEST:-benchmark/manifests/alarm_clock_tight_crop.json}"
CROP_DIR="${CROP_DIR:-data/vico_alarm_clock_tight_crop/alarm_clock}"
WORK_DIR="${WORK_DIR:-benchmark_work/alarm_clock_tight_crop_smoke}"
RESULTS="${RESULTS:-benchmark_outputs/alarm_clock_tight_crop_smoke}"
REPORT_DIR="${REPORT_DIR:-benchmark_reports/alarm_clock_tight_crop_smoke}"
METHOD="${METHOD:-ours_alarm_clock_tight_crop_smoke}"
STEPS="${STEPS:-50}"
BATCH_SIZE="${BATCH_SIZE:-5}"
GRAD_ACCUM="${GRAD_ACCUM:-1}"
LR="${LR:-4e-3}"
INFER_SEED="${INFER_SEED:-0}"
ALPHA_SCHEDULE="${ALPHA_SCHEDULE:-0.25,0.20,0.15,0.10,0.05}"
EARLY_CFG="${EARLY_CFG:-1.2}"
LATE_CFG="${LATE_CFG:-2.0}"

if [[ -d data/vico/alarm_clock && -f data/vico/prompts_nonlive_objects.txt ]]; then
    SOURCE="data/vico/alarm_clock"
    DATASET="data/vico"
elif [[ -d models/ARBooth/inputs/alarm_clock && -f models/ARBooth/inputs/prompts_nonlive_objects.txt ]]; then
    SOURCE="models/ARBooth/inputs/alarm_clock"
    DATASET="models/ARBooth/inputs"
else
    echo "Missing ViCo alarm-clock inputs. Run: bash scripts/prepare_vico_assets.sh" >&2
    exit 1
fi

"${PYTHON_BIN[@]}" benchmark/prepare_alarm_clock_crops.py \
    --source "$SOURCE" --output "$CROP_DIR" --manifest "$MANIFEST"

echo "[smoke] alarm_clock: five crops, one offset, first five prompts, one inference seed"
echo "[smoke] scales 0-4; alpha=$ALPHA_SCHEDULE; early CFG=$EARLY_CFG; late G_prompt CFG=$LATE_CFG"
"${PYTHON_BIN[@]}" benchmark/run_ours_vico_two_variant.py \
    --manifest "$MANIFEST" --subjects alarm_clock \
    --offset-seeds 1 --samples-per-offset 1 \
    --inference-seed-policy fixed --fixed-inference-seed "$INFER_SEED" \
    --max-prompts 5 --steps "$STEPS" --batch-size "$BATCH_SIZE" \
    --grad-accum "$GRAD_ACCUM" --lr "$LR" \
    --alpha-schedule "$ALPHA_SCHEDULE" --early-cfg "$EARLY_CFG" --late-cfg "$LATE_CFG" \
    --variants prompt --method-prefix "$METHOD" \
    --output "$RESULTS" --work-dir "$WORK_DIR" --force-optimize

# Evaluate against the unchanged source references for a fair comparison to
# previous alarm_clock runs. These are smoke metrics from five prompts only.
"${PYTHON_BIN[@]}" benchmark/evaluate_vico_arbooth_protocol.py \
    --manifest "$MANIFEST" --dataset "$DATASET" \
    --results "$RESULTS/${METHOD}_late_prompt" \
    --model-name "$METHOD" --subjects alarm_clock --max-prompts 5 \
    --expected-images-per-prompt 1 --selection mean8 \
    --text-protocol arbooth-code --report-dir "$REPORT_DIR"

echo "[complete] Crops: $CROP_DIR"
echo "[complete] Offset: $WORK_DIR/offsets/alarm_clock/train_seed000/offset.pth"
echo "[complete] Images: $RESULTS/${METHOD}_late_prompt/alarm_clock"
echo "[complete] Metrics: $REPORT_DIR/summary.csv"
