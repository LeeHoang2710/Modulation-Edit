#!/usr/bin/env bash
# Sequentially prepares each requested baseline, saves every generated image,
# then reports DINO, CLIP-I and CLIP-T with preparation/object and inference/image.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

SUBJECTS="${SUBJECTS:-all}"              # smoke test: SUBJECTS=dog
MODELS="${MODELS:-arbooth,csd_var,dreambooth,textual_inversion,unziplora}"
IMAGES_PER_PROMPT="${IMAGES_PER_PROMPT:-4}"
SEED="${SEED:-0}"
RESULTS="${RESULTS:-benchmark_outputs}"
REPORTS="${REPORTS:-benchmark_reports}"
MANIFEST="${MANIFEST:-benchmark/manifests/dreambooth.json}"

[[ -f "$MANIFEST" ]] || python benchmark/prepare_dreambooth.py

env_for() {
    case "$1" in
        arbooth) echo benchmark-arbooth ;;
        csd_var) echo benchmark-csdvar ;;
        dreambooth|textual_inversion) echo benchmark-ldm ;;
        unziplora) echo benchmark-unziplora ;;
        *) echo "Unknown baseline: $1" >&2; exit 2 ;;
    esac
}

IFS=',' read -r -a requested <<< "$MODELS"
for model in "${requested[@]}"; do
    model="${model// /}"
    environment="$(env_for "$model")"
    echo "========== [$model] subjects=$SUBJECTS =========="
    conda run --no-capture-output -n "$environment" python benchmark/run_baseline.py \
        --model "$model" --manifest "$MANIFEST" --output "$RESULTS" --subjects "$SUBJECTS" \
        --images-per-prompt "$IMAGES_PER_PROMPT" --seed "$SEED"
done

conda run --no-capture-output -n benchmark-eval python benchmark/evaluate_dreambooth.py \
    --results "$RESULTS" --dataset data/dreambooth/dataset --report-dir "$REPORTS" --models "$MODELS"

echo "Completed. Images: $RESULTS | numerical results: $REPORTS/summary.csv"
