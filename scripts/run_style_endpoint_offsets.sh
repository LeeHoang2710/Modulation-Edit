#!/usr/bin/env bash
set -euo pipefail

# Train one scale-0..4 offset from each top-level styles/*.png image, then
# compare four tests using the normal prompt "a dog": no style injection,
# injection at scale 0, injection at scale 1, and injection at scales 0+1.
#
# Example:
#   bash scripts/run_style_endpoint_offsets.sh
#   PROMPT='a dog' STEPS=100 bash scripts/run_style_endpoint_offsets.sh

cd "$(dirname "${BASH_SOURCE[0]}")/.."

PYTHON_BIN="${PYTHON_BIN:-python}"
PROMPT="${PROMPT:-a dog}"
STEPS="${STEPS:-50}"
BATCH_SIZE="${BATCH_SIZE:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-1}"
LR="${LR:-4e-3}"
SEED="${SEED:-0}"
NUM_SAMPLES="${NUM_SAMPLES:-1}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/style_single_scale_dog}"
WORK_DIR="${WORK_DIR:-benchmark_work/style_endpoint_dog}"

"${PYTHON_BIN}" tools/run_style_endpoint_offset.py \
  --styles-dir styles \
  --prompt "${PROMPT}" \
  --output-dir "${OUTPUT_DIR}" \
  --work-dir "${WORK_DIR}" \
  --steps "${STEPS}" \
  --batch-size "${BATCH_SIZE}" \
  --grad-accum "${GRAD_ACCUM}" \
  --lr "${LR}" \
  --seed "${SEED}" \
  --num-samples "${NUM_SAMPLES}"
