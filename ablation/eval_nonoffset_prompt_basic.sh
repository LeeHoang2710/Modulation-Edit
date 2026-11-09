#!/usr/bin/env bash
# Evaluate only DINO, CLIP-I, and CLIP-T for the new non-offset G_prompt variant.
set -euo pipefail
source "$(dirname "$0")/_env.sh"

PART="${1:-${ROOT}/21_nonoffset_prompt_reopt_seed3}"
VARIANT="${VARIANT:-test_nonoffset_gprompt}"
RESULT_DIR="$PART/$VARIANT"

if [ ! -d "$RESULT_DIR" ]; then
  echo "Missing generated images: $RESULT_DIR" >&2
  exit 1
fi

echo "[evaluate-basic] $RESULT_DIR"
"$PYTHON" ablation/evaluate_personalization.py \
  --basic-only \
  --result_dir "$RESULT_DIR" \
  --data_dir "$VICO_DIR" \
  --arbooth_root "$ARBOOTH_ROOT" \
  | tee "$RESULT_DIR/eval_basic.log"

echo "[done] $RESULT_DIR/metrics.csv"
