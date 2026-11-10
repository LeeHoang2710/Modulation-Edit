#!/usr/bin/env bash
set -euo pipefail

# Re-optimize black_cat and fat_dog on scale prefixes 0-3 through 0-12,
# then make one wrapped comparison grid per object.
#
# Example:
#   bash scripts/run_scale_reconstruction_ablation.sh
#   STEPS=200 SUBJECTS=black_cat bash scripts/run_scale_reconstruction_ablation.sh

cd "$(dirname "${BASH_SOURCE[0]}")/.."

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
STEPS="${STEPS:-100}"
SUBJECTS="${SUBJECTS:-black_cat,fat_dog}"
ENDPOINTS="${ENDPOINTS:-3,4,5,6,7,8,9,10,11,12}"

extra=()
if [[ "${FORCE:-0}" == "1" ]]; then
  extra+=(--force)
fi

"$PYTHON_BIN" tools/run_scale_reconstruction_ablation.py \
  --subjects "$SUBJECTS" \
  --endpoints "$ENDPOINTS" \
  --steps "$STEPS" \
  "${extra[@]}"
