#!/usr/bin/env bash
# Paired one-seed comparison:
#   main: scales 1,6-12 use G_subject
#   test: scales 1,6-12 use G_prompt
# Both use G_subject + D_s on scales 0,2-5, beta=1, prompt CA-KV, CFG=1.5.
set -euo pipefail
source "$(dirname "$0")/_env.sh"

OUT="${OUT:-${ROOT}/21_nonoffset_prompt_seed3}"
OFFSET_ROOT="${OFFSET_ROOT:-${ROOT}/offsets/ls0-5_subjtext_it200_bsc_noaug}"
SEED="${SEED:-3}"
NUM_SAMPLES=1

if [ ! -d "$OFFSET_ROOT" ]; then
  echo "Missing offset root: $OFFSET_ROOT" >&2
  echo "Set OFFSET_ROOT to the directory containing <subject>/offset.pth." >&2
  exit 1
fi

mkdir -p "$OUT/logs"
echo "[generate] paired seed=$SEED; offsets=$OFFSET_ROOT; output=$OUT"
"$PYTHON" ablation/ablation_nonoffset_prompt.py \
  --out_dir "$OUT" \
  --offset_root "$OFFSET_ROOT" \
  --offset-scales "0,2-5" \
  --prompt-only-scales "1,6-12" \
  --beta 1.0 \
  --prompt_name subject \
  --subjects "$SUBJECTS" \
  --num_prompts "$NUM_PROMPTS" \
  --num_samples 1 \
  --batch 1 \
  --cfg "$CFG" \
  --tau "$TAU" \
  --seed "$SEED" \
  --progress \
  --progress_desc "Gsubject vs Gprompt (seed 3)" \
  > "$OUT/logs/generation.log" 2>&1

echo "[evaluate] evaluating each variant separately"
NUM_SAMPLES=1 NUM_PROMPTS="$NUM_PROMPTS" FORCE_EVAL="${FORCE_EVAL:-0}" \
  bash ablation/eval_vico.sh "$OUT" > "$OUT/logs/evaluation.log" 2>&1

cat "$OUT/summary.csv"
echo "[done] images: $OUT"
echo "[done] metrics: $OUT/summary.csv"
