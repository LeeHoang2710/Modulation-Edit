#!/usr/bin/env bash
# Subject-conditioned BSC offset ablation for
#   G_s = G_subject + beta*D_s + alpha*G_prompt, s in {0,2,3,4}
#         G_subject,                           otherwise.
#
# Usage:
#   SWEEP=beta FIXED_ALPHA=0 bash ablation/05_subject_prompt_alpha_beta.sh all
#   SWEEP=alpha FIXED_BETA=0.9 bash ablation/05_subject_prompt_alpha_beta.sh all
#   SWEEP=grid bash ablation/05_subject_prompt_alpha_beta.sh all
# Stages: optimize | generate | eval | all (default: all)
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source ablation/_env.sh >/dev/null

STAGE="${1:-all}"
SWEEP="${SWEEP:-beta}"
ALPHAS="${ALPHAS:-0,0.1,0.2,0.3,0.4}"
BETAS="${BETAS:-0.6,0.7,0.8,0.9,1.0}"
FIXED_ALPHA="${FIXED_ALPHA:-0.0}"
FIXED_BETA="${FIXED_BETA:-0.9}"
OFFSET_ROOT="${OFFSET_ROOT:-output/ablation_vico/offsets/ls0-4_subjtext_it200_bsc_noaug}"
OUT_BASE="${OUT_BASE:-output/ablation_vico/20_subject_prompt_alpha_beta}"
OUT="${OUT:-${OUT_BASE}/${SWEEP}}"

case "$STAGE" in
  optimize|generate|eval|all) ;;
  *) echo "Unknown stage '$STAGE'; expected optimize, generate, eval, or all" >&2; exit 2 ;;
esac
case "$SWEEP" in
  alpha|beta|grid) ;;
  *) echo "Unknown SWEEP '$SWEEP'; expected alpha, beta, or grid" >&2; exit 2 ;;
esac

echo "[config] stage=$STAGE sweep=$SWEEP subjects=${SUBJECTS:-all} prompts=$NUM_PROMPTS samples=$NUM_SAMPLES"
echo "[config] alphas=$ALPHAS betas=$BETAS fixed_alpha=$FIXED_ALPHA fixed_beta=$FIXED_BETA"
echo "[config] offsets=$OFFSET_ROOT output=$OUT"

if [[ "$STAGE" == "optimize" || "$STAGE" == "all" ]]; then
  # Reuse the friend's selected subject-text + BSC recipe. Completed subjects
  # are skipped by that launcher.
  OFFSET_ROOT="$OFFSET_ROOT" bash run_vico_bsc_best.sh optimize
fi

if [[ "$STAGE" == "generate" || "$STAGE" == "all" ]]; then
  "$PYTHON" ablation/ablation_alpha_beta.py \
    --out_dir "$OUT" \
    --offset_root "$OFFSET_ROOT" \
    --sweep "$SWEEP" \
    --alphas "$ALPHAS" \
    --betas "$BETAS" \
    --fixed-alpha "$FIXED_ALPHA" \
    --fixed-beta "$FIXED_BETA" \
    --prompt_name subject \
    "${infer_args[@]}"
fi

if [[ "$STAGE" == "eval" || "$STAGE" == "all" ]]; then
  bash ablation/eval_vico.sh "$OUT"
fi

echo "[complete] $OUT"
