#!/usr/bin/env bash
# Two-stage subject-conditioned BSC offset ablation:
#   1. beta sweep with alpha=0 -> evaluate -> automatically select beta
#   2. alpha sweep with the selected beta -> evaluate
#
# Automatic run (default):
#   bash ablation/05_subject_prompt_alpha_beta.sh
# Manual stages remain available:
#   SWEEP=beta bash ablation/05_subject_prompt_alpha_beta.sh all
#   SWEEP=alpha FIXED_BETA=0.9 bash ablation/05_subject_prompt_alpha_beta.sh all
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source ablation/_env.sh >/dev/null

STAGE="${1:-auto}"
SWEEP="${SWEEP:-beta}"
ALPHAS="${ALPHAS:-0,0.1,0.2,0.3,0.4}"
BETAS="${BETAS:-0.6,0.7,0.8,0.9,1.0}"
FIXED_ALPHA="${FIXED_ALPHA:-0.0}"
FIXED_BETA="${FIXED_BETA:-0.9}"
BETA_SELECTION_METRIC="${BETA_SELECTION_METRIC:-dino}"
OFFSET_ROOT="${OFFSET_ROOT:-output/ablation_vico/offsets/ls0-4_subjtext_it200_bsc_noaug}"
OUT_BASE="${OUT_BASE:-output/ablation_vico/20_subject_prompt_alpha_beta}"
LOG_DIR="${LOG_DIR:-${OUT_BASE}/logs}"
SELECTION_FILE="${SELECTION_FILE:-${OUT_BASE}/selected_beta.json}"
mkdir -p "$LOG_DIR"

case "$STAGE" in
  auto|optimize|generate|eval|all) ;;
  *) echo "Unknown stage '$STAGE'; expected auto, optimize, generate, eval, or all" >&2; exit 2 ;;
esac
case "$SWEEP" in
  alpha|beta|grid) ;;
  *) echo "Unknown SWEEP '$SWEEP'; expected alpha, beta, or grid" >&2; exit 2 ;;
esac

run_optimize() {
  OFFSET_ROOT="$OFFSET_ROOT" bash run_vico_bsc_best.sh optimize \
    > "$LOG_DIR/offset_optimization.log" 2>&1
}

run_generate() {
  local sweep="$1" fixed_alpha="$2" fixed_beta="$3" description="$4" log_file="$5"
  "$PYTHON" ablation/ablation_alpha_beta.py \
    --out_dir "${OUT_BASE}/${sweep}" \
    --offset_root "$OFFSET_ROOT" \
    --sweep "$sweep" \
    --alphas "$ALPHAS" \
    --betas "$BETAS" \
    --fixed-alpha "$fixed_alpha" \
    --fixed-beta "$fixed_beta" \
    --prompt_name subject \
    --progress \
    --progress_desc "$description" \
    "${infer_args[@]}" \
    > "$LOG_DIR/$log_file" 2>&1
}

run_eval() {
  local sweep="$1" log_file="$2"
  bash ablation/eval_vico.sh "${OUT_BASE}/${sweep}" \
    > "$LOG_DIR/$log_file" 2>&1
}

if [[ "$STAGE" == "auto" ]]; then
  trap 'printf "\nPipeline failed. Check logs under %s\n" "$LOG_DIR" >&2' ERR
  run_optimize
  run_generate beta 0 "$FIXED_BETA" "1/2 beta sweep" beta_generation.log
  run_eval beta beta_evaluation.log

  "$PYTHON" ablation/select_best_beta.py \
    --summary "${OUT_BASE}/beta/summary.csv" \
    --metric "$BETA_SELECTION_METRIC" \
    --output "$SELECTION_FILE" \
    > "$LOG_DIR/beta_selection.log" 2>&1
  SELECTED_BETA="$("$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["selected_beta"])' "$SELECTION_FILE")"

  run_generate alpha "$FIXED_ALPHA" "$SELECTED_BETA" "2/2 alpha sweep" alpha_generation.log
  run_eval alpha alpha_evaluation.log
  exit 0
fi

OUT="${OUT:-${OUT_BASE}/${SWEEP}}"
if [[ "$STAGE" == "optimize" || "$STAGE" == "all" ]]; then
  run_optimize
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
