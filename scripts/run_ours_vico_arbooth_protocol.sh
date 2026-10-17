#!/usr/bin/env bash
# Run our current G_prompt modulation method under ARBooth's 16 x 31 x 8
# output/evaluation protocol. One saved offset is fitted per concept, then the
# same offset is sampled with seeds 0..7 for every prompt.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

ENV_NAME="${ENV_NAME:-vico-modulation}"
SUBJECTS="${SUBJECTS:-all}"
MAX_PROMPTS="${MAX_PROMPTS:-0}"
STEPS="${STEPS:-50}"
BATCH_SIZE="${BATCH_SIZE:-5}"
GRAD_ACCUM="${GRAD_ACCUM:-1}"
LR="${LR:-4e-3}"
RESULTS="${RESULTS:-benchmark_outputs_vico_ours_alpha01}"
WORK_DIR="${WORK_DIR:-benchmark_work/vico_modulation_ours_alpha01}"
REPORT_DIR="${REPORT_DIR:-benchmark_reports_vico_ours_alpha01}"

# Fixed-alpha variant requested for the ARBooth comparison.
ALPHA_SCHEDULE="${ALPHA_SCHEDULE:-0.10,0.10,0.10,0.10,0.10}"
EARLY_CFG="${EARLY_CFG:-1.0}"
LATE_CFG="${LATE_CFG:-2.0}"

command -v conda >/dev/null 2>&1 || {
    echo "conda is required; run scripts/setup_vico_vastai.sh first." >&2
    exit 1
}
CONDA_BASE="$(conda info --base)"
# shellcheck disable=SC1091
source "${CONDA_BASE}/etc/profile.d/conda.sh"
PYTHON_BIN=(conda run --no-capture-output -n "$ENV_NAME" python)

"${PYTHON_BIN[@]}" benchmark/prepare_vico.py
echo "[ours] subjects=$SUBJECTS prompts=$MAX_PROMPTS; one fitted offset + eight inference seeds (0..7)"
"${PYTHON_BIN[@]}" benchmark/run_ours_vico_two_variant.py \
    --subjects "$SUBJECTS" --offset-seeds 1 --samples-per-offset 8 \
    --max-prompts "$MAX_PROMPTS" --steps "$STEPS" --batch-size "$BATCH_SIZE" \
    --grad-accum "$GRAD_ACCUM" --lr "$LR" \
    --alpha-schedule "$ALPHA_SCHEDULE" --early-cfg "$EARLY_CFG" --late-cfg "$LATE_CFG" \
    --variants prompt --output "$RESULTS" --work-dir "$WORK_DIR"

# We intentionally use the same ARBooth-code metric definition as the released
# checkpoint run. Generated prompt conditioning still contains the class noun;
# only the evaluation's CLIP-T template is held identical for fair comparison.
"${PYTHON_BIN[@]}" benchmark/evaluate_vico_arbooth_protocol.py \
    --results "$RESULTS/modulation_offset_late_prompt" --model-name ours_alpha01_gprompt \
    --subjects "$SUBJECTS" --max-prompts "$MAX_PROMPTS" \
    --report-dir "$REPORT_DIR" --expected-images-per-prompt 8 \
    --text-protocol arbooth-code

echo "[complete] Ours images: $RESULTS/modulation_offset_late_prompt"
echo "[complete] Ours metrics: $REPORT_DIR/summary.csv"
