#!/usr/bin/env bash
# Run CSD-VAR Switti and score it independently using the ARBooth ViCo protocol.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
SUBJECTS="${SUBJECTS:-all}"
MAX_PROMPTS="${MAX_PROMPTS:-0}"
IMAGES_PER_PROMPT="${IMAGES_PER_PROMPT:-8}"
STEPS="${STEPS:-200}"
CHECKPOINT_STEP="${CHECKPOINT_STEP:-150}"
SAVE_EVERY="${SAVE_EVERY:-50}"
SEED="${SEED:-42}"
RESULTS="${RESULTS:-benchmark_outputs_vico_csd_var}"
WORK_DIR="${WORK_DIR:-benchmark_work/vico_csd_var}"
REPORT_DIR="${REPORT_DIR:-benchmark_reports_vico_csd_var}"
SWITTI="${SWITTI:-yresearch/Switti-AR}"
VAE="${VAE:-yresearch/VQVAE-Switti}"
CSD_PYTHON="${CSD_PYTHON:-}"
if [[ -z "$CSD_PYTHON" && -z "${CSD_ENV_NAME:-}" && -x /venv/benchmark-csdvar/bin/python ]]; then
    CSD_PYTHON=/venv/benchmark-csdvar/bin/python
fi
EVAL_PYTHON="${EVAL_PYTHON:-}"

if [[ -n "$CSD_PYTHON" && -x "$CSD_PYTHON" ]]; then
    CSD_CMD=("$CSD_PYTHON")
elif [[ -n "$CSD_PYTHON" ]]; then
    echo "CSD_PYTHON is not executable: $CSD_PYTHON" >&2
    exit 1
elif command -v conda >/dev/null 2>&1; then
    CSD_CMD=(conda run --no-capture-output -n "${CSD_ENV_NAME:-benchmark-csdvar}" python)
else
    echo "Missing CSD-VAR Python; set CSD_PYTHON or install the benchmark-csdvar environment." >&2
    exit 1
fi
if [[ -z "$EVAL_PYTHON" && -z "${EVAL_ENV_NAME:-}" ]]; then
    EVAL_CMD=("${CSD_CMD[@]}")
elif [[ -n "$EVAL_PYTHON" && -x "$EVAL_PYTHON" ]]; then
    EVAL_CMD=("$EVAL_PYTHON")
elif [[ -n "${EVAL_ENV_NAME:-}" ]] && command -v conda >/dev/null 2>&1; then
    EVAL_CMD=(conda run --no-capture-output -n "$EVAL_ENV_NAME" python)
else
    echo "Missing evaluator Python; set EVAL_PYTHON or install the vico-modulation environment." >&2
    exit 1
fi

"${CSD_CMD[@]}" -c 'import torch, transformers, diffusers, safetensors; assert torch.cuda.is_available(), "CSD-VAR requires an available CUDA GPU"; print("[preflight] CSD-VAR CUDA:", torch.cuda.get_device_name())'
"${EVAL_CMD[@]}" -c 'import torch, torchvision, clip; print("[preflight] evaluation dependencies available")'
"${CSD_CMD[@]}" benchmark/prepare_vico.py
"${CSD_CMD[@]}" benchmark/run_csd_var_vico.py --subjects "$SUBJECTS" --max-prompts "$MAX_PROMPTS" \
    --images-per-prompt "$IMAGES_PER_PROMPT" --steps "$STEPS" --checkpoint-step "$CHECKPOINT_STEP" \
    --save-every "$SAVE_EVERY" --seed "$SEED" --switti "$SWITTI" --vae "$VAE" \
    --output "$RESULTS" --work-dir "$WORK_DIR"
"${EVAL_CMD[@]}" benchmark/evaluate_vico_arbooth_protocol.py --results "$RESULTS" \
    --report-dir "$REPORT_DIR" --model-name csd_var_switti --subjects "$SUBJECTS" --max-prompts "$MAX_PROMPTS" \
    --expected-images-per-prompt "$IMAGES_PER_PROMPT" --text-protocol arbooth-code
echo "[complete] images=$RESULTS metrics=$REPORT_DIR/summary.csv"
