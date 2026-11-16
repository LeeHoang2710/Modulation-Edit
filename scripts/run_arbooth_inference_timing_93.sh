#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
if [ -z "${ARBOOTH_PYTHON:-}" ]; then
  candidates=(/venv/arbooth-vico/bin/python /root/miniconda3/envs/arbooth-vico/bin/python /opt/conda/envs/arbooth-vico/bin/python /opt/miniforge3/envs/arbooth-vico/bin/python)
  if command -v conda >/dev/null 2>&1; then candidates=("$(conda info --base)/envs/arbooth-vico/bin/python" "${candidates[@]}"); fi
  for candidate in "${candidates[@]}"; do
    if [ -x "$candidate" ]; then ARBOOTH_PYTHON="$candidate"; break; fi
  done
fi
[ -x "${ARBOOTH_PYTHON:-}" ] || { echo "Missing ARBooth Python. Follow TIMING_BENCHMARK.md first." >&2; exit 1; }
"$ARBOOTH_PYTHON" benchmark/benchmark_arbooth_93.py \
  --arbooth-root "${ARBOOTH_ROOT:-/workspace/ARBooth}" \
  --subjects "${SUBJECTS:-brown_dog,black_cat,alarm_clock}" \
  --data-dir "${VICO_DIR:-/workspace/CSD-VAR/benchmark_data/vico_csd}" \
  --checkpoint-root "${ARBOOTH_CHECKPOINT_ROOT:-weights/arbooth/by_subject}" \
  --vae-path "${VAE_PATH:-weights/infinity_vae_d32reg.pth}" \
  --text-encoder-ckpt "${TEXT_ENCODER:-weights/flan-t5-xl}" \
  --output "${OUTPUT:-output/timing/arbooth_3subjects_93}" \
  --seed "${SEED:-0}"
