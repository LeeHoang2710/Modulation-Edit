#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
if [ -z "${CSD_PYTHON:-}" ]; then
  candidates=(
    /venv/benchmark-csdvar/bin/python
    /venv/csd-var/bin/python
    /root/miniconda3/envs/benchmark-csdvar/bin/python
    /root/miniconda3/envs/csd-var/bin/python
    /opt/conda/envs/benchmark-csdvar/bin/python
    /opt/conda/envs/csd-var/bin/python
    /opt/miniforge3/envs/benchmark-csdvar/bin/python
    /opt/miniforge3/envs/csd-var/bin/python
  )
  if command -v conda >/dev/null 2>&1; then
    conda_base="$(conda info --base)"
    candidates=(
      "$conda_base/envs/benchmark-csdvar/bin/python"
      "$conda_base/envs/csd-var/bin/python"
      "${candidates[@]}"
    )
  fi
  for candidate in "${candidates[@]}"; do
    if [ -x "$candidate" ]; then CSD_PYTHON="$candidate"; break; fi
  done
fi
[ -x "${CSD_PYTHON:-}" ] || { echo "Missing CSD-VAR Python. Follow TIMING_BENCHMARK.md." >&2; exit 1; }
"$CSD_PYTHON" -c 'import torch, transformers, diffusers, safetensors; assert torch.cuda.is_available()' || {
  echo "Detected CSD-VAR Python is incomplete: $CSD_PYTHON" >&2
  echo "Follow the CSD-VAR environment section in TIMING_BENCHMARK.md." >&2
  exit 1
}
echo "[environment] CSD_PYTHON=$CSD_PYTHON"
"$CSD_PYTHON" benchmark/benchmark_csd_var_93.py \
  --csd-root "${CSD_ROOT:-/workspace/CSD-VAR}" \
  --data-dir "${VICO_DIR:-/workspace/CSD-VAR/benchmark_data/vico_csd}" \
  --subjects "${SUBJECTS:-brown_dog,black_cat,alarm_clock}" \
  --seed "${SEED:-0}" --steps "${STEPS:-200}" \
  --checkpoint-step "${CHECKPOINT_STEP:-150}" --save-every "${SAVE_EVERY:-50}" \
  --work-dir "${WORK_DIR:-output/timing/csd_var_3subjects_93/work}" \
  --output "${OUTPUT:-output/timing/csd_var_3subjects_93/images}"
