#!/usr/bin/env bash
# Sequentially run and independently evaluate all three five-seed models.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

INFERENCE_SEEDS="${INFERENCE_SEEDS:-0,1,2,3,4}" \
IMAGES_PER_PROMPT=5 \
RESULTS="${CSD_RESULTS:-benchmark_outputs_vico_csd_var_5seeds}" \
WORK_DIR="${CSD_WORK_DIR:-benchmark_work/vico_csd_var_5seeds}" \
REPORT_DIR="${CSD_REPORT_DIR:-benchmark_reports_vico_csd_var_5seeds}" \
bash scripts/run_csd_var_vico.sh

OUTPUT="${CUSTOM_RESULTS:-benchmark_outputs_vico_custom_diffusion_5seeds}" \
REPORT_DIR="${CUSTOM_REPORT_DIR:-benchmark_reports_vico_custom_diffusion_5seeds}" \
bash scripts/run_custom_diffusion_vico_5seeds.sh

OUTPUT="${DREAMMATCHER_RESULTS:-benchmark_outputs_vico_dreammatcher_custom_diffusion_5seeds}" \
REPORT_DIR="${DREAMMATCHER_REPORT_DIR:-benchmark_reports_vico_dreammatcher_custom_diffusion_5seeds}" \
bash scripts/run_dreammatcher_custom_diffusion_vico_5seeds.sh
