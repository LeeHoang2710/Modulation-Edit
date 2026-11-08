#!/usr/bin/env bash
# One ViCo subject × two prompts × five independent seeds = 10 images.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

SUBJECT="${SUBJECT:-brown_dog}"
echo "[smoke:DreamMatcher + Custom Diffusion] subject=$SUBJECT prompts=2 seeds=0,1,2,3,4 expected_images=10"

SUBJECTS="$SUBJECT" \
MAX_PROMPTS=2 \
SEEDS=0,1,2,3,4 \
OUTPUT="${OUTPUT:-benchmark_outputs_smoke_dreammatcher_custom_diffusion_${SUBJECT}}" \
REPORT_DIR="${REPORT_DIR:-benchmark_reports_smoke_dreammatcher_custom_diffusion_${SUBJECT}}" \
bash scripts/run_dreammatcher_custom_diffusion_vico_5seeds.sh
