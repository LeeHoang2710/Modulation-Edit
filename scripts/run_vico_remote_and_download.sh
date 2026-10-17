#!/usr/bin/env bash
# Run either benchmark protocol on Vast.ai from a Mac, then pull images and
# metric reports into ~/Downloads.  This script is intentionally LOCAL-only.
set -euo pipefail

MODE="${1:-}"
[[ "$MODE" == "arbooth" || "$MODE" == "ours" ]] || {
    echo "Usage: VAST_HOST=... VAST_PORT=... bash scripts/run_vico_remote_and_download.sh {arbooth|ours} [smoke|full]" >&2
    exit 2
}
RUN_KIND="${2:-full}"
[[ "$RUN_KIND" == "smoke" || "$RUN_KIND" == "full" ]] || { echo "Run kind must be smoke or full." >&2; exit 2; }

VAST_HOST="${VAST_HOST:?Set VAST_HOST, for example ssh4.vast.ai}"
VAST_PORT="${VAST_PORT:?Set VAST_PORT, for example 12345}"
VAST_USER="${VAST_USER:-root}"
VAST_REPO_DIR="${VAST_REPO_DIR:-/workspace/Modulation-Edit}"
LOCAL_DOWNLOAD_DIR="${LOCAL_DOWNLOAD_DIR:-$HOME/Downloads/vico_benchmark_${MODE}_${RUN_KIND}}"
SSH=(ssh -p "$VAST_PORT" "${VAST_USER}@${VAST_HOST}")
RSYNC_SSH="ssh -p ${VAST_PORT}"

case "$MODE:$RUN_KIND" in
    arbooth:smoke)
        REMOTE_COMMAND="bash scripts/run_arbooth_vico_smoke.sh"
        REMOTE_OUTPUT="benchmark_outputs_vico_arbooth_smoke benchmark_reports_vico_arbooth_smoke"
        ;;
    arbooth:full)
        REMOTE_COMMAND="bash scripts/run_arbooth_vico_paper_protocol.sh"
        REMOTE_OUTPUT="benchmark_outputs_vico_arbooth benchmark_reports_vico_arbooth"
        ;;
    ours:smoke)
        REMOTE_COMMAND="bash scripts/run_ours_vico_multi_offset_smoke.sh"
        REMOTE_OUTPUT="benchmark_outputs_vico_ours_multi_offset_smoke benchmark_reports_vico_ours_multi_offset_smoke"
        ;;
    ours:full)
        REMOTE_COMMAND="bash scripts/run_ours_vico_multi_offset_fixed_seed.sh"
        REMOTE_OUTPUT="benchmark_outputs_vico_multi_offset8_seed0 benchmark_reports_vico_multi_offset8_seed0"
        ;;
esac

"${SSH[@]}" "cd '$VAST_REPO_DIR' && $REMOTE_COMMAND"
mkdir -p "$LOCAL_DOWNLOAD_DIR"
for remote_path in $REMOTE_OUTPUT; do
    rsync -az --partial --progress -e "$RSYNC_SSH" \
        "${VAST_USER}@${VAST_HOST}:${VAST_REPO_DIR}/${remote_path}/" "$LOCAL_DOWNLOAD_DIR/${remote_path}/"
done
echo "[downloaded] $LOCAL_DOWNLOAD_DIR"
