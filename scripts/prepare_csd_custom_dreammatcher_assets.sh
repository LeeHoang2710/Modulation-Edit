#!/usr/bin/env bash
# Download ViCo, CSD-VAR assets, SD-v1.4, and all released Custom Diffusion deltas.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

bash scripts/prepare_csd_var_vico_assets.sh

ENV_NAME="${PERSONALIZATION_ENV_NAME:-benchmark-personalization}"
PY=(conda run --no-capture-output -n "$ENV_NAME" python)
WEIGHTS="${CUSTOM_DIFFUSION_WEIGHTS:-weights/custom_diffusion}"
DRIVE_URL="${CUSTOM_DIFFUSION_DRIVE_URL:-https://drive.google.com/drive/folders/1BvkCAAkPVRRy0uFY4O_Bzr1lO-uP-TqH}"
mkdir -p "$WEIGHTS"
"${PY[@]}" - "$DRIVE_URL" "$WEIGHTS" <<'PY'
import sys
from pathlib import Path
import gdown

url, destination = sys.argv[1], Path(sys.argv[2]).resolve()
subjects = {
    "alarm_clock", "black_cat", "brown_cat", "brown_dog", "brown_dog2",
    "brown_teddybear", "cat_statue", "duck_toy", "elephant_statue", "fat_dog",
    "monster_toy", "pink_sunglasses", "red_teapot", "red_vase",
    "tortoise_plushy", "wooden_barn",
}
missing = [name for name in subjects if not (destination / name / "delta.bin").is_file()]
if missing:
    gdown.download_folder(url=url, output=str(destination), quiet=False, use_cookies=False, remaining_ok=True)
missing = [name for name in subjects if not (destination / name / "delta.bin").is_file()]
if missing:
    raise SystemExit(f"Custom Diffusion checkpoints are incomplete: {missing}")
print(f"[ready] {len(subjects)} Custom Diffusion checkpoints in {destination}")
PY

"${PY[@]}" - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download(repo_id="CompVis/stable-diffusion-v1-4")
print("[ready] CompVis/stable-diffusion-v1-4 cached")
PY

test -f models/DreamMatcher/inputs/imgs_source.txt
echo "[ready] all CSD-VAR, Custom Diffusion, DreamMatcher, ViCo, and metric assets"
