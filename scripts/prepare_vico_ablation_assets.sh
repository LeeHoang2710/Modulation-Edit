#!/usr/bin/env bash
# Prepare the official ARBooth ViCo inputs and CSD-VAR metric environment used
# by ablation/eval_vico.sh.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKSPACE="${WORKSPACE:-/workspace}"
ARBOOTH_ROOT="${ARBOOTH_ROOT:-${WORKSPACE}/ARBooth}"
CSD_ROOT="${CSD_ROOT:-${WORKSPACE}/CSD-VAR}"
VICO_DIR="${VICO_DIR:-${CSD_ROOT}/benchmark_data/vico_csd}"
EVAL_ENV="${EVAL_ENV:-csd-var}"
CUDA_WHL="${CUDA_WHL:-cu121}"

command -v git >/dev/null 2>&1 || { echo "git is required" >&2; exit 1; }
if ! command -v conda >/dev/null 2>&1; then
    INSTALLER="${WORKSPACE}/miniconda.sh"
    if command -v wget >/dev/null 2>&1; then
        wget -O "${INSTALLER}" https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
    else
        curl -fL -o "${INSTALLER}" https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
    fi
    bash "${INSTALLER}" -b -p /root/miniconda3
    export PATH="/root/miniconda3/bin:${PATH}"
fi

mkdir -p "${WORKSPACE}"
if [[ ! -d "${ARBOOTH_ROOT}/.git" ]]; then
    git clone https://github.com/jiwoogit/ARBooth.git "${ARBOOTH_ROOT}"
fi
if [[ ! -d "${CSD_ROOT}/.git" ]]; then
    git clone https://github.com/visual-ar-group/CSD-VAR.git "${CSD_ROOT}"
fi

CONDA_BASE="$(conda info --base)"
# shellcheck disable=SC1091
source "${CONDA_BASE}/etc/profile.d/conda.sh"
if ! conda env list | awk '{print $1}' | grep -Fxq "${EVAL_ENV}"; then
    conda create -y -n "${EVAL_ENV}" python=3.10 pip
fi

PY=(conda run --no-capture-output -n "${EVAL_ENV}" python)
"${PY[@]}" -m pip install --upgrade 'pip<24.1' setuptools wheel
"${PY[@]}" -m pip install \
    torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 \
    --index-url "https://download.pytorch.org/whl/${CUDA_WHL}"
"${PY[@]}" -m pip install -r "${CSD_ROOT}/requirements.txt"
"${PY[@]}" -m pip install 'numpy<2' lpips opencv-python-headless
"${PY[@]}" -m pip install -e "${CSD_ROOT}/evaluator/CSD"

mkdir -p "${CSD_ROOT}/evaluator/CSD/csd_models"
"${PY[@]}" - "${CSD_ROOT}/evaluator/CSD/csd_models" <<'PY'
import sys
from huggingface_hub import hf_hub_download

hf_hub_download(
    repo_id="tomg-group-umd/CSD-ViT-L",
    filename="pytorch_model.bin",
    local_dir=sys.argv[1],
)
PY

"${PY[@]}" - "${ARBOOTH_ROOT}/inputs" "${VICO_DIR}" <<'PY'
import csv
import os
import shutil
import sys
from pathlib import Path

inputs = Path(sys.argv[1]).resolve()
out = Path(sys.argv[2]).resolve()
if not inputs.is_dir():
    raise SystemExit(f"ARBooth ViCo inputs are missing: {inputs}")

live = {"black_cat", "brown_cat", "brown_dog", "brown_dog2", "fat_dog"}
classes = {
    "alarm_clock": "clock",
    "black_cat": "cat",
    "brown_cat": "cat",
    "brown_dog": "dog",
    "brown_dog2": "dog",
    "brown_teddybear": "teddybear",
    "cat_statue": "cat statue",
    "duck_toy": "duck toy",
    "elephant_statue": "elephant statue",
    "fat_dog": "dog",
    "monster_toy": "toy",
    "pink_sunglasses": "sunglasses",
    "red_teapot": "teapot",
    "red_vase": "vase",
    "tortoise_plushy": "plushy",
    "wooden_barn": "barn",
}

out.mkdir(parents=True, exist_ok=True)
for name in ("prompts_live_objects.txt", "prompts_nonlive_objects.txt"):
    source = inputs / name
    if not source.is_file():
        raise SystemExit(f"missing ARBooth prompt file: {source}")
    shutil.copy2(source, out / name)

rows = []
for subject, class_name in classes.items():
    source = inputs / subject
    if not source.is_dir():
        raise SystemExit(f"missing ViCo subject: {source}")
    image_dir = out / f"{subject}+photo" / "imgs"
    image_dir.parent.mkdir(parents=True, exist_ok=True)
    if image_dir.is_symlink() or image_dir.exists():
        if image_dir.is_symlink() and image_dir.resolve() == source:
            pass
        elif image_dir.is_dir() and not image_dir.is_symlink():
            shutil.rmtree(image_dir)
            image_dir.symlink_to(source, target_is_directory=True)
        else:
            image_dir.unlink()
            image_dir.symlink_to(source, target_is_directory=True)
    else:
        image_dir.symlink_to(source, target_is_directory=True)
    prompt_file = "prompts_live_objects.txt" if subject in live else "prompts_nonlive_objects.txt"
    rows.append({
        "concept": "live" if subject in live else "object",
        "subject": subject,
        "class_name": class_name,
        "prompts": prompt_file,
    })

with (out / "subjects.csv").open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=["concept", "subject", "class_name", "prompts"])
    writer.writeheader()
    writer.writerows(rows)

print(f"[ready] {out}: {len(rows)} subjects")
PY

test -f "${VICO_DIR}/subjects.csv"
test -f "${VICO_DIR}/prompts_live_objects.txt"
test -f "${VICO_DIR}/prompts_nonlive_objects.txt"
test -f "${CSD_ROOT}/evaluator/CSD/csd_models/pytorch_model.bin"
test -d "${ARBOOTH_ROOT}/booth_evaluation"

echo "[ready] ARBooth: ${ARBOOTH_ROOT}"
echo "[ready] CSD-VAR: ${CSD_ROOT}"
echo "[ready] ViCo benchmark: ${VICO_DIR}"
echo "[ready] evaluator Python: ${CONDA_BASE}/envs/${EVAL_ENV}/bin/python"
