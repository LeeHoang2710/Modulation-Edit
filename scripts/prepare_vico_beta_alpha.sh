#!/usr/bin/env bash
# Prepare and validate everything required by the automatic beta-then-alpha
# ViCo ablation. Safe to rerun after an interrupted installation/download.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

echo "[1/4] Installing or repairing the Infinity environment"
bash scripts/setup_venv.sh

echo "[2/4] Downloading or repairing Infinity and FLAN-T5 weights"
USE_ACTIVE_PYTHON=1 PYTHON_BIN=.venv/bin/python \
  bash scripts/download_vastai_weights.sh

echo "[3/4] Preparing ViCo data and the ARBooth/CSD evaluation environment"
bash scripts/prepare_vico_ablation_assets.sh

echo "[4/4] Validating the complete pipeline"
.venv/bin/python - <<'PY'
import csv
import json
from pathlib import Path

import cv2
import flash_attn
import torch
import tqdm
from transformers import AutoConfig, AutoTokenizer

root = Path.cwd()
weights = root / "weights"
required = [
    weights / "infinity_2b_reg.pth",
    weights / "infinity_vae_d32reg.pth",
    weights / "flan-t5-xl/config.json",
    weights / "flan-t5-xl/tokenizer_config.json",
    weights / "flan-t5-xl/spiece.model",
    weights / "flan-t5-xl/model.safetensors.index.json",
]
missing = [str(path) for path in required if not path.is_file()]
if missing:
    raise SystemExit("missing required weights:\n  " + "\n  ".join(missing))

index_path = weights / "flan-t5-xl/model.safetensors.index.json"
index = json.loads(index_path.read_text())
shards = sorted(set(index.get("weight_map", {}).values()))
if not shards:
    raise SystemExit(f"no model shards listed in {index_path}")
missing_shards = [name for name in shards if not (index_path.parent / name).is_file()]
if missing_shards:
    raise SystemExit("missing FLAN-T5 shards:\n  " + "\n  ".join(missing_shards))
AutoConfig.from_pretrained(index_path.parent, local_files_only=True)
AutoTokenizer.from_pretrained(index_path.parent, local_files_only=True, legacy=True)

data = Path("/workspace/CSD-VAR/benchmark_data/vico_csd")
manifest = data / "subjects.csv"
if not manifest.is_file():
    raise SystemExit(f"missing ViCo manifest: {manifest}")
with manifest.open() as handle:
    subjects = list(csv.DictReader(handle))
if len(subjects) != 16:
    raise SystemExit(f"expected 16 ViCo subjects, found {len(subjects)}")
for item in subjects:
    images = data / f"{item['subject']}+photo/imgs"
    if not images.is_dir() or not any(images.iterdir()):
        raise SystemExit(f"missing reference images: {images}")
    prompt_file = data / item["prompts"]
    prompts = [line for line in prompt_file.read_text().splitlines() if line.strip()]
    if len(prompts) != 31:
        raise SystemExit(f"expected 31 prompts in {prompt_file}, found {len(prompts)}")

if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available in .venv")
print(
    f"[ready] torch={torch.__version__} cuda={torch.version.cuda} "
    f"gpu={torch.cuda.get_device_name(0)} cv2={cv2.__version__} "
    f"flash_attn={flash_attn.__version__} tqdm={tqdm.__version__}"
)
print(f"[ready] FLAN-T5 shards={len(shards)}; ViCo subjects={len(subjects)}; prompts/subject=31")
PY

if command -v conda >/dev/null 2>&1; then
  CONDA_BASE="$(conda info --base)"
else
  CONDA_BASE=/root/miniconda3
fi
EVAL_PYTHON="${EVAL_PYTHON:-${CONDA_BASE}/envs/csd-var/bin/python}"
if [[ ! -x "$EVAL_PYTHON" ]]; then
  echo "Missing evaluator Python: $EVAL_PYTHON" >&2
  exit 1
fi
"$EVAL_PYTHON" - <<'PY'
import sys
from pathlib import Path

arbooth = Path("/workspace/ARBooth")
csd = Path("/workspace/CSD-VAR")
sys.path[:0] = [str(arbooth), str(csd)]

import clip
import lpips
import torch
from booth_evaluation.clip_eval import CLIPEvaluator
from booth_evaluation.dino_eval import DINOEvaluator
from booth_evaluation.lpips_eval import LPIPSEvaluator
from vis.metrics import CSDEvaluator

checkpoint = csd / "evaluator/CSD/csd_models/pytorch_model.bin"
if not checkpoint.is_file():
    raise SystemExit(f"missing CSD checkpoint: {checkpoint}")
print(f"[ready] evaluator Python={sys.executable}; torch={torch.__version__}; CSD={checkpoint}")
PY

echo "[ready] The beta-then-alpha ViCo pipeline is fully prepared."
echo "Smoke test:"
echo "  SUBJECTS=black_cat NUM_PROMPTS=5 NUM_SAMPLES=1 bash ablation/05_subject_prompt_alpha_beta.sh"
