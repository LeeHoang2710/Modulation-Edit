#!/usr/bin/env bash
# Fast fail before any long optimization/generation benchmark begins.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

WORKSPACE="${WORKSPACE:-/workspace}"
ARBOOTH_ROOT="${ARBOOTH_ROOT:-$WORKSPACE/ARBooth}"
CSD_ROOT="${CSD_ROOT:-$WORKSPACE/CSD-VAR}"
VICO_DIR="${VICO_DIR:-$CSD_ROOT/benchmark_data/vico_csd}"

find_python() {
  local explicit="$1"; shift
  if [ -n "$explicit" ] && [ -x "$explicit" ]; then printf '%s\n' "$explicit"; return; fi
  local candidate
  for candidate in "$@"; do
    if [ -x "$candidate" ]; then printf '%s\n' "$candidate"; return; fi
  done
  return 1
}

conda_base=""
if command -v conda >/dev/null 2>&1; then conda_base="$(conda info --base)"; fi
OURS_PYTHON="$(find_python "${OURS_PYTHON:-}" .venv/bin/python)" || { echo "Missing .venv. Run scripts/prepare_three_model_timing.sh" >&2; exit 1; }
ARBOOTH_PYTHON="$(find_python "${ARBOOTH_PYTHON:-}" \
  /venv/arbooth-vico/bin/python "${conda_base:+$conda_base/envs/arbooth-vico/bin/python}" \
  /opt/miniforge3/envs/arbooth-vico/bin/python /opt/conda/envs/arbooth-vico/bin/python \
  /root/miniconda3/envs/arbooth-vico/bin/python)" || {
  echo "Missing arbooth-vico environment. Run scripts/prepare_three_model_timing.sh" >&2; exit 1;
}
CSD_PYTHON="$(find_python "${CSD_PYTHON:-}" \
  /venv/csd-var/bin/python /venv/benchmark-csdvar/bin/python \
  "${conda_base:+$conda_base/envs/csd-var/bin/python}" "${conda_base:+$conda_base/envs/benchmark-csdvar/bin/python}" \
  /opt/miniforge3/envs/csd-var/bin/python /opt/conda/envs/csd-var/bin/python \
  /root/miniconda3/envs/csd-var/bin/python)" || {
  echo "Missing CSD-VAR environment. Run scripts/prepare_three_model_timing.sh" >&2; exit 1;
}

required=(
  weights/infinity_2b_reg.pth weights/infinity_vae_d32reg.pth
  weights/flan-t5-xl/config.json weights/flan-t5-xl/tokenizer_config.json
  "$ARBOOTH_ROOT/tools/run_arbooth_for_eval.py" "$CSD_ROOT/train.py" "$CSD_ROOT/inference.py"
  "$VICO_DIR/subjects.csv" "$VICO_DIR/prompts_live_objects.txt" "$VICO_DIR/prompts_nonlive_objects.txt"
)
for path in "${required[@]}"; do [ -e "$path" ] || { echo "Missing required asset: $path" >&2; exit 1; }; done
for subject in brown_dog black_cat alarm_clock; do
  checkpoint="weights/arbooth/by_subject/$subject/$subject.pth"
  [ -f "$checkpoint" ] || { echo "Missing ARBooth checkpoint: $checkpoint" >&2; exit 1; }
done

"$OURS_PYTHON" -c 'import cv2, flash_attn, torch, transformers; from transformers import T5EncoderModel; assert torch.cuda.is_available(); print("ours", torch.__version__, torch.cuda.get_device_name(0))'
PYTHONPATH="$ARBOOTH_ROOT${PYTHONPATH:+:$PYTHONPATH}" "$ARBOOTH_PYTHON" -c 'import cv2, flash_attn, inspect, torch, transformers; from torch.nn.attention.flex_attention import flex_attention; from tools.run_arbooth_for_eval import encode_prompt, gen_one_img; from transformers import T5EncoderModel; assert "text_cond_tuple" in inspect.signature(gen_one_img).parameters; assert torch.cuda.is_available(); print("arbooth", torch.__version__, torch.cuda.get_device_name(0))'
PYTHONPATH="$CSD_ROOT${PYTHONPATH:+:$PYTHONPATH}" "$CSD_PYTHON" -c 'import diffusers, safetensors, torch, transformers; assert torch.cuda.is_available(); print("csd-var", torch.__version__, torch.cuda.get_device_name(0))'

# Fail now if a future CSD-VAR source update no longer matches the safe,
# upstream-preserving instrumentation used by the benchmark.
"$OURS_PYTHON" - "$CSD_ROOT" <<'PY'
import sys, tempfile
from pathlib import Path
from benchmark.benchmark_csd_var_93 import create_timed_inference, create_timed_training
source = Path(sys.argv[1])
with tempfile.TemporaryDirectory() as directory:
    out = Path(directory)
    create_timed_training(source / "train.py", out / "train.py")
    create_timed_inference(source / "inference.py", out / "inference.py")
print("CSD-VAR timing instrumentation: compatible")
PY

"$OURS_PYTHON" - "$VICO_DIR" <<'PY'
import csv, sys
from pathlib import Path
root = Path(sys.argv[1])
with (root / "subjects.csv").open() as handle:
    rows = list(csv.DictReader(handle))
assert len(rows) == 16, f"expected 16 subjects, found {len(rows)}"
for row in rows:
    refs = root / f"{row['subject']}+photo" / "imgs"
    assert refs.is_dir() and any(refs.iterdir()), f"missing references: {refs}"
    prompts = [x for x in (root / row["prompts"]).read_text().splitlines() if x.strip()]
    assert len(prompts) == 31, f"{row['subject']}: expected 31 prompts, found {len(prompts)}"
print(f"ViCo data: {len(rows)} subjects x 31 prompts")
PY

echo "[ready] All three timing pipelines passed preflight."
echo "[ready] OURS_PYTHON=$OURS_PYTHON"
echo "[ready] ARBOOTH_PYTHON=$ARBOOTH_PYTHON"
echo "[ready] CSD_PYTHON=$CSD_PYTHON"
