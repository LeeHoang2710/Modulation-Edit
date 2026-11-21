#!/usr/bin/env bash
# Train exactly two main-variant offsets, then generate one dog-left/cat-right image.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

if [[ -n "${INFINITY_PYTHON:-}" ]]; then
  PYTHON_BIN="$INFINITY_PYTHON"
elif [[ -x /venv/infinity-modulation/bin/python ]]; then
  PYTHON_BIN=/venv/infinity-modulation/bin/python
elif command -v conda >/dev/null 2>&1; then
  CONDA_BASE="$(conda info --base)"
  PYTHON_BIN="$CONDA_BASE/envs/infinity-modulation/bin/python"
else
  PYTHON_BIN=""
fi
[[ -n "$PYTHON_BIN" && -x "$PYTHON_BIN" ]] || {
  echo "Missing infinity-modulation Python. Run: bash scripts/setup_vastai_env.sh" >&2
  exit 1
}

"$PYTHON_BIN" - <<'PY'
import torch

if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable in the infinity-modulation environment")
capability = torch.cuda.get_device_capability()
architectures = torch.cuda.get_arch_list()
print(f"[cuda] gpu={torch.cuda.get_device_name(0)!r} capability={capability} torch={torch.__version__} cuda={torch.version.cuda}")
if capability == (12, 0) and "sm_120" not in architectures:
    raise SystemExit(
        "RTX 5090 requires a PyTorch wheel compiled for sm_120. "
        "Run: bash scripts/repair_rtx5090_env.sh"
    )
print("[cuda] compiled architectures:", architectures)
print("[cuda] smoke:", (torch.ones(1, device="cuda") + 1).item())
PY

STEPS=200
BATCH_SIZE=4
GRAD_ACCUM=1
LR=5e-3
BSC_LAYERS=13
BSC_STRENGTH=0.3
TRAIN_SEED=0
INFER_SEED=0
WORK_ROOT="${WORK_ROOT:-benchmark_work/fat_dog_grey_cat_spatial_test}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/fat_dog_grey_cat_spatial_test}"
REUSE_OFFSETS="${REUSE_OFFSETS:-1}"

MODEL=(
  --pn 1M --model_type infinity_2b
  --model_path weights/infinity/infinity_2b_reg.pth
  --vae_type 32 --vae_path weights/infinity/infinity_vae_d32reg.pth
  --text_encoder_ckpt weights/infinity/flan-t5-xl
  --add_lvl_embeding_only_first_block 1 --use_bit_label 1
  --rope2d_each_sa_layer 1 --rope2d_normalized_by_hw 2
  --use_scale_schedule_embedding 0 --apply_spatial_patchify 0
  --checkpoint_type torch --bf16 1 --text_channels 2048
)

for required in \
  weights/infinity/infinity_2b_reg.pth \
  weights/infinity/infinity_vae_d32reg.pth \
  weights/infinity/flan-t5-xl \
  references/vico_16/fat_dog \
  references/vico_16/black_cat; do
  [[ -e "$required" ]] || { echo "Missing required asset: $required" >&2; exit 1; }
done

REFERENCE_ROOT=references/vico_16
reference_subjects="$(find "$REFERENCE_ROOT" -mindepth 1 -maxdepth 1 -type d | wc -l | tr -d ' ')"
reference_images="$(find "$REFERENCE_ROOT" -mindepth 2 -maxdepth 2 -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' \) | wc -l | tr -d ' ')"
[[ "$reference_subjects" == "16" && "$reference_images" == "94" ]] || {
  echo "Incomplete first-party references: expected 16 folders/94 images, found $reference_subjects/$reference_images" >&2
  exit 1
}
echo "[references] first-party ViCo inventory verified: 16 folders, 94 images"

mkdir -p "$WORK_ROOT" "$OUTPUT_DIR"

train_offset() {
  local subject="$1"
  local prompt="$2"
  local images="$REFERENCE_ROOT/$subject"
  local run_dir="$WORK_ROOT/$subject"
  local offset="$run_dir/offset.pth"
  mkdir -p "$run_dir"

  if [[ "$REUSE_OFFSETS" == "1" && -f "$offset" ]] && \
     "$PYTHON_BIN" -c 'import sys,torch; c=torch.load(sys.argv[1],map_location="cpu"); assert c.get("train_prompt")==sys.argv[2]; assert c.get("iters")==200; assert c.get("batch_size")==4; assert c.get("grad_accum")==1; assert abs(c.get("lr")-5e-3)<1e-12; assert set(c.get("loss_scales",[]))==set(range(4)); assert c.get("noise_apply_layers")==13; assert abs(c.get("noise_apply_strength")-0.3)<1e-12' "$offset" "$prompt"; then
    echo "[reuse] $subject -> $offset"
    return
  fi

  echo "[optimize] subject=$subject prompt='$prompt' steps=$STEPS batch=$BATCH_SIZE grad_accum=$GRAD_ACCUM BSC=$BSC_LAYERS/$BSC_STRENGTH"
  "$PYTHON_BIN" tools/optimize_ada_offset.py \
    --mode optimize --images "$images" "${MODEL[@]}" \
    --train_prompt "$prompt" --seed "$TRAIN_SEED" --hflip 0 \
    --grad_checkpointing 1 --offset_mode per_scale \
    --iters "$STEPS" --batch_size "$BATCH_SIZE" --grad_accum "$GRAD_ACCUM" --lr "$LR" \
    --noise_apply_layers "$BSC_LAYERS" --noise_apply_strength "$BSC_STRENGTH" \
    --loss_scales 0-3 --offset_infer_scales 0-3 \
    --num_samples 1 --sample_every 0 \
    --save_offset "$offset" --save_file "" \
    2>&1 | tee "$run_dir/optimize.log"
}

train_offset fat_dog "a fat dog"
train_offset black_cat "a grey cat"

echo "[infer] prompt CA-KV enabled; dog=left, cat=right; one seed=$INFER_SEED"
"$PYTHON_BIN" tools/infer_two_offset_spatial.py \
  --left-offset-path "$WORK_ROOT/fat_dog/offset.pth" \
  --right-offset-path "$WORK_ROOT/black_cat/offset.pth" \
  --left-base-prompt "a fat dog" --right-base-prompt "a grey cat" \
  --joint-base-prompt "a fat dog and a grey cat" \
  --prompt "a dog next to a cat" --output-dir "$OUTPUT_DIR" \
  "${MODEL[@]}" \
  --alpha-schedule 0,0,0,0 --early-composition preserve-offset \
  --offset-scales 0-3 --left-offset-scale 0.9 --right-offset-scale 0.9 \
  --split-ratio 0.5 --early-cfg 1.5 --late-cfg 1.5 --late-gss prompt \
  --num-samples 1 --seed "$INFER_SEED" \
  2>&1 | tee "$WORK_ROOT/infer.log"

[[ -f "$OUTPUT_DIR/seed_000.png" ]] || { echo "Inference did not produce seed_000.png" >&2; exit 1; }
echo "[complete] $OUTPUT_DIR/seed_000.png"
