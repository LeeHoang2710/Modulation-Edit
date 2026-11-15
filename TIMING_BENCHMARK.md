# ARBooth and offset-training timing benchmark

The commands below assume a Vast.ai CUDA development image and `/workspace` as
the persistent workspace.

## 1. Clone the timing branch

```bash
cd /workspace
git clone --branch friend_module_space_prompt_offset_sync \
  https://github.com/LeeHoang2710/Modulation-Edit.git
cd Modulation-Edit
```

For an existing clone:

```bash
cd /workspace/Modulation-Edit
git fetch origin
git switch friend_module_space_prompt_offset_sync
git pull --ff-only origin friend_module_space_prompt_offset_sync
```

## 2. Prepare our Infinity environment, weights, and ViCo data

Set a Hugging Face read token if the server is not already authenticated:

```bash
export HF_TOKEN='YOUR_HUGGINGFACE_READ_TOKEN'
```

Then run the existing complete preparation command:

```bash
cd /workspace/Modulation-Edit
bash scripts/prepare_vico_beta_alpha.sh
```

Do not continue until it prints:

```text
[ready] The beta-then-alpha ViCo pipeline is fully prepared.
```

This creates the Infinity environment at `.venv`, downloads Infinity-2B,
Infinity-VAE and FLAN-T5 XL under `weights/`, clones ARBooth and CSD-VAR, and
creates the 16-object ViCo benchmark under
`/workspace/CSD-VAR/benchmark_data/vico_csd`.

## 3. Create the ARBooth inference environment

```bash
cd /workspace/Modulation-Edit
CONDA_BASE="$(conda info --base)"
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda create -y -n arbooth-vico python=3.10 pip
conda run -n arbooth-vico python -m pip install --upgrade 'pip<24.1' setuptools wheel
conda run -n arbooth-vico python -m pip install \
  torch==2.5.1 torchvision==0.20.1 \
  --index-url https://download.pytorch.org/whl/cu121
```

Install ARBooth requirements after removing its FlashAttention line. Torch must
exist before FlashAttention is installed:

```bash
sed -E \
  -e '/^[[:space:]]*flash[_-]attn([[:space:]]|$)/d' \
  -e 's|^[[:space:]]*httpx==0\.20\.0[[:space:]]*$|httpx==0.28.1|' \
  /workspace/ARBooth/requirements.txt > /tmp/arbooth-requirements.txt
conda run -n arbooth-vico python -m pip install -r /tmp/arbooth-requirements.txt
conda run -n arbooth-vico python -m pip install ninja packaging psutil
conda run -n arbooth-vico python -m pip install \
  'https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.0.post2/flash_attn-2.7.0.post2+cu12torch2.5cxx11abiFALSE-cp310-cp310-linux_x86_64.whl'
conda run -n arbooth-vico python -m pip install \
  'transformers==4.46.3' 'numpy<2' opencv-python-headless
```

Verify the environment:

```bash
conda run -n arbooth-vico python -c \
  'import torch, flash_attn, transformers; print(torch.__version__, torch.cuda.is_available())'
```

The final value must be `True`.

## 4. Download the three released ARBooth checkpoints

The timing test only requires `brown_dog`, `black_cat`, and `alarm_clock`, not
all 16 released checkpoints:

```bash
cd /workspace/Modulation-Edit
mkdir -p weights/arbooth/released
.venv/bin/hf download wldn0202/ARBooth \
  --include 'ckpts/brown_dog.pth' \
            'ckpts/black_cat.pth' \
            'ckpts/alarm_clock.pth' \
  --local-dir weights/arbooth/released \
  --token "$HF_TOKEN"
```

Create the directory layout expected by the released ARBooth evaluator:

```bash
for subject in brown_dog black_cat alarm_clock; do
  mkdir -p "weights/arbooth/by_subject/$subject"
  ln -sfn "$(pwd)/weights/arbooth/released/ckpts/$subject.pth" \
    "weights/arbooth/by_subject/$subject/$subject.pth"
done
```

Confirm all required files:

```bash
test -x .venv/bin/python
test -x "$(conda info --base)/envs/arbooth-vico/bin/python"
test -f weights/infinity_2b_reg.pth
test -f weights/infinity_vae_d32reg.pth
test -d weights/flan-t5-xl
find weights/arbooth/by_subject -name '*.pth' -type l | sort
```

## 5. ARBooth: three objects, 31 prompts, one seed

```bash
bash scripts/run_arbooth_inference_timing_93.sh
```

The default subjects are `brown_dog`, `black_cat`, and `alarm_clock`. This generates exactly `3 × 31 × 1 = 93` images with fixed seed 0. `output/timing/arbooth_3subjects_93/timing.json` reports CUDA-synchronized overall and per-subject timing; checkpoint loading and JPEG writing are excluded.

Override the three subjects or seed if needed:

```bash
SUBJECTS=brown_dog,cat_statue,tortoise_plushy SEED=0 \
  bash scripts/run_arbooth_inference_timing_93.sh
```

## 6. Our method: one subject, three optimization settings

```bash
SUBJECT=black_cat bash ablation/08_training_speed_benchmark.sh
```

This runs the same subject, optimization seed, training prompt, loss, learning rate, and 100-image inference workload for:

1. batch 2, gradient accumulation 2, 200 optimizer steps;
2. batch 4, gradient accumulation 1, 200 optimizer steps;
3. batch 4, gradient accumulation 2, 100 optimizer steps.

Results are written to `output/ablation_vico/24_training_speed_black_cat/timing_summary.csv`. Preparation excludes checkpoint-writing time. Inference excludes model loading and records mean/std over 100 generated images for each configuration.

To use another object or inference batch:

```bash
SUBJECT=alarm_clock INFER_BATCH=1 bash ablation/08_training_speed_benchmark.sh
```
