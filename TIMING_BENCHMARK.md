# Three-model preparation and generation timing benchmark

The commands below assume a Vast.ai CUDA development image and `/workspace` as
the persistent workspace.

## Recommended path (fresh Vast.ai instance)

After cloning this branch, this single command installs or repairs all three
environments, downloads the required assets, and runs a strict preflight:

```bash
cd /workspace/Modulation-Edit
export HF_TOKEN='YOUR_HUGGINGFACE_READ_TOKEN'
bash scripts/prepare_three_model_timing.sh
```

Do not start a long run unless it ends with:

```text
[ready] All three timing pipelines passed preflight.
```

Run all three benchmarks into a fresh timestamped directory:

```bash
bash scripts/run_three_model_timing.sh
```

The run finishes by writing `<timestamp>/timing_summary.csv`, while retaining
all per-step, per-image, per-subject, warm-up, setup and I/O diagnostics in the
method-specific JSON reports.

The sections below retain the individual commands for rerunning one method.

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
bash scripts/preflight_three_model_timing.sh
```

## 5. ARBooth: three objects, 31 prompts, one seed

```bash
bash scripts/run_arbooth_inference_timing_93.sh
```

The default subjects are `brown_dog`, `black_cat`, and `alarm_clock`. This
generates exactly `3 × 31 × 1 = 93` images with fixed seed 0. One untimed
warm-up runs after each subject checkpoint is loaded. T5 prompt encoding is
reported separately. The primary CUDA-synchronized generation number includes
only autoregressive generation and VAE decoding; checkpoint loading, prompt
encoding, device-to-host transfer, and JPEG writing are excluded.

The runner reads `subjects.csv` and the two official 31-prompt files directly
from `/workspace/CSD-VAR/benchmark_data/vico_csd`; this branch does not require
`benchmark/manifests/vico.json`.

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

Results are written to
`output/ablation_vico/24_training_speed_black_cat/timing_summary.csv`. The
primary `preparation_seconds`/`pure_optimization_seconds` metric is the exact
CUDA-synchronized optimizer loop. Model loading, text encoding, reference
preprocessing, one-time VAE feature caching, and checkpoint I/O are excluded.
Generation runs one untimed warm-up and reports mean/std over 100 images.
After generation, the saved images from all three configurations are evaluated
with the same ARBooth-compatible DINO, CLIP-I, and CLIP-T implementation. The
quality-only and combined reports are written to `metrics_summary.csv` and
`timing_metrics_summary.csv`. Metric evaluation is outside every timing region.

To evaluate an already completed timing run without regenerating images:

```bash
SUBJECT=black_cat bash ablation/evaluate_training_timing_metrics.sh \
  output/timing/YOUR_RUN_ID/ours
```

The old end-to-end number remains as `process_wall_seconds_diagnostic`.
`feature_cache_seconds` and `checkpoint_io_seconds` expose overhead, while
`preparation_seconds_to_materialized_offset` answers the deployment question
“how long until the reusable offset file exists?” without conflating that with
the pure optimization metric.

To use another object or inference batch:

```bash
SUBJECT=alarm_clock INFER_BATCH=1 bash ablation/08_training_speed_benchmark.sh
```

## 7. CSD-VAR: preparation and 93-image inference timing

If `scripts/prepare_vico_beta_alpha.sh` already created a working `csd-var`
environment, the runner detects and reuses it automatically. Verify it with:

```bash
bash scripts/preflight_three_model_timing.sh
```

If preflight passes, skip directly to the run command below. Otherwise, create
the dedicated environment as follows.

Create an isolated CSD-VAR environment and install the released requirements:

```bash
cd /workspace/Modulation-Edit
CONDA_BASE="$(conda info --base)"
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda create -y -n benchmark-csdvar python=3.10 pip
conda run -n benchmark-csdvar python -m pip install --upgrade 'pip<24.1' setuptools wheel
conda run -n benchmark-csdvar python -m pip install \
  torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 \
  --index-url https://download.pytorch.org/whl/cu121
conda run -n benchmark-csdvar python -m pip install -r /workspace/CSD-VAR/requirements.txt
conda run -n benchmark-csdvar python -m pip install 'numpy<2'
```

Download/cache the Switti and VQ-VAE weights before timing:

```bash
conda run -n benchmark-csdvar python -c \
  'from huggingface_hub import snapshot_download; [snapshot_download(repo_id=x) for x in ("yresearch/Switti-AR", "yresearch/VQVAE-Switti", "openai/clip-vit-large-patch14", "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k")]'
```

Run three subjects × 31 prompts × one fixed seed:

```bash
bash scripts/run_csd_var_timing_93.sh
```

Outputs and timing are saved under:

```text
output/timing/csd_var_3subjects_93/images/
output/timing/csd_var_3subjects_93/timing.json
```

Preparation fields are reported separately for each subject:

- `pure_optimization_seconds`: sum of CUDA-synchronized `trainer.train_step`
  calls; model/data setup, SVD, logging and checkpoint writes are excluded;
- `preparation_seconds_to_used_checkpoint`: SVD extraction plus training until
  every component of the step-150 checkpoint used for inference is written;
- `preparation_full_200_step_seconds`: SVD extraction plus the complete
  200-step optimization process.

Weight downloads, input staging, and image copying are excluded. The benchmark
creates an instrumented copy of released `inference.py`; upstream source is not
modified. The three text conditions are encoded before the timer.
CUDA-synchronized generation includes autoregressive sampling and VQ-VAE
decoding. Switti/checkpoint loading, prompt encoding, GPU-to-CPU/PIL conversion,
directory creation, JSON/PNG writing, and process startup are excluded. The
full subprocess wall time remains in the report only as a diagnostic.

All three generation reports now use the same timing contract:

```text
include: autoregressive sampling + image decoding + GPU postprocessing
exclude: prompt encoding + frozen-model/checkpoint loading + process setup
         + device-to-host/PIL conversion + image writing
CUDA synchronize immediately before and after the measured region
```

ARBooth uses released pretrained subject checkpoints, so its preparation time
is recorded as unavailable (`null`); checkpoint loading is not mislabeled as
personalization. Our method and CSD-VAR both report the subject-specific
optimization required before reusable inference.
