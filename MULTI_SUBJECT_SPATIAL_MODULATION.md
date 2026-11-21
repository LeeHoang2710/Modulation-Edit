# Fat-dog + grey-cat spatial modulation test on Vast.ai

This recipe trains exactly two class-conditioned SharedAdaLN offsets and
generates one test image:

- left region: the `fat_dog` reference object, conditioned as `a fat dog`;
- right region: the `black_cat` reference object, conditioned as `a grey cat`;
- joint prompt: `a dog next to a cat`;
- one inference seed (`0`).

The fixed optimization configuration is 200 steps, batch size 4, gradient
accumulation 1, learning rate `5e-3`, and per-scale offsets on scales 0-3.
Bitwise labels and bitwise self-correction are enabled. BSC is requested for
13 history layers with maximum bit-flip probability 0.3; because this run
forwards only scales 0-3, the optimizer correctly caps corruption to the three
available preceding histories. Inference uses offset strength 0.9 and CFG 1.5.
Conditional cross-attention K/V comes from the joint prompt at every scale.

## 1. Create the Vast.ai machine

Choose a CUDA **development** image with enough VRAM for Infinity-2B. The image
must include an NVIDIA driver and CUDA compiler. After opening the Vast.ai
terminal, verify:

```bash
nvidia-smi
nvcc --version
```

Do not use a runtime-only CUDA image: FlashAttention installation requires
`nvcc`. A high-memory GPU is recommended because this test uses a true
micro-batch of four 1M-resolution samples.

## 2. Clone this feature branch

The branch must be pushed to GitHub before running these commands on a fresh
instance.

```bash
cd /workspace
git clone --recurse-submodules \
  --branch feature/multi-subject-spatial-modulation \
  --single-branch \
  https://github.com/LeeHoang2710/Modulation-Edit.git
cd Modulation-Edit
git submodule update --init --recursive
git branch --show-current
```

The last command must print:

```text
feature/multi-subject-spatial-modulation
```

## 3. Install Miniconda if necessary

Skip this section if `conda --version` already works.

```bash
cd /workspace
curl -L -o miniconda.sh \
  https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash miniconda.sh -b -p /workspace/miniconda3
source /workspace/miniconda3/etc/profile.d/conda.sh
conda init bash
cd /workspace/Modulation-Edit
```

If you start a new terminal afterward:

```bash
source /workspace/miniconda3/etc/profile.d/conda.sh
cd /workspace/Modulation-Edit
```

## 4. Install the project environment

The setup script installs CUDA PyTorch 2.5.1, FlashAttention, and the remaining
Infinity dependencies into the `vico-modulation` Conda environment.

```bash
bash scripts/setup_vico_vastai.sh
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate vico-modulation
python - <<'PY'
import torch
import flash_attn
print("torch:", torch.__version__)
print("flash-attn:", flash_attn.__version__)
print("CUDA available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0))
PY
```

Do not continue unless `CUDA available` is `True`.

## 5. Download the Infinity weights

The experiment expects the model, VAE, and T5 encoder under
`weights/infinity/`. Set `HF_TOKEN` first only if Hugging Face requests one.

```bash
# Optional for gated/authenticated downloads:
# export HF_TOKEN='hf_...'

INFINITY_CONDA_ENV=vico-modulation \
  bash scripts/download_vastai_weights.sh weights/infinity
```

Verify the expected assets:

```bash
ls -lh \
  weights/infinity/infinity_2b_reg.pth \
  weights/infinity/infinity_vae_d32reg.pth \
  weights/infinity/flan-t5-xl/config.json
```

## 6. Run the complete one-image experiment

```bash
cd /workspace/Modulation-Edit
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate vico-modulation
bash scripts/run_fat_dog_grey_cat_spatial_test.sh
```

The runner performs these operations in order:

1. optimizes the fat-dog offset for 200 steps;
2. optimizes the grey-cat offset for 200 steps;
3. constructs left/right token-wise transformations using all six SharedAdaLN
   vectors;
4. runs global self-attention over both regions;
5. keeps prompt CA-KV enabled at every scale;
6. writes exactly one generated image.

The final result is:

```text
outputs/fat_dog_grey_cat_spatial_test/seed_000.png
```

Offsets and logs are saved under:

```text
benchmark_work/fat_dog_grey_cat_spatial_test/
├── fat_dog/offset.pth
├── fat_dog/optimize.log
├── black_cat/offset.pth
├── black_cat/optimize.log
└── infer.log
```

The output metadata records the active scales, split ratio, CFG schedule, and
the CA-KV sources:

```text
outputs/fat_dog_grey_cat_spatial_test/metadata.json
```

The script reuses offsets only when their stored configuration matches this
experiment exactly. To retrain both offsets from scratch without deleting
anything, choose new output directories:

```bash
REUSE_OFFSETS=0 \
WORK_ROOT=benchmark_work/fat_dog_grey_cat_spatial_test_rerun \
OUTPUT_DIR=outputs/fat_dog_grey_cat_spatial_test_rerun \
  bash scripts/run_fat_dog_grey_cat_spatial_test.sh
```

## 7. Download the result to your computer

Run this from your local computer, replacing the Vast.ai host and port:

```bash
scp -P YOUR_VAST_SSH_PORT \
  root@YOUR_VAST_SSH_HOST:/workspace/Modulation-Edit/outputs/fat_dog_grey_cat_spatial_test/seed_000.png \
  ~/Downloads/fat_dog_grey_cat_spatial_test.png
```

To download the result, metadata, offsets, and logs together:

```bash
rsync -av --progress -e "ssh -p YOUR_VAST_SSH_PORT" \
  root@YOUR_VAST_SSH_HOST:/workspace/Modulation-Edit/outputs/fat_dog_grey_cat_spatial_test/ \
  ~/Downloads/fat_dog_grey_cat_spatial_test/

rsync -av --progress -e "ssh -p YOUR_VAST_SSH_PORT" \
  root@YOUR_VAST_SSH_HOST:/workspace/Modulation-Edit/benchmark_work/fat_dog_grey_cat_spatial_test/ \
  ~/Downloads/fat_dog_grey_cat_spatial_test/training/
```

## Implementation note

At each early scale, the code forms one spatial SharedAdaLN tensor:

```text
[ fat-dog transformation | grey-cat transformation ]
```

All six vectors—SA gate, FFN gate, SA scale, FFN scale, SA shift, and FFN
shift—use the same token partition. Self-attention is executed once over the
full sequence, so the two sides can interact. Attention preserves sequence
positions; therefore the left/right gate values multiply the corresponding
left/right output tokens. Odd-width boundary tokens use overlap-weighted
blending, and the one-token coarsest scale uses a 50/50 transformation blend.
