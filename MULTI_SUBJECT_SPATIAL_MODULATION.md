# Fat-dog + grey-cat spatial modulation test on Vast.ai

This recipe trains exactly two class-conditioned SharedAdaLN offsets and
generates one test image:

- left region: the `fat_dog` reference object, conditioned as `a fat dog`;
- right region: the `black_cat` reference object, conditioned as `a grey cat`;
- joint prompt: `a dog next to a cat`;
- training and inference seed `3`.

The fixed optimization configuration is 200 steps, batch size 4, gradient
accumulation 1, learning rate `5e-3`, and per-scale offsets on the paper's
main-variant scales `{0,2,3,4,5}`; scale 1 retains pretrained prompt behavior.
Bitwise labels and bitwise self-correction are enabled. BSC is requested for
13 history layers with maximum bit-flip probability 0.3; because this run
forwards through scale 5, the optimizer correctly caps corruption to the five
available preceding histories. Inference uses offset strength 1.0 and CFG 1.5.
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

## 2. Clone only our repository

Use a normal clone and do **not** initialize submodules: the spatial experiment
does not need DreamMatcher, ARBooth, or any other comparison-method source
tree, data, or environment.

```bash
cd /workspace
git clone --branch feature/multi-subject-spatial-modulation \
  --single-branch \
  https://github.com/LeeHoang2710/Modulation-Edit.git
cd Modulation-Edit
git branch --show-current
```

The last command must print:

```text
feature/multi-subject-spatial-modulation
```

The clone already contains all 16 reference-object folders under
`references/vico_16/`. Verify the two used by this test and the full inventory:

```bash
find references/vico_16 -mindepth 1 -maxdepth 1 -type d | sort
find references/vico_16/fat_dog -maxdepth 1 -type f | sort
find references/vico_16/black_cat -maxdepth 1 -type f | sort
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

The setup script creates only the `infinity-modulation` environment and
installs the official PyTorch 2.9.1 CUDA 12.8 wheel, torchvision 0.24.1, and
the upstream prebuilt FlashAttention 2.8.3 wheel for Python 3.12/Torch 2.9.
This combination supports the RTX 5090's `sm_120` architecture without a local
CUDA compilation. Pip caching is disabled. The requirements installation also
excludes its legacy PyTorch/FlashAttention pins, so it cannot silently
downgrade the compatible GPU stack afterward. No comparison-method or
evaluation environments are created.

```bash
bash scripts/setup_vastai_env.sh
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate infinity-modulation
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

### Repair an existing RTX 5090 environment

If you already created `/venv/infinity-modulation` and see a warning saying
that RTX 5090 / `sm_120` is incompatible, repair that environment in place:

```bash
cd /workspace/Modulation-Edit
git pull --ff-only
INFINITY_PYTHON=/venv/infinity-modulation/bin/python \
  bash scripts/repair_rtx5090_env.sh
```

First interrupt any existing `Building wheel for flash-attn` process with
Ctrl-C. The repair then checks for an RTX 5090-class `sm_120` GPU, removes only
abandoned pip build directories and pip's download cache, installs prebuilt
wheels without compilation, and runs both a CUDA tensor test and the same
varlen packed FlashAttention operation used by prompt CA-KV. After it succeeds,
rerun:

```bash
bash scripts/run_fat_dog_grey_cat_spatial_test.sh
```

## 5. Download the Infinity weights

The experiment expects the model, VAE, and T5 encoder under
`weights/infinity/`. Set `HF_TOKEN` first only if Hugging Face requests one.

```bash
# Optional for gated/authenticated downloads:
# export HF_TOKEN='hf_...'

bash scripts/download_vastai_weights.sh weights/infinity
```

Verify the expected assets:

```bash
ls -lh \
  weights/infinity/infinity_2b_reg.pth \
  weights/infinity/infinity_vae_d32reg.pth \
  weights/infinity/flan-t5-xl/config.json
```

## 6. Run the complete experiment

```bash
cd /workspace/Modulation-Edit
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate infinity-modulation
bash scripts/run_fat_dog_grey_cat_spatial_test.sh
```

The runner performs these operations in order:

1. reads the committed first-party references from `references/vico_16/`;
2. optimizes the fat-dog offset for 200 steps;
3. optimizes the grey-cat offset for 200 steps;
4. renders the fat-dog offset independently as a full-image control;
5. renders the grey-cat offset independently as a full-image control;
6. constructs left/right token-wise transformations using all six SharedAdaLN
   vectors;
7. runs global self-attention over both regions;
8. keeps prompt CA-KV enabled at every scale;
9. writes the final spatial composition.

The individual offset controls are written first:

```text
outputs/fat_dog_grey_cat_spatial_test/fat_dog_seed_003.png
outputs/fat_dog_grey_cat_spatial_test/grey_cat_seed_003.png
```

The final spatial result is:

```text
outputs/fat_dog_grey_cat_spatial_test/seed_003.png
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
  root@YOUR_VAST_SSH_HOST:/workspace/Modulation-Edit/outputs/fat_dog_grey_cat_spatial_test/seed_003.png \
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
