# CSD-VAR, Custom Diffusion, and DreamMatcher on ViCo

This branch provides three separate ViCo replications with the same normalized
output layout and ARBooth-style DINO, CLIP-I, and CLIP-T evaluator. Every
concept-prompt pair produces five independently seeded images (`0,1,2,3,4`).
Each full model run therefore generates:

```text
16 concepts × 31 prompts × 5 seeds = 2,480 images
```

DreamMatcher is applied on top of the released Custom Diffusion checkpoints;
it does not retrain Custom Diffusion.

## 1. Clone and select the branch

```bash
cd /workspace
git clone --recurse-submodules https://github.com/LeeHoang2710/Modulation-Edit.git
cd Modulation-Edit
git switch five-model-dreambooth-benchmark
git submodule update --init --recursive
```

For an existing clone:

```bash
cd /workspace/Modulation-Edit
git fetch origin
git switch five-model-dreambooth-benchmark
git pull origin five-model-dreambooth-benchmark
git submodule update --init --recursive
```

## 2. Install the two isolated environments

Use a Vast.ai CUDA development image with a visible NVIDIA GPU:

```bash
bash scripts/setup_csd_custom_dreammatcher_vico.sh
```

This creates `benchmark-csdvar` for CSD-VAR and
`benchmark-personalization` for Custom Diffusion and DreamMatcher.

## 3. Download ViCo, checkpoints, and metric assets

If Hugging Face requests authentication, export a read token first:

```bash
export HF_TOKEN='YOUR_HUGGING_FACE_READ_TOKEN'
bash scripts/prepare_csd_custom_dreammatcher_assets.sh
```

The script downloads ViCo, CSD-VAR/Switti assets, Stable Diffusion v1.4, the
16 released Custom Diffusion `delta.bin` checkpoints from Google Drive, and
the DINO/CLIP evaluation assets.

## 4. Smoke tests: one subject, two prompts, five seeds

Each dedicated smoke launcher defaults to `brown_dog` and produces exactly
`1 × 2 × 5 = 10` images. Pass `SUBJECT=<ViCo subject>` to test another object.

CSD-VAR:

```bash
bash scripts/smoke_csd_var_vico_5seeds.sh
```

Custom Diffusion:

```bash
bash scripts/smoke_custom_diffusion_vico_5seeds.sh
```

DreamMatcher on Custom Diffusion:

```bash
bash scripts/smoke_dreammatcher_custom_diffusion_vico_5seeds.sh
```

For example, test `cat_statue` with:

```bash
SUBJECT=cat_statue bash scripts/smoke_custom_diffusion_vico_5seeds.sh
SUBJECT=cat_statue bash scripts/smoke_dreammatcher_custom_diffusion_vico_5seeds.sh
```

Each smoke test also runs the evaluator and writes a `summary.csv` in its
model-specific report directory.

## 5. Full runs

Run each model separately:

```bash
bash scripts/run_csd_var_vico.sh
bash scripts/run_custom_diffusion_vico_5seeds.sh
bash scripts/run_dreammatcher_custom_diffusion_vico_5seeds.sh
```

Or run and evaluate all three sequentially:

```bash
bash scripts/run_csd_custom_dreammatcher_vico_5seeds.sh
```

Default image directories:

```text
benchmark_outputs_vico_csd_var_5seeds/
benchmark_outputs_vico_custom_diffusion_5seeds/
benchmark_outputs_vico_dreammatcher_custom_diffusion_5seeds/
```

Default metric reports:

```text
benchmark_reports_vico_csd_var_5seeds/summary.csv
benchmark_reports_vico_custom_diffusion_5seeds/summary.csv
benchmark_reports_vico_dreammatcher_custom_diffusion_5seeds/summary.csv
```

Normalized images use this layout:

```text
<output>/<subject>/prompt_00/seed_000.png
...
<output>/<subject>/prompt_30/seed_004.png
```

## Protocol notes

- **CSD-VAR:** one deterministic ViCo reference, one learned representation
  per subject, then five separate upstream inference processes with seeds 0–4.
- **Custom Diffusion:** released ViCo `delta.bin`, SD v1.4, 200 inference
  steps, CFG 6, and `eta=1`, matching its official inference defaults.
- **DreamMatcher + Custom Diffusion:** the same released `delta.bin`, the
  official designated source image, DDIM inversion, foreground appearance
  matching, 50 inference steps, and CFG 10. Only sampling is changed to five
  explicit independent seeds.
- Reports stay separate because these are three distinct evaluated models.
