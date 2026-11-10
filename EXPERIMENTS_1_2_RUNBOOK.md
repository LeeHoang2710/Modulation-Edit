# Experiment 1 and Experiment 2 runbook

Run these commands from the repository root on Vast.ai:

```bash
git clone -b experiment1-2-csd-styles https://github.com/LeeHoang2710/Modulation-Edit.git
cd Modulation-Edit
```

Install or activate the project environment and make sure these assets exist:

```text
weights/infinity/infinity_2b_reg.pth
weights/infinity/infinity_vae_d32reg.pth
weights/infinity/flan-t5-xl/
```

The scripts default to `.venv/bin/python`. Override it with `PYTHON_BIN` if the
Vast.ai environment uses another Python executable.

## Experiment 1: style offset placement

This uses the eight committed style images in `styles/` and fits one per-scale
offset per style image on scales 0–5. It then generates four variants for the
prompt `a dog`:

1. no style injection;
2. style offset injected at scale 0;
3. style offset injected at scale 1;
4. style offset injected at scales 0 and 1.

Run:

```bash
PYTHON_BIN=.venv/bin/python bash scripts/run_style_endpoint_offsets.sh
```

Useful overrides:

```bash
PROMPT='a dog' STEPS=200 GRAD_ACCUM=2 NUM_SAMPLES=1 \
  PYTHON_BIN=.venv/bin/python bash scripts/run_style_endpoint_offsets.sh
```

The default optimization uses batch size 1, gradient accumulation 2, BSC
noise on up to 13 history layers, and maximum bit-flip strength `0.3`.

Outputs:

```text
outputs/style_single_scale_dog/style_<name>_none.png
outputs/style_single_scale_dog/style_<name>_scale0.png
outputs/style_single_scale_dog/style_<name>_scale1.png
outputs/style_single_scale_dog/style_<name>_scale0_1.png
outputs/style_single_scale_dog/style_scale_placement_comparison.png
```

## Experiment 2: reconstruction scale-prefix sweep

This trains one subject-conditioned offset on scales `0-3` for `black_cat`
(`a grey cat`) and `fat_dog` (`a fat dog`), then tests ten injection sets:

`0`, `1`, `2`, `3`, `0+1`, `0+1+2`, `0+1+2+3`, `0+2+3`, `1+2+3`, and `2+3`.

```bash
PYTHON_BIN=.venv/bin/python bash scripts/run_scale_reconstruction_ablation.sh
```

The friend main-variant settings are 200 optimization steps, batch size 2, gradient
accumulation 2, BSC noise on up to 13 history layers with maximum bit-flip
strength `0.3`, learning rate `5e-3`, subject prompts `a grey cat` and
`a fat dog`, offset strength `beta=0.9`, prompt cross-attention, and CFG 1.5.
To run one object
or use a longer optimization:

```bash
SUBJECTS=black_cat STEPS=200 \
  PYTHON_BIN=.venv/bin/python bash scripts/run_scale_reconstruction_ablation.sh
```

Outputs:

```text
benchmark_work/scale_reconstruction_ablation/<subject>/scale_0-<end>/offset.pth
benchmark_work/scale_reconstruction_ablation/<subject>/scale_0-<end>/reconstruction.jpg
outputs/scale_reconstruction_ablation/black_cat_scale_reconstruction_grid.png
outputs/scale_reconstruction_ablation/fat_dog_scale_reconstruction_grid.png
```

The `data/` directory is git-ignored. The CSD-100 download is therefore only
local; it is not required for these two runs because Experiment 1 uses the
committed style images and Experiment 2 uses `ARBooth/inputs/`.
