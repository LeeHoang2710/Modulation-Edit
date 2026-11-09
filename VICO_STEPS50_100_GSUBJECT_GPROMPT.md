# Step 100 × late `G_subject`/`G_prompt`

This experiment studies both offset convergence and the modulation used on
scales without an injected offset.

One optimization is run for 100 steps with batch size 4 and gradient
accumulation 2. Only the final step-100 offset is saved.

## Two evaluated variants

The step-100 offset is active at scales `0,2,3,4,5`:

\[
G_s=G_{\mathrm{subj}}+D_s,\qquad s\in\{0,2,3,4,5\}.
\]

The remaining scales use either:

\[
G_s=G_{\mathrm{subj}},\qquad s\in\{1,6,\ldots,12\},
\]

or:

\[
G_s=G_{\mathrm{prompt}},\qquad s\in\{1,6,\ldots,12\}.
\]

The two variants are:

1. `step100_late_gsubject`
2. `step100_late_gprompt`

All variants use `beta=1.0`, prompt CA-KV at all scales, CFG `1.5`, and the
same inference seed (default `3`).

## Preparation

```bash
cd /workspace/Modulation-Edit
git fetch origin
git switch friend_module_space_prompt_offset_sync
git pull --ff-only origin friend_module_space_prompt_offset_sync
bash scripts/prepare_vico_beta_alpha.sh
```

## Smoke test: one subject and five prompts

```bash
SUBJECTS=black_cat \
NUM_PROMPTS=5 \
bash ablation/07_steps50_100_late_base.sh
```

This optimizes one offset and generates `2 variants × 5 prompts = 10` images.

## Full ViCo experiment

```bash
NUM_PROMPTS=0 \
bash ablation/07_steps50_100_late_base.sh
```

This optimizes 16 subject offsets and generates
`2 variants × 16 subjects × 31 prompts = 992` images.

## Optimization settings

```text
steps:                 100
checkpoint:             100 only
batch size:             4
gradient accumulation:  2
learning rate:           5e-3
optimization seed:       3
loss scales:             0-5
training condition:      subject/class text
augmentation:            disabled
bitwise correction:      13 layers, strength 0.3
```

Checkpoint writing is timed separately. Each subject's `timing.json` reports:

- `preparation_seconds_excluding_checkpoint_io`
- `checkpoint_save_seconds`
- preparation time at step 100

Therefore checkpoint serialization does not inflate the reported preparation
time.

The inference batch cap is 3 and generation is measured around model generation
only. Image encoding to JPEG and filesystem writing happen after the timer
stops. Because `NUM_SAMPLES=1`, each prompt has an actual batch size of 1; the
configured cap of 3 cannot combine images across different prompts or variants.

## Outputs

```text
output/ablation_vico/23_step100_two_late_bases_seed3/
├── offsets/
│   └── step100/<subject>/offset.pth
├── step100_late_gsubject/
├── step100_late_gprompt/
├── logs/
├── preparation_times.csv
├── inference_times_per_image.csv
├── inference_time_summary.csv
├── summary.csv
└── summary.md
```

`preparation_times.csv` contains the step-100 preparation time for every
subject, plus the mean and population standard deviation. Checkpoint write time
is reported separately and is not included in the preparation column.

`inference_times_per_image.csv` contains one row for every generated image,
including its variant, subject, prompt index, image index, seed, batch size,
full batch latency, and per-image throughput time. `inference_time_summary.csv`
reports mean and population standard deviation of throughput seconds per image,
per subject and for each complete variant.

The evaluator computes only the three main metrics:

- DINO
- CLIP-I
- CLIP-T

## Resume or rerun

Completed offsets and prompt folders are reused automatically. To optimize the
offsets again:

```bash
FORCE_OPTIMIZE=1 SUBJECTS=black_cat NUM_PROMPTS=5 \
bash ablation/07_steps50_100_late_base.sh
```
