# Step 50/100 × late `G_subject`/`G_prompt`

This experiment studies both offset convergence and the modulation used on
scales without an injected offset.

One uninterrupted optimization is run for 100 steps with batch size 4. The
same optimizer trajectory is checkpointed at steps 50 and 100. It does **not**
start a second optimization at step 50.

## Four evaluated variants

For each checkpoint, offsets are active at scales `0,2,3,4,5`:

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

The resulting variants are:

1. `step050_late_gsubject`
2. `step050_late_gprompt`
3. `step100_late_gsubject`
4. `step100_late_gprompt`

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
NUM_SAMPLES=4 \
INFER_BATCH=4 \
bash ablation/07_steps50_100_late_base.sh
```

This optimizes one offset trajectory and generates
`4 variants × 5 prompts × 4 images = 80` images. Four images from the same
variant and prompt are generated together.

## Full ViCo experiment

```bash
NUM_PROMPTS=0 \
NUM_SAMPLES=4 \
INFER_BATCH=4 \
bash ablation/07_steps50_100_late_base.sh
```

This optimizes 16 subject offsets and generates
`4 variants × 16 subjects × 31 prompts × 4 images = 7,936` images.

## Optimization settings

```text
steps:                 100
checkpoints:            50 and 100
batch size:             4
gradient accumulation:  1
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
- cumulative preparation time at steps 50 and 100

Therefore checkpoint serialization does not inflate the reported preparation
time.

Inference uses batch size 4 by default and is measured around model generation
only. Image encoding to JPEG and filesystem writing happen after the timer
stops. Every generated image records both the full batch latency and per-image
throughput time (`batch latency / actual batch size`). The final partial batch
is handled correctly when `NUM_SAMPLES` is not divisible by four.

## Outputs

```text
output/ablation_vico/22_steps50_100_late_base_seed3/
├── offsets/
│   ├── step50/<subject>/offset.pth
│   └── step100/<subject>/offset.pth
├── step050_late_gsubject/
├── step050_late_gprompt/
├── step100_late_gsubject/
├── step100_late_gprompt/
├── logs/
├── preparation_times.csv
├── inference_times_per_image.csv
├── inference_time_summary.csv
├── summary.csv
└── summary.md
```

`preparation_times.csv` contains the step-50 and step-100 preparation time for
every subject, plus the mean and population standard deviation. Checkpoint
write time is reported separately and is not included in either preparation
column.

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
