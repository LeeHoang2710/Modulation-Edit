# Non-offset `G_subject` vs `G_prompt` — seed 3

This experiment leaves the main variant untouched and compares exactly two
inference variants using the same learned offset, prompts, CA-KV, and random
seed (`3`).

## Formulas

The main variant is:

\[
G_s=
\begin{cases}
G_{\mathrm{subj}}+D_s,&s\in\{0,2,3,4,5\},\\
G_{\mathrm{subj}},&s\in\{1,6,\ldots,12\}.
\end{cases}
\]

The new test variant is:

\[
G_s=
\begin{cases}
G_{\mathrm{subj}}+D_s,&s\in\{0,2,3,4,5\},\\
G_{\mathrm{prompt}},&s\in\{1,6,\ldots,12\}.
\end{cases}
\]

For both variants, `beta=1.0`, prompt CA-KV is active at all 13 scales, and
CFG defaults to `1.5`. Before inference, the runner freshly optimizes one
subject-conditioned offset on scales `0–5` with optimization seed `3`. The
same resulting offset is then used by both inference variants. Scale 1 is
learned but deliberately omitted at inference.

Default offset optimization settings are:

```text
optimization seed: 3
steps:             200
batch size:        2
gradient accum.:   2
learning rate:     5e-3
loss scales:       0-5
training text:     subject/class name
augmentation:      disabled
bit correction:    13 layers, strength 0.3
```

## 1. Clone or update the correct branch

For a fresh Vast.ai machine:

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

Confirm the branch before running:

```bash
git branch --show-current
```

It must print `friend_module_space_prompt_offset_sync`.

## 2. Prepare Vast.ai

From the repository root:

```bash
bash scripts/prepare_vico_beta_alpha.sh
```

Wait until it prints:

```text
[ready] The beta-then-alpha ViCo pipeline is fully prepared.
```

## 3. Smoke test: one object, five prompts

```bash
SUBJECTS=black_cat \
NUM_PROMPTS=5 \
bash ablation/06_nonoffset_prompt.sh
```

This first learns one new `black_cat` offset, then creates
`2 × 5 × 1 = 10` images and evaluates the two variants separately.

Completed offsets are reused when resuming an interrupted run. To deliberately
optimize them again, use:

```bash
FORCE_OPTIMIZE=1 SUBJECTS=black_cat NUM_PROMPTS=5 \
bash ablation/06_nonoffset_prompt.sh
```

## 4. Full comparison: 16 objects × 31 prompts × one seed

```bash
NUM_PROMPTS=0 \
bash ablation/06_nonoffset_prompt.sh
```

This first learns 16 offsets, one per subject. Each variant then produces
`16 × 31 = 496` images. Both variants use inference seed 3 and the same
corresponding subject offset, so the comparison is paired.

## Outputs

```text
output/ablation_vico/21_nonoffset_prompt_reopt_seed3/
├── offsets_seed3_ls0-5_subjtext/
│   └── <subject>/
│       ├── offset.pth
│       └── optimize.log
├── main_nonoffset_gsubject/
├── test_nonoffset_gprompt/
├── logs/
└── summary.csv
```

`summary.csv` contains the aggregate metrics for direct comparison. Detailed
per-image metrics and evaluator logs remain inside each variant directory.

To use another object for the smoke test:

```bash
SUBJECTS=cat_statue NUM_PROMPTS=5 \
bash ablation/06_nonoffset_prompt.sh
```

To change optimization settings explicitly:

```bash
SUBJECTS=cat_statue NUM_PROMPTS=5 \
OPT_SEED=3 ITERS=200 BATCH_SIZE=2 GRAD_ACCUM=2 LR=5e-3 \
bash ablation/06_nonoffset_prompt.sh
```
