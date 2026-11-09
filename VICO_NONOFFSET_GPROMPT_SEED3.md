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
CFG defaults to `1.5`. The offsets must have been optimized on scales `0–5`
using the subject/class text. Scale 1 is learned but deliberately omitted at
inference.

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

If the subject-conditioned `0–5` offsets already exist, no optimization is
repeated.

## 3. Smoke test: one object, five prompts

Change `OFFSET_ROOT` if your offsets are stored elsewhere:

```bash
OFFSET_ROOT=output/ablation_vico/offsets/ls0-5_subjtext_it200_bsc_noaug \
SUBJECTS=black_cat \
NUM_PROMPTS=5 \
bash ablation/06_nonoffset_prompt.sh
```

This creates `2 × 5 × 1 = 10` images and evaluates the two variants
separately.

## 4. Full comparison: 16 objects × 31 prompts × one seed

```bash
OFFSET_ROOT=output/ablation_vico/offsets/ls0-5_subjtext_it200_bsc_noaug \
NUM_PROMPTS=0 \
bash ablation/06_nonoffset_prompt.sh
```

Each variant produces `16 × 31 = 496` images. Both use inference seed 3, so
the comparison is paired.

## Outputs

```text
output/ablation_vico/21_nonoffset_prompt_seed3/
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
OFFSET_ROOT=/path/to/offset/root \
bash ablation/06_nonoffset_prompt.sh
```
