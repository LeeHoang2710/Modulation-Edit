# ViCo ablation: select beta first, then sweep alpha

This is the recommended two-stage ablation for

\[
G_s=
\begin{cases}
G_{\text{subj}}+\beta D_s+\alpha G_{\text{prompt}},
&s\in\{0,2,3,4\},\\[2mm]
G_{\text{subj}},
&s\in\{1,5,\ldots,12\}.
\end{cases}
\]

The complete prompt remains in conditional `ca_kv` at every scale and CFG remains `1.5`. Run the beta sweep first with `alpha=0`, select the best beta from the metrics, and then keep that beta fixed for the alpha sweep.

## 1. Clone and prepare Vast.ai

Use a CUDA 12 development image with Python 3.10 and at least 40 GB of GPU memory. The commands below assume the persistent disk is mounted at `/workspace`.

### 1.1 Clone this branch

```bash
cd /workspace
git clone --branch friend_module_space_prompt_offset_sync \
  https://github.com/LeeHoang2710/Modulation-Edit.git
cd Modulation-Edit
```

If the repository already exists:

```bash
cd /workspace/Modulation-Edit
git fetch origin
git switch friend_module_space_prompt_offset_sync
git pull --ff-only origin friend_module_space_prompt_offset_sync
```

### 1.2 Prepare the environment, weights and ViCo assets

From the repository root, run:

```bash
cd /workspace/Modulation-Edit

bash scripts/setup_venv.sh

USE_ACTIVE_PYTHON=1 PYTHON_BIN=.venv/bin/python \
  bash scripts/download_vastai_weights.sh

bash scripts/prepare_vico_ablation_assets.sh
```

The third command clones ARBooth and CSD-VAR, pulls the official 16-subject ViCo references and prompts from ARBooth, creates `/workspace/CSD-VAR/benchmark_data/vico_csd`, installs the `csd-var` evaluator environment, and downloads the CSD-VAR metric checkpoint.

The offset stage uses the friend's selected setup:

```yaml
conditioning: subject-name text
loss scales: 0-4
steps: 200
batch size: 2
gradient accumulation: 2
learning rate: 5e-3
BSC: layers 13, strength 0.3
horizontal flip: disabled
offset initialization: zero
```

Only one offset is prepared per subject and it is reused by every beta and alpha variant.

## 2. Smoke-test the beta sweep

Run one subject, five prompts, one sample per prompt:

```bash
cd /workspace/Modulation-Edit

SUBJECTS=black_cat \
NUM_PROMPTS=5 \
NUM_SAMPLES=1 \
SWEEP=beta \
FIXED_ALPHA=0 \
bash ablation/05_subject_prompt_alpha_beta.sh all
```

This tests:

```text
beta = 0.6, 0.7, 0.8, 0.9, 1.0
alpha = 0
```

The smoke test generates `5 beta variants × 5 prompts × 1 sample = 25 images`.

## 3. Run the full beta sweep

```bash
SWEEP=beta \
FIXED_ALPHA=0 \
bash ablation/05_subject_prompt_alpha_beta.sh all
```

The complete beta stage generates:

```text
5 variants × 16 subjects × 31 prompts × 8 samples = 19,840 images
```

Results are saved under:

```text
output/ablation_vico/20_subject_prompt_alpha_beta/beta/
├── beta_a0_b0.6/
├── beta_a0_b0.7/
├── beta_a0_b0.8/
├── beta_a0_b0.9/
├── beta_a0_b1/
├── summary.csv
└── summary.md
```

Inspect the summary:

```bash
column -s, -t output/ablation_vico/20_subject_prompt_alpha_beta/beta/summary.csv | less -S
```

Select beta from the `overall` row, using DINO and CLIP-I as the primary identity criteria while checking that CLIP-T, CLIP-T-cls and DIV do not collapse. PRES and CSD metrics can be used as additional evidence when the evaluator emits them. Do not select beta from a single best image.

For example, suppose the selected value is:

```bash
BEST_BETA=0.9
```

## 4. Sweep alpha with the selected beta

Run a small alpha smoke test first:

```bash
BEST_BETA=0.9

SUBJECTS=black_cat \
NUM_PROMPTS=5 \
NUM_SAMPLES=1 \
SWEEP=alpha \
FIXED_BETA="$BEST_BETA" \
bash ablation/05_subject_prompt_alpha_beta.sh all
```

Then run the complete alpha experiment:

```bash
BEST_BETA=0.9

SWEEP=alpha \
FIXED_BETA="$BEST_BETA" \
bash ablation/05_subject_prompt_alpha_beta.sh all
```

This tests:

```text
alpha = 0, 0.1, 0.2, 0.3, 0.4
beta = 0.9 fixed
```

The `alpha=0` row is the selected beta baseline repeated in the alpha stage. It is intentionally retained as a paired reference for the four nonzero alpha values.

Alpha results are saved under:

```text
output/ablation_vico/20_subject_prompt_alpha_beta/alpha/
├── alpha_a0_b0.9/
├── alpha_a0.1_b0.9/
├── alpha_a0.2_b0.9/
├── alpha_a0.3_b0.9/
├── alpha_a0.4_b0.9/
├── summary.csv
└── summary.md
```

## 5. Metrics produced

`all` performs optimization/reuse, generation and evaluation. Every variant keeps all generated images and writes:

```text
config.json
eval.log
metrics.csv
<subject>/<subject>_metrics.csv
```

The stage-level `summary.csv` and `summary.md` include overall, live-subject and object-subject aggregates for every metric implemented by the evaluation pipeline:

```text
DINO
CLIP-I
CLIP-T
CLIP-T-cls
DIV
PRES          (when emitted by the evaluator)
CSD-content   (when emitted by the evaluator)
CSD-style     (when emitted by the evaluator)
```

Optional metrics remain blank when the external evaluator does not emit them; they are no longer silently removed from the summary.

## 6. Run stages separately

Use the same `SWEEP` value for generation and evaluation so the same output folder is selected.

```bash
# Prepare/reuse offsets
SWEEP=beta bash ablation/05_subject_prompt_alpha_beta.sh optimize

# Generate the beta variants
SWEEP=beta FIXED_ALPHA=0 bash ablation/05_subject_prompt_alpha_beta.sh generate

# Evaluate beta variants and rebuild summary.csv/summary.md
SWEEP=beta FIXED_ALPHA=0 bash ablation/05_subject_prompt_alpha_beta.sh eval

# Generate/evaluate alpha after choosing beta=0.9
SWEEP=alpha FIXED_BETA=0.9 bash ablation/05_subject_prompt_alpha_beta.sh generate
SWEEP=alpha FIXED_BETA=0.9 bash ablation/05_subject_prompt_alpha_beta.sh eval
```

To force metric recomputation:

```bash
FORCE_EVAL=1 SWEEP=beta FIXED_ALPHA=0 \
bash ablation/05_subject_prompt_alpha_beta.sh eval
```

## 7. Expected runtime

Using the previous measured rate of approximately `1.855 seconds/image`:

```text
beta inference:  ~10.2 hours
alpha inference: ~10.2 hours
both:            ~20.4 hours
```

Allow additional time for the initial 16 offsets and metric evaluation. The practical end-to-end estimate for both stages is approximately `23–27 hours` on one comparable GPU.
