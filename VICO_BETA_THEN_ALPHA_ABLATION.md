# ViCo automatic beta-then-alpha ablation

The experiment uses

\[
G_s=
\begin{cases}
G_{\text{subj}}+\beta D_s+\alpha G_{\text{prompt}},
&s\in\{0,2,3,4\},\\[2mm]
G_{\text{subj}},
&s\in\{1,5,\ldots,12\}.
\end{cases}
\]

The complete modification prompt remains in conditional `ca_kv` at every scale and CFG remains `1.5`.

The launcher now performs the complete protocol automatically:

1. prepare or reuse the subject-conditioned BSC offsets;
2. run the beta sweep with `alpha=0`;
3. evaluate every beta;
4. select the best beta automatically;
5. run the alpha sweep with that beta fixed;
6. evaluate every alpha.

## 1. Setup Vast.ai

Clone or update the branch:

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

Prepare the Infinity environment, Infinity weights, ViCo benchmark, ARBooth evaluator, and CSD-VAR metrics:

```bash
cd /workspace/Modulation-Edit

bash scripts/setup_venv.sh

USE_ACTIVE_PYTHON=1 PYTHON_BIN=.venv/bin/python \
  bash scripts/download_vastai_weights.sh

bash scripts/prepare_vico_ablation_assets.sh
```

The asset script creates:

```text
/workspace/ARBooth
/workspace/CSD-VAR
/workspace/CSD-VAR/benchmark_data/vico_csd
/root/miniconda3/envs/csd-var
weights/infinity_2b_reg.pth
weights/infinity_vae_d32reg.pth
weights/flan-t5-xl
```

## 2. Offset optimization configuration

One offset is learned per subject and reused by every beta and alpha variant:

```yaml
conditioning: subject-name text
loss scales: 0-4
steps: 200
batch size: 2
gradient accumulation: 2
learning rate: 5e-3
BSC layers: 13
BSC strength: 0.3
horizontal flip: disabled
offset initialization: zero
```

Inference applies the learned offset only at scales `0,2,3,4`.

## 3. Automatic smoke test

Run one subject, five prompts, and one image per prompt:

```bash
cd /workspace/Modulation-Edit

SUBJECTS=black_cat \
NUM_PROMPTS=5 \
NUM_SAMPLES=1 \
bash ablation/05_subject_prompt_alpha_beta.sh
```

The smoke test produces:

```text
beta stage:  5 beta values  × 5 prompts × 1 image = 25 images
alpha stage: 5 alpha values × 5 prompts × 1 image = 25 images
total: 50 images
```

## 4. Automatic full experiment

Run all 16 subjects, 31 prompts, and 8 images per prompt:

```bash
cd /workspace/Modulation-Edit
bash ablation/05_subject_prompt_alpha_beta.sh
```

The full run produces:

```text
beta stage:  5 × 16 × 31 × 8 = 19,840 images
alpha stage: 5 × 16 × 31 × 8 = 19,840 images
total: 39,680 images
```

The terminal displays only these two image-level progress bars:

```text
1/2 beta sweep:  ... / 19,840 img
2/2 alpha sweep: ... / 19,840 img
```

Model loading, image-processing messages, warnings, evaluation output, and optimizer status are written to log files instead of the terminal.

## 5. Automatic beta selection

The default criterion is the highest overall DINO score because beta directly controls subject-identity offset strength. Ties are resolved using:

```text
CLIP-I -> CLIP-T-cls -> CLIP-T -> DIV
```

The selected value and complete winning metrics row are saved to:

```text
output/ablation_vico/20_subject_prompt_alpha_beta/selected_beta.json
```

To select using another metric:

```bash
BETA_SELECTION_METRIC=clip_i \
bash ablation/05_subject_prompt_alpha_beta.sh
```

Supported selection fields include `dino`, `clip_i`, `clip_t`, `clip_t_cls`, `div`, `csd_content`, and `csd_style`.

## 6. Sweep values

Beta stage:

```text
alpha = 0
beta  = 0.6, 0.7, 0.8, 0.9, 1.0
```

Alpha stage:

```text
beta  = automatically selected
alpha = 0, 0.1, 0.2, 0.3, 0.4
```

Override either list if needed:

```bash
BETAS=0.7,0.8,0.9 \
ALPHAS=0,0.05,0.1,0.15 \
bash ablation/05_subject_prompt_alpha_beta.sh
```

## 7. Images, metrics, and logs

Generated images and metrics:

```text
output/ablation_vico/20_subject_prompt_alpha_beta/
├── beta/
│   ├── beta_a0_b0.6/
│   ├── beta_a0_b0.7/
│   ├── beta_a0_b0.8/
│   ├── beta_a0_b0.9/
│   ├── beta_a0_b1/
│   ├── summary.csv
│   └── summary.md
├── alpha/
│   ├── alpha_a0_b<selected>/
│   ├── alpha_a0.1_b<selected>/
│   ├── alpha_a0.2_b<selected>/
│   ├── alpha_a0.3_b<selected>/
│   ├── alpha_a0.4_b<selected>/
│   ├── summary.csv
│   └── summary.md
└── selected_beta.json
```

Central pipeline logs:

```text
output/ablation_vico/20_subject_prompt_alpha_beta/logs/
├── offset_optimization.log
├── beta_generation.log
├── beta_evaluation.log
├── beta_selection.log
├── alpha_generation.log
└── alpha_evaluation.log
```

Detailed step-by-step loss logs remain under each subject offset:

```text
output/ablation_vico/offsets/ls0-4_subjtext_it200_bsc_noaug/<subject>/optimize.log
```

Every evaluated variant retains:

```text
config.json
eval.log
metrics.csv
<subject>/<subject>_metrics.csv
```

The summaries contain:

```text
DINO
CLIP-I
CLIP-T
CLIP-T-cls
DIV
PRES when available
CSD-content
CSD-style
```

## 8. Resume behavior

Rerun the same automatic command after an interruption:

```bash
bash ablation/05_subject_prompt_alpha_beta.sh
```

Existing offsets and complete prompt folders are reused. The progress bars count reused images as completed.

To force metric recomputation:

```bash
FORCE_EVAL=1 bash ablation/05_subject_prompt_alpha_beta.sh
```

To use a fresh output directory:

```bash
OUT_BASE=output/ablation_vico/new_beta_then_alpha \
bash ablation/05_subject_prompt_alpha_beta.sh
```

## 9. Manual stages

The old manual controls remain available:

```bash
# Offsets only
bash ablation/05_subject_prompt_alpha_beta.sh optimize

# Beta only
SWEEP=beta FIXED_ALPHA=0 \
bash ablation/05_subject_prompt_alpha_beta.sh all

# Alpha only with an explicit beta
SWEEP=alpha FIXED_BETA=0.9 \
bash ablation/05_subject_prompt_alpha_beta.sh all
```

At the previously measured rate of approximately `1.855 seconds/image`, generation takes about `10.2 hours` per stage or `20.4 hours` for both stages. Allow additional time for initial offset optimization and metric evaluation.
