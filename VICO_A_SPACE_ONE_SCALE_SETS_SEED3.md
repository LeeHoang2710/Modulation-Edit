# A-space-one scale-set comparison

This experiment learns one shared offset per subject for each of four scale
sets, then injects that same offset at the selected scales:

```text
{0,2,3}
{0,1,2,3}
{0,2,3,4}
{0,2,3,4,5}
```

All variants use Adam, learning rate `5e-3`, zero weight decay, 200 optimizer
steps, batch size 4, gradient accumulation 1, optimization seed 3, inference
seed 3 for both optimization and inference, one output per prompt, CFG 1.5,
and no horizontal flips. The loss is computed only on the
variant's selected scales. At inference, the subject-conditioned base
modulation plus the shared offset is used on the selected scales; prompt
modulation is used on scales `1,6-12`, with prompt cross-attention at every
scale.

## Run

Smoke test:

```bash
SUBJECTS=black_cat NUM_PROMPTS=5 \
bash ablation/10_a_space_one_scale_sets_seed3.sh
```

Full ViCo protocol:

```bash
NUM_PROMPTS=0 bash ablation/10_a_space_one_scale_sets_seed3.sh
```

Outputs are written to:

```text
output/ablation_vico/27_a_space_one_scale_sets_seed3/
```

The script resumes completed offsets and prompt folders. Set
`FORCE_OPTIMIZE=1` to re-run offset optimization.
