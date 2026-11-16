# Main variant: faster training configurations, inference seed 3

This experiment compares two faster offset-training configurations against the
existing `bs2_ga2_it200` result. It does **not** regenerate that existing
baseline.

Both new configurations use the exact main inference variant:

```text
optimized offset scales: 0-5
applied offset scales:   0,2-5 (scale 1 is omitted at inference)
offset strength beta:    1.0
scales 0,2-5:            G_subject + D_s
scales 1,6-12:           G_prompt
CA-KV:                   full modification prompt at every scale
CFG:                     1.5
optimization seed:       3
inference seed:          3
augmentation:            disabled
bitwise correction:      13 layers, strength 0.3
```

Equivalently, inference uses:

\[
G_s=
\begin{cases}
G_{\mathrm{subj}}+D_s,&s\in\{0,2,3,4,5\},\\
G_{\mathrm{prompt}},&s\in\{1,6,\ldots,12\}.
\end{cases}
\]

The tested optimization configurations are:

| Name | Batch | Grad accumulation | Optimizer steps |
|---|---:|---:|---:|
| `bs4_ga1_it200` | 4 | 1 | 200 |
| `bs4_ga2_it100` | 4 | 2 | 100 |

## Update and prepare the server

```bash
cd /workspace/Modulation-Edit
git pull origin friend_module_space_prompt_offset_sync
bash scripts/prepare_three_model_timing.sh
```

## Smoke test: one subject, five prompts

```bash
SUBJECTS=black_cat NUM_PROMPTS=5 \
bash ablation/09_main_training_configs_seed3.sh
```

This creates `2 × 5 = 10` images after optimizing two `black_cat` offsets.

## Full ViCo experiment

```bash
NUM_PROMPTS=0 bash ablation/09_main_training_configs_seed3.sh
```

This optimizes `2 × 16 = 32` subject offsets and generates
`2 × 16 × 31 = 992` images. Each prompt uses exactly one inference sample with
seed 3.

Completed offsets and prompt folders are reused automatically. To deliberately
restart offset optimization, add `FORCE_OPTIMIZE=1`.

## Outputs

```text
output/ablation_vico/25_main_training_configs_seed3/
├── offsets/
│   ├── bs4_ga1_it200/<subject>/{offset.pth,timing.json,optimize.log}
│   └── bs4_ga2_it100/<subject>/{offset.pth,timing.json,optimize.log}
├── bs4_ga1_it200/<subject>/prompt_XX/00.jpg
├── bs4_ga2_it100/<subject>/prompt_XX/00.jpg
├── summary.csv
├── summary.md
└── logs/
```

Each variant also contains `metrics.csv`. `summary.csv` compares DINO, CLIP-I,
CLIP-T, CLIP-T class, DIV, CSD content, and CSD style using the same evaluator
as the existing ablations.
