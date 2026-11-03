# Six-object, five-offset class-conditioned ViCo experiment

This experiment runs the class-conditioned fixed-alpha method on six selected subjects:

- `black_cat`
- `brown_dog`
- `cat_statue`
- `tortoise_plushy`
- `wooden_barn`
- `alarm_clock`

For every subject, it independently optimizes five offsets with training seeds 0–4. Each optimization uses `a <subject name>` rather than null text and supervises scales 0–4. Every offset then uses the same inference seed, 0 by default.

Each offset produces six images:

1. `a <subject name>`, used to inspect identity reconstruction;
2. the first official ViCo prompt with the full subject name;
3. the second official ViCo prompt;
4. the third official ViCo prompt;
5. the fourth official ViCo prompt;
6. the fifth official ViCo prompt.

Inference uses:

\[
G_s=0.9(G_{\mathrm{class}}+D_s)+0.1G_{\mathrm{prompt}},\quad s=0,\ldots,4,
\]

and \(G_s=G_{\mathrm{prompt}}\) on scales 5–12. The full prompt enters conditional `ca_kv` at every scale.

## Prepare Vast.ai

```bash
cd /workspace
git clone --branch five-model-dreambooth-benchmark \
  https://github.com/LeeHoang2710/Modulation-Edit.git
cd Modulation-Edit
bash scripts/setup_vico_vastai.sh
bash scripts/prepare_vico_assets.sh
```

## Run all 180 images

```bash
cd /workspace/Modulation-Edit
bash scripts/run_vico_class_conditioned_five_offsets.sh
```

Defaults are five offset seeds, 50 optimization steps, batch size 5, gradient accumulation 1, learning rate `4e-3`, five ViCo templates, and inference seed 0.

The default run produces exactly:

\[
5\text{ offset seeds}\times6\text{ subjects}\times6\text{ prompts}=180\text{ images}.
\]

The launcher verifies all 180 expected files before reporting completion.

The outputs are organized as:

```text
benchmark_outputs/vico_class_conditioned_five_offsets_alpha01/
└── method/
    └── cat_statue/
        ├── prompt_00/   # a cat statue
        ├── prompt_01/   # first ViCo prompt
        ├── ...
        └── prompt_05/   # fifth ViCo prompt
```

Every prompt folder contains `seed_000.png` through `seed_004.png`. These filenames identify the five independently optimized offset seeds; generation itself uses the same inference seed for all five.

Offsets, reconstructions, timing JSON files, and logs are stored under `benchmark_work/vico_class_conditioned_five_offsets/`.

## Useful overrides

Run one subject while keeping all five offset seeds:

```bash
SUBJECTS=cat_statue bash scripts/run_vico_class_conditioned_five_offsets.sh
```

Run a faster pipeline check:

```bash
SUBJECTS=cat_statue OFFSET_SEEDS=1 STEPS=2 \
bash scripts/run_vico_class_conditioned_five_offsets.sh
```

Set `REUSE_OFFSETS=0` to refit offsets that already exist. The launcher validates the saved class prompt, scales, and optimization seed before reusing a checkpoint.
