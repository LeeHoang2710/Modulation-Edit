# One-subject, five-offset class-conditioned ViCo experiment

This experiment learns five independent offsets for one selected subject using the subject's class-name prompt during optimization. The default subject is `black_cat`, so its optimization prompt is `a black cat`.

Optimization uses:

- `a <full subject name>` as text conditioning;
- separate offsets on scales 0–4;
- batch size 2;
- gradient accumulation 1;
- 200 optimization steps;
- learning rate `5e-4`;
- bitwise self-correction (BSC) on histories from scales 0–3, with a newly sampled bit-flip probability in `[0, 0.3]` for every affected scale and step;
- no horizontal-flip augmentation.

Only the teacher-forced histories are corrupted; target bits stay clean. The offset must therefore recover the subject under imperfect early-scale histories instead of memorizing the five references under perfect teacher forcing.

At inference, scales 0–4 use:

\[
G_s=0.9G_{\text{class}}+1.0D_s+0.1G_{\text{prompt}}.
\]

The implementation computes the equivalent class-relative form:

\[
G_s=G_{\text{class}}+D_s+0.1(G_{\text{prompt}}-G_{\text{class}}).
\]

Scales 5–12 use \(G_{\text{prompt}}\), and the complete prompt enters conditional `ca_kv` at every scale.

Each offset generates six images with the same inference seed:

1. `a <subject name>` for identity reconstruction;
2. the first five official ViCo prompts containing the full subject name.

The default run creates `5 offsets × 6 prompts = 30 images`.

## Prepare Vast.ai

```bash
cd /workspace
git clone --branch five-model-dreambooth-benchmark \
  https://github.com/LeeHoang2710/Modulation-Edit.git
cd Modulation-Edit
bash scripts/setup_vico_vastai.sh
bash scripts/prepare_vico_assets.sh
```

## Run the default black-cat experiment

```bash
cd /workspace/Modulation-Edit
bash scripts/run_vico_class_conditioned_one_subject_five_offsets.sh
```

## Run another subject

```bash
bash scripts/run_vico_class_conditioned_one_subject_five_offsets.sh cat_statue
bash scripts/run_vico_class_conditioned_one_subject_five_offsets.sh alarm_clock
```

Outputs are written under:

```text
benchmark_outputs/vico_class_conditioned_one_subject_five_offsets_bsc03_alpha01/
└── method/
    └── black_cat/
        ├── prompt_00/   # a black cat
        │   ├── seed_000.png
        │   ├── ...
        │   └── seed_004.png
        ├── prompt_01/   # first ViCo prompt
        ├── ...
        └── prompt_05/   # fifth ViCo prompt
```

The filenames identify the five independently optimized offsets. All five generations use inference seed 0 by default.

Offsets, reconstructions, timing JSON files, and logs are saved under `benchmark_work/vico_class_conditioned_one_subject_five_offsets_bsc03/<subject>/`.

## Compare against null-conditioned optimization

Run the matching null-text experiment with identical optimization and inference settings:

```bash
bash scripts/run_vico_null_conditioned_five_offsets.sh black_cat
```

The only intended conditioning difference is:

- class experiment: optimize and infer from `a black cat` as the base;
- null experiment: optimize and infer from Infinity's null text as the base.

Both use five offset seeds, one fixed inference seed, batch size 2, 200 steps, learning rate `5e-4`, BSC strength `0.3`, offsets 0–4, and prompt coefficient `0.1`.

## Useful overrides

```bash
OFFSET_SEEDS=1 STEPS=2 \
bash scripts/run_vico_class_conditioned_one_subject_five_offsets.sh black_cat
```

Set `REUSE_OFFSETS=0` to force retraining. Existing checkpoints are reused only when their conditioning prompt, scales, seed, steps, batch size, gradient accumulation, learning rate, and BSC settings match the current run. `BSC_LAYERS` and `BSC_STRENGTH` can be overridden for an ablation.
