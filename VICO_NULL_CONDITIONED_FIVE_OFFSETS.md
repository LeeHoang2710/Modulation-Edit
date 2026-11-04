# One-subject, five-offset null-conditioned ViCo experiment

This experiment returns offset optimization to the model's null-text input. It learns five independent offsets for one selected subject using optimization seeds 0–4. The default subject is `black_cat`.

Optimization uses:

- null-text conditioning with length 32;
- separate offsets on scales 0–4;
- batch size 2;
- gradient accumulation 1;
- 200 optimization steps;
- learning rate `5e-4`;
- bitwise self-correction (BSC) on histories from scales 0–3, with a newly sampled bit-flip probability in `[0, 0.3]` for every affected scale and step;
- no horizontal-flip augmentation.

The BSC corruption is applied only to the teacher-forced history. The supervised target bits remain clean. This prevents the offset from relying on perfect reference histories that are unavailable during autoregressive inference.

At inference, the modulation base remains Infinity's null text while the conditional prompt (for example, `a black cat ...`) enters prompt modulation and `ca_kv`. Scales 0–4 use:

\[
G_s=0.9G_{\text{null}}+1.0D_s+0.1G_{\text{prompt}}.
\]

The implementation computes the equivalent null-relative form:

\[
G_s=G_{\text{null}}+D_s+0.1(G_{\text{prompt}}-G_{\text{null}}).
\]

Scales 5–12 use \(G_{\text{prompt}}\), and the complete prompt enters conditional `ca_kv` at every scale.

Each offset generates six images with the same inference seed:

1. `a <subject name>` for identity reconstruction;
2. the first five official ViCo prompts containing the full subject name.

The default run therefore creates `5 offsets × 6 prompts = 30 images`.

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
bash scripts/run_vico_null_conditioned_five_offsets.sh
```

## Run another subject

Pass its ViCo folder name as the first argument:

```bash
bash scripts/run_vico_null_conditioned_five_offsets.sh cat_statue
bash scripts/run_vico_null_conditioned_five_offsets.sh alarm_clock
```

The output layout is:

```text
benchmark_outputs/vico_null_conditioned_five_offsets_bsc03_alpha01/
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

The filenames `seed_000.png` through `seed_004.png` identify the independently optimized offsets. All five use inference seed 0 by default.

Offsets, reconstructions, timing JSON files, and logs are saved under `benchmark_work/vico_null_conditioned_five_offsets_bsc03/<subject>/`. The separate BSC path prevents an older pure-teacher-forcing checkpoint from being mistaken for this experiment.

## Useful overrides

Run a short pipeline check:

```bash
OFFSET_SEEDS=1 STEPS=2 \
bash scripts/run_vico_null_conditioned_five_offsets.sh black_cat
```

Force retraining instead of reusing validated checkpoints:

```bash
REUSE_OFFSETS=0 bash scripts/run_vico_null_conditioned_five_offsets.sh black_cat
```

You can also override `INFER_SEED`, `EARLY_CFG`, `LATE_CFG`, `BSC_LAYERS`, `BSC_STRENGTH`, `WORK_ROOT`, or `OUTPUT_ROOT`. Changing the optimization hyperparameters is supported, but the launcher validates them before reusing an existing checkpoint.

To reproduce the previous pure-teacher-forcing baseline without BSC, use a separate path:

```bash
BSC_LAYERS=0 \
WORK_ROOT=benchmark_work/vico_null_conditioned_five_offsets_no_bsc \
OUTPUT_ROOT=benchmark_outputs/vico_null_conditioned_five_offsets_no_bsc_alpha01 \
bash scripts/run_vico_null_conditioned_five_offsets.sh black_cat
```
