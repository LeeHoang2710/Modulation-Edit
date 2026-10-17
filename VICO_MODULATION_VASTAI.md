# Run the current Modulation-Edit method on ViCo (16 objects × 31 prompts)

This is the active benchmark protocol for **our Modulation-Edit method only**.
It uses the ViCo data folders as input: 16 objects, each with reference
images and 31 prompt templates. It does not install, train, run, or evaluate
any of the five external baseline models.

For each selected object and each requested offset seed, the workflow is:

1. Fit and save one scale offset from all reference images, with null text and
   reconstruction supervision only at scales 0–4.
2. For every prompt, reuse that saved offset with prompt cross-attention at all
   scales. At scales 0–4 it uses
   \(G_s=(1-\alpha_s)(G_{null}+D_s)+\alpha_sG_{prompt}\), where
   \(\alpha=[0.25,0.20,0.15,0.10,0.05]\) and CFG is 1.2.
3. At scales 5–12, run both sequential ablations: `G_prompt` and `G_null`,
   with CFG 2.0. The cross-attention remains the text prompt for both.

`OFFSET_SEEDS=y` therefore fits and saves `x × y` offsets for `x` selected
objects. `SAMPLES_PER_OFFSET=z` then draws `z` inference seeds from every
saved offset, so each method has `y × z` generated images per object/prompt.

## 1. Create a Vast.ai instance and clone the branch

Use a **CUDA development** image, not a runtime-only image. `nvcc --version`
must work because this Infinity implementation builds FlashAttention. A GPU
with at least 24 GB VRAM is recommended.

```bash
cd /workspace
git clone --branch five-model-dreambooth-benchmark \
  https://github.com/LeeHoang2710/Modulation-Edit.git
cd Modulation-Edit
```

The active setup does not initialize any model submodules. The benchmark data
is fetched separately in step 3, and only its `inputs/` directory is retained.

## 2. Install the two required environments

```bash
bash scripts/setup_vico_vastai.sh
```

For a CUDA 11.8 machine, select a matching PyTorch wheel index:

```bash
CUDA_WHL=cu118 bash scripts/setup_vico_vastai.sh
```

## 3. Download benchmark assets and weights

Set your Hugging Face token only in the shell; do not place it in a tracked
file or commit it.

```bash
export HF_TOKEN='your_huggingface_read_token'
bash scripts/prepare_vico_assets.sh
```

This downloads only the ViCo input data (under `data/vico/`), frozen Infinity
weights, and Flan-T5 locally under `weights/infinity/`, then writes
`benchmark/manifests/vico.json`.

## 4. Smoke test: one subject and five prompts

This fits one offset for `brown_dog`, then runs both late-scale variants over
the first five official prompt templates. It prints the pure model generation
time per output image after metric evaluation.

```bash
SUBJECTS=brown_dog OFFSET_SEEDS=1 SAMPLES_PER_OFFSET=1 MAX_PROMPTS=5 \
  RESULTS=benchmark_outputs_vico_smoke REPORT_DIR=benchmark_reports_vico_smoke \
  bash scripts/run_ours_vico_two_variant.sh
```

Expected outputs:

```text
benchmark_work/vico_modulation/offsets/brown_dog/train_seed000/offset.pth
benchmark_outputs_vico_smoke/modulation_offset_late_prompt/brown_dog/prompt_00/seed_000.png
benchmark_outputs_vico_smoke/modulation_offset_late_null/brown_dog/prompt_00/seed_000.png
benchmark_reports_vico_smoke/modulation_offset_late_prompt/summary.csv
benchmark_reports_vico_smoke/modulation_offset_late_null/summary.csv
```

Each late-scale variant is evaluated as a separate model and receives its own
report directory:

```text
benchmark_reports_vico_smoke/modulation_offset_late_prompt/summary.csv
benchmark_reports_vico_smoke/modulation_offset_late_null/summary.csv
```

The terminal table and each variant's `summary.csv` report:

| Column | Meaning |
| --- | --- |
| `dino` / I-DINO ↑ | image identity similarity to all subject references |
| `clip_i` / I-CLIP ↑ | CLIP image identity similarity to the references |
| `clip_t` / T-CLIP ↑ | CLIP text alignment with the exact prompt |
| `inference_seconds_per_image` / Time ↓ | pure autoregressive generation time per saved image; model loading and offset fitting are excluded |

## 5. Run all 16 objects and all 31 prompts

The default is one independently fitted offset per object:

```bash
OFFSET_SEEDS=1 SAMPLES_PER_OFFSET=1 MAX_PROMPTS=0 bash scripts/run_ours_vico_two_variant.sh
```

To fit five independent offsets per object, save them locally, then infer from
each one for both variants:

```bash
OFFSET_SEEDS=5 SAMPLES_PER_OFFSET=1 MAX_PROMPTS=0 bash scripts/run_ours_vico_two_variant.sh
```

### Eight independent offsets with one shared inference seed

For an optimization-robustness experiment, fit eight independently initialized
five-scale offset bundles (`D_0` through `D_4`), then evaluate every bundle
with the same generation seed. This isolates offset-training variation from
sampling variation. It uses the fixed-alpha `G_prompt` formulation:
`alpha=[0.1]*5`, early CFG `1.0`, late `G_prompt` CFG `2.0`, and prompt
cross-attention at all scales. Output files remain unique (`seed_000.png` to
`seed_007.png`), but their number identifies the offset run—not the actual
generation seed. The subject `timing.json` records the exact index →
offset-training-seed → inference-seed mapping.

```bash
FIXED_INFERENCE_SEED=0 bash scripts/run_ours_vico_multi_offset_fixed_seed.sh
```

This is a multi-offset ensemble experiment, not a one-offset ARBooth-equivalent
comparison. It uses the exact ARBooth-style evaluator but should be reported
separately. Use the resulting eight outputs per prompt to report their mean
and standard deviation across offset-training seeds.

The two variants are always sequential by default:

```text
modulation_offset_late_prompt  # scales 5–12 use G_prompt, CFG=2.0
modulation_offset_late_null    # scales 5–12 use G_null,   CFG=2.0
```

To re-fit existing offsets rather than reuse the saved files, pass an extra
argument directly to the Python runner:

```bash
conda run --no-capture-output -n vico-modulation python \
  benchmark/run_ours_vico_two_variant.py --subjects brown_dog \
  --offset-seeds 1 --max-prompts 5 --force-optimize
```

## Useful configuration overrides

```bash
# Change the two late-scale branches or use only one for a focused rerun.
VARIANTS=prompt bash scripts/run_ours_vico_two_variant.sh
VARIANTS=null bash scripts/run_ours_vico_two_variant.sh

# Keep the current formulation but change optimization length or learning rate.
STEPS=100 LR=1e-3 OFFSET_SEEDS=3 bash scripts/run_ours_vico_two_variant.sh
```

All generated images remain under `benchmark_outputs_vico/`; detailed image,
per-subject, and aggregate numerical results remain under
`benchmark_reports_vico/`.

For the exact ARBooth 16 × 31 × 8 output and evaluation protocol, use
[ARBOOTH_PROTOCOL_REPRODUCTION.md](ARBOOTH_PROTOCOL_REPRODUCTION.md). It runs
the released ARBooth checkpoints and our fixed-alpha comparison independently,
using ARBooth's own DINO/CLIP metric implementation.

Use a different `RESULTS` and `REPORT_DIR` name for an independent rerun with
a different number of offset seeds; this prevents old samples from being mixed
into a new metric report.
