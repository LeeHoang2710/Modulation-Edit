# Reproduce ARBooth’s ViCo Table 1 protocol and compare our method

This guide runs two independent 16-concept ViCo experiments on the same
output/evaluation protocol:

1. the released ARBooth checkpoints;
2. our current modulation method: `G_prompt`, fixed early alpha `0.1`, early
   CFG `1.0`, late CFG `2.0`, and prompt cross-attention at every scale.

Both produce **16 concepts × 31 prompts × 8 samples = 3,968 images**. The
eight samples are seeds `0..7` from a single subject representation, matching
ARBooth's released `run_arbooth_for_eval.py`. They are not eight independently
trained checkpoints.

## 1. Create the Vast.ai environment

Use a CUDA *development* image: `nvcc --version` must work because Infinity
requires FlashAttention. An A6000-class GPU is the most comparable hardware
for the ARBooth timing number.

```bash
cd /workspace
git clone --branch five-model-dreambooth-benchmark \
  https://github.com/LeeHoang2710/Modulation-Edit.git
cd Modulation-Edit
bash scripts/setup_vico_vastai.sh
```

For CUDA 11.8, use matching PyTorch wheels:

```bash
CUDA_WHL=cu118 bash scripts/setup_vico_vastai.sh
```

## 2. Download all needed weights once

The ARBooth download includes all 16 released subject checkpoints (roughly
72 GB), along with Infinity-2B, the Infinity VAE, Flan-T5-XL, DINO ViT-S/16,
and OpenAI CLIP ViT-B/32. The ViCo reference images and 31 prompt templates
are also prepared under `data/vico/`.

```bash
export HF_TOKEN='your_read_token'
bash scripts/prepare_arbooth_vico_assets.sh
```

The script creates symlinks under `weights/arbooth/by_subject/<subject>/` so
the released ARBooth evaluator can load exactly one checkpoint per concept.

## 3. Reproduce the released ARBooth inference and scores

```bash
bash scripts/run_arbooth_vico_paper_protocol.sh
```

It preserves ARBooth's generator and records its printed `infinity cost` as
the per-image inference time. It evaluates the outputs with the released
metric implementation:

- DINO: `facebookresearch/dino:main`, `dino_vits16`;
- CLIP: OpenAI CLIP `ViT-B/32`;
- inputs resized directly to `256 × 256` using bicubic interpolation and
  represented in `[-1, 1]`;
- identity score: mean of every reference–generated feature dot product;
- concept score: mean across 31 prompts; final score: mean across 16 concepts.

For strict code-level reproduction, CLIP-T uses the original unrendered ARBooth
template (for example `a {} in the snow`). This is what
`models/ARBooth/tools/evaluate.py` passes to CLIP. It is different from a
rendered class prompt, even though the paper describes CLIP-T as excluding the
placeholder.

Compare the observed summary with Table 1:

```bash
conda run --no-capture-output -n vico-modulation python \
  benchmark/compare_arbooth_paper_row.py \
  --summary benchmark_reports_vico_arbooth/summary.csv
```

The paper reports `I-DINO=0.705`, `I-CLIP=0.824`, `T-CLIP=0.253`, and
`0.5 s/image`. The time is hardware- and environment-dependent; assess metric
reproduction primarily using the three similarity scores.

## 4. Run our fixed-alpha method under the same 3,968-image protocol

```bash
bash scripts/run_ours_vico_arbooth_protocol.sh
```

The runner first fits exactly one scale-0–4 offset per concept. It then uses
that saved offset for all 31 prompts and inference seeds `0..7`. At scales
0–4 it uses fixed alpha `[0.1, 0.1, 0.1, 0.1, 0.1]` with CFG `1.0`; at scales
5–12 it uses `G_prompt` with CFG `2.0`. Prompt cross-attention remains active
at all 13 scales.

Its output and report are separate:

```text
benchmark_outputs_vico_ours_alpha01/modulation_offset_late_prompt/
benchmark_reports_vico_ours_alpha01/summary.csv
```

Compare it against the same published reference row:

```bash
conda run --no-capture-output -n vico-modulation python \
  benchmark/compare_arbooth_paper_row.py \
  --summary benchmark_reports_vico_ours_alpha01/summary.csv \
  --label 'Our fixed-alpha G_prompt method'
```

## Useful short smoke test

The exact released ARBooth script always runs all 16 subjects. For a quick
test of our full data flow, use one concept and five prompts, while retaining
eight generated samples per prompt:

```bash
SUBJECTS=brown_dog MAX_PROMPTS=5 \
RESULTS=benchmark_outputs_vico_ours_smoke \
WORK_DIR=benchmark_work/vico_modulation_ours_smoke \
REPORT_DIR=benchmark_reports_vico_ours_smoke \
bash scripts/run_ours_vico_arbooth_protocol.sh
```

This creates `1 × 5 × 8 = 40` images and requires the evaluator to find all
eight images in every prompt directory before it writes a score.
