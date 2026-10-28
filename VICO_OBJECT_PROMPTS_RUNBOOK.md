# Run Modulation-Edit with object-specific ViCo prompts

This runbook uses the active Modulation-Edit method only. It avoids overly generic nouns for compound object concepts:

| Subject folder | Prompt noun inserted into `{}` |
| --- | --- |
| `cat_statue` | `cat statue` |
| `duck_toy` | `duck toy` |
| `elephant_statue` | `elephant statue` |

For example, a ViCo template now becomes `a cat statue wearing a red hat`, not `a statue wearing a red hat`. The specific subject appearance is still learned only from the reference images through the modulation offsets.

## 1. Clone the correct branch

On the Vast.ai server:

```bash
cd /workspace
git clone --branch five-model-dreambooth-benchmark \
  https://github.com/LeeHoang2710/Modulation-Edit.git
cd Modulation-Edit
```

If the repository is already present:

```bash
cd /workspace/Modulation-Edit
git switch five-model-dreambooth-benchmark
git pull origin five-model-dreambooth-benchmark
```

Use a CUDA development Vast.ai image. This project compiles FlashAttention, so these must succeed before setup:

```bash
nvcc --version
nvidia-smi
```

## 2. Build the Modulation-Edit environment

```bash
cd /workspace/Modulation-Edit
bash scripts/setup_vico_vastai.sh
```

For interactive commands, activate it if desired:

```bash
conda activate vico-modulation
```

## 3. Download ViCo inputs and frozen Infinity weights

Set `HF_TOKEN` only if Hugging Face asks for authentication while downloading Flan-T5 or Infinity files.

```bash
cd /workspace/Modulation-Edit
export HF_TOKEN='your_huggingface_token'
bash scripts/prepare_vico_assets.sh
```

The asset script downloads `data/vico/` and creates `benchmark/manifests/vico.json`.

## 4. Regenerate the manifest with the new object mapping

Run this after copying or pulling the updated `benchmark/prepare_vico.py`. It replaces an old manifest that still contains `class_name: "statue"` for `cat_statue`.

```bash
cd /workspace/Modulation-Edit
/venv/vico-modulation/bin/python benchmark/prepare_vico.py --inputs data/vico
```

If your environment is Conda-based rather than `/venv/vico-modulation`, use:

```bash
conda run --no-capture-output -n vico-modulation \
  python benchmark/prepare_vico.py --inputs data/vico
```

Confirm the new mapping:

```bash
python - <<'PY'
import json
manifest = json.load(open('benchmark/manifests/vico.json'))
for item in manifest['subjects']:
    if item['subject'] in {'cat_statue', 'duck_toy', 'elephant_statue'}:
        print(item['subject'], '->', item['class_name'])
PY
```

Expected:

```text
cat_statue -> cat statue
duck_toy -> duck toy
elephant_statue -> elephant statue
```

## 5. Smoke test: one object, all prompts, one learned offset

This trains exactly one five-scale offset for `cat_statue` and uses it for all 31 prompts. It uses the linear alpha schedule `[0.25, 0.20, 0.15, 0.10, 0.05]`, prompt cross-attention at every scale, and late conditional modulation from `G_prompt`. There is only one generated candidate per prompt, so its `mean8` evaluation is simply that candidate's score.

```bash
cd /workspace/Modulation-Edit
bash scripts/run_ours_vico_object_prompt_smoke.sh
```

### Alpha-schedule ablations

`ALPHA_SCHEDULE` is a comma-separated five-value list for learned scales
`0, 1, 2, 3, 4`, respectively. Use a separate output/work/report directory
for each ablation so generated images and metric CSVs are not overwritten.

Equal alpha `0.15` at all five scales:

```bash
cd /workspace/Modulation-Edit
ALPHA_SCHEDULE='0.15,0.15,0.15,0.15,0.15' \
RESULTS=benchmark_outputs_vico_cat_statue_alpha_equal015 \
WORK_DIR=benchmark_work/vico_cat_statue_alpha_equal015 \
REPORT_DIR=benchmark_reports_vico_cat_statue_alpha_equal015 \
bash scripts/run_ours_vico_object_prompt_smoke.sh
```

Linear increasing alpha from `0.05` to `0.25`:

```bash
cd /workspace/Modulation-Edit
ALPHA_SCHEDULE='0.05,0.10,0.15,0.20,0.25' \
RESULTS=benchmark_outputs_vico_cat_statue_alpha_increasing \
WORK_DIR=benchmark_work/vico_cat_statue_alpha_increasing \
REPORT_DIR=benchmark_reports_vico_cat_statue_alpha_increasing \
bash scripts/run_ours_vico_object_prompt_smoke.sh
```

Inspect the rendered prompts:

```bash
find benchmark_outputs_vico_cat_statue_object_prompt_smoke -name prompt.txt -print -exec cat {} \;
```

Outputs:

```text
benchmark_outputs_vico_cat_statue_object_prompt_smoke/
benchmark_reports_vico_cat_statue_object_prompt_smoke/summary.csv
benchmark_reports_vico_cat_statue_object_prompt_smoke/per_prompt_metrics.csv
```

## 6. Full 16-subject run

Current method configuration:

- First five scales: `0–4`.
- Eight independent offset optimizations per subject, one fixed inference seed per offset.
- `50` optimization steps; batch size `5`; gradient accumulation `1`; learning rate `4e-3`.
- Linear alpha schedule `[0.25, 0.20, 0.15, 0.10, 0.05]` at scales `0–4`; early CFG `1.2`; late `G_prompt` CFG `2.0`; prompt cross-attention at all scales.
- Evaluation: `mean8`, i.e. the standard mean across all eight independently optimized offset outputs per subject-prompt pair; no oracle best-DINO selection.
- Terminal output is limited to a `128`-offset progress bar, then a `3,968`-image progress bar. Detailed optimizer and per-image output is stored in `benchmark_work/.../logs/`.

```bash
cd /workspace/Modulation-Edit
MAX_PROMPTS=0 \
FIXED_INFERENCE_SEED=0 \
RESULTS=benchmark_outputs_vico_object_prompts \
WORK_DIR=benchmark_work/vico_object_prompts \
REPORT_DIR=benchmark_reports_vico_object_prompts \
bash scripts/run_ours_vico_multi_offset_fixed_seed.sh
```

`MAX_PROMPTS=0` means all 31 ViCo templates. The runner defaults to the same linear alpha schedule, early CFG, late CFG, prompt cross-attention, and `G_prompt` formulation as the smoke test. The run creates `16 × 31 × 8 = 3,968` images and evaluates their mean score. `summary.csv` also reports preparation time per offset and pure inference time per image, each as mean ± population standard deviation across all measured offset-level samples.

## 7. Non-oracle mean-of-eight ablation

This reports the mean metric over all eight independently optimized offsets rather than selecting the highest-DINO output:

```bash
cd /workspace/Modulation-Edit
SELECTION=mean8 \
SUBJECTS=cat_statue MAX_PROMPTS=5 \
RESULTS=benchmark_outputs_vico_cat_statue_mean8 \
WORK_DIR=benchmark_work/vico_cat_statue_mean8 \
REPORT_DIR=benchmark_reports_vico_cat_statue_mean8 \
bash scripts/run_ours_vico_multi_offset_fixed_seed.sh
```

Do not compare `mean8` directly with `best-dino`: best-DINO selects an output after generation using reference-image DINO similarity, so it is an oracle metric.
