# CSD-VAR Switti on ViCo

This adapter uses the released `models/CSD-VAR` Switti code and evaluates with
`benchmark/evaluate_vico_arbooth_protocol.py`, separately from ARBooth and our
method. It does not modify upstream CSD-VAR.

**Protocol:** CSD-VAR receives the first sorted ViCo reference image only. Its
content token is initialized with the generic class noun; a generic `photo`
style token and ten generic class descriptions are used for its required
training captions and text-projection inputs. This adaptation is not the
same as giving our method all subject references, and it is not the upstream
CSD-VAR benchmark. Record it when comparing results.
The ten descriptions are heuristic generic prompts, not the curated captions
from the authors' datasets, so their effect on CSD-VAR quality must be tested.
At inference, a ViCo template such as `a {} in snow` becomes
`a <learned content tokens> dog in snow` for a dog subject.

The default upstream settings train 200 iterations and evaluate checkpoint
150. Inference generates eight images per prompt using the upstream batch
sampler from a single RNG seed; those are not eight independently seeded runs.
Images are saved as `subject/prompt_XX/seed_000.png` etc. The evaluator reports
I-DINO, I-CLIP, T-CLIP, and timing. Since upstream inference is invoked as a
subprocess, only its end-to-end wall time per image is reported: it includes
model loading and image I/O, so it is not directly comparable to pure
autoregressive inference time. See `end_to_end_seconds_per_image` in the
report; `inference_seconds_per_image` is deliberately left empty.

## Prerequisites

On a CUDA-capable Vast.ai server, clone the benchmark branch and enter the
repository:

```bash
git clone --branch five-model-dreambooth-benchmark https://github.com/LeeHoang2710/Modulation-Edit.git
cd Modulation-Edit
```

From the project root, initialize the
CSD-VAR submodule and create its separate environment:

```bash
bash scripts/setup_csd_var_vico.sh
```

This uses Conda and CUDA 12.1 PyTorch wheels by default. It does not reinstall
ARBooth or our method. It reuses an existing `benchmark-csdvar` environment if
present; setting `CSD_ENV_NAME` selects a different Conda environment.

The same CSD-VAR environment is used for evaluation by default. Then download
ViCo, the CSD-VAR Switti/VAE/text weights and the evaluator weights:

```bash
bash scripts/prepare_csd_var_vico_assets.sh
```

The Hugging Face model files use the Hugging Face cache; set `HF_HOME` before
both preparation and benchmark execution to use the same alternate cache.
These models are large and need substantial disk space and network access.
The setup and full GPU pipeline have not yet been tested on your Vast.ai
instance.

Set `CSD_PYTHON` to the CSD-VAR environment's Python if needed. If
`/venv/benchmark-csdvar/bin/python` exists it is used automatically; otherwise
the named Conda environment is used. To evaluate with another environment,
set `EVAL_PYTHON` to its Python executable or set `EVAL_ENV_NAME` to its Conda
name; it must include PyTorch, torchvision and OpenAI CLIP.

## Smoke test: one subject, five prompts, eight samples each

```bash
SUBJECTS=brown_dog MAX_PROMPTS=5 IMAGES_PER_PROMPT=8 \
  RESULTS=benchmark_outputs_vico_csd_var_smoke \
  WORK_DIR=benchmark_work/vico_csd_var_smoke \
  REPORT_DIR=benchmark_reports_vico_csd_var_smoke \
  bash scripts/run_csd_var_vico.sh
```

## Full benchmark: 16 subjects × 31 prompts × 8 images

```bash
bash scripts/run_csd_var_vico.sh
```

For a quicker diagnostic run set `IMAGES_PER_PROMPT=1`; this checks the
pipeline but does not test the eight-image inference batch. `SEED=42` is the
single upstream RNG initialization, not a list of eight independent seeds.

The script refuses to overwrite existing subject outputs. Use fresh `RESULTS`
and `WORK_DIR` values for a rerun, including after an interrupted inference.
Sample filenames `seed_000.png` etc. are evaluator-compatible indices, not
claims that individual RNG seeds 0–7 were set. The image folder and numerical reports are:

```text
benchmark_outputs_vico_csd_var/
benchmark_reports_vico_csd_var/summary.csv
benchmark_reports_vico_csd_var/per_subject_metrics.csv
benchmark_reports_vico_csd_var/per_prompt_metrics.csv
```

CLIP-T defaults to the same *unrendered* ViCo templates as the released ARBooth
evaluator; this differs from evaluating the human-readable rendered prompt.
