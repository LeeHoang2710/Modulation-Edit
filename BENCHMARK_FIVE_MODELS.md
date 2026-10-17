# Five-model DreamBooth benchmark

This branch is `five-model-dreambooth-benchmark`. It keeps the five upstream implementations as pinned Git submodules:

| Folder | Method | Backbone / protocol |
| --- | --- | --- |
| `models/ARBooth` | ARBooth | Infinity-2B; all subject references |
| `models/CSD-VAR` | CSD-VAR | Switti; one deterministic reference (`00` / first file), because the released method is single-reference |
| `models/DreamBooth` | DreamBooth | Stable Diffusion v1.4; all subject references |
| `models/Textual-Inversion` | Textual Inversion | original Latent Diffusion; all subject references |
| `models/UnZipLoRA` | UnZipLoRA | SDXL; all subject references |

The CSD-VAR release currently provides the **Switti** implementation, not its unreleased Infinity-2B version. Its single-reference restriction is intentionally recorded in every row of the report. Therefore, compare it as a single-image personalization baseline, not as an equal few-shot method.

## Vast.ai setup

Clone the benchmark branch, enter it, and initialise the pinned baseline repositories:

```bash
git clone --branch five-model-dreambooth-benchmark --recurse-submodules \
  https://github.com/LeeHoang2710/Modulation-Edit.git
cd Modulation-Edit
git submodule update --init --recursive
```

Then initialise the environments and download the benchmark assets:

```bash
bash scripts/setup_five_model_benchmark_envs.sh
bash scripts/download_five_model_benchmark_assets.sh
```

`setup_five_model_benchmark_envs.sh` creates five isolated Conda environments because the original 2022 LDM code cannot safely share dependencies with modern Switti/SDXL code.

The legacy DreamBooth/Textual-Inversion environment intentionally uses
`pip<24.1`: its pinned `pytorch-lightning==1.5.9` has metadata that newer pip
versions reject. This is isolated to `benchmark-ldm` and does not downgrade the
other four environments.

ARBooth requires FlashAttention. The setup script installs PyTorch first and
then builds FlashAttention with PyTorch visible to the build process. If this
step reports that `nvcc` or a CUDA toolkit is missing, choose a Vast.ai CUDA
**development** image (not a runtime-only image) whose CUDA toolkit matches the
selected `CUDA_WHL` wheel, then rerun the setup script. Existing environments
are reused; do not delete them.

ARBooth also requires PyTorch 2.5+ because its released Infinity code imports
`torch.nn.attention.flex_attention`. The setup script pins its isolated
`benchmark-arbooth` environment to PyTorch 2.5.1.

The downloader obtains the original Google DreamBooth dataset, Infinity/ARBooth dependencies, Switti, VQ-VAE, and CSD evaluator weights. It does not commit assets. SDXL and the two legacy checkpoints may require accepting Hugging Face licenses. Before downloading SDXL, export an approved token:

```bash
export HF_TOKEN=hf_...
bash scripts/download_five_model_benchmark_assets.sh
```

Do not put `HF_TOKEN` in a tracked file. To persist it only for the active
Vast.ai shell, run the `export` command above. To remove it later:

```bash
unset HF_TOKEN
```

Place the separately licensed legacy files at:

```text
weights/stable-diffusion/sd-v1-4-full-ema.ckpt
weights/latent-diffusion/model.ckpt
```

## Run

Smoke-test one subject first:

```bash
SUBJECTS=dog IMAGES_PER_PROMPT=1 bash scripts/run_five_model_dreambooth_benchmark.sh
```

Run all 30 DreamBooth subjects with the standard four images per prompt:

```bash
bash scripts/run_five_model_dreambooth_benchmark.sh
```

Choose a subset if needed:

```bash
MODELS=arbooth,unziplora SUBJECTS=dog,cat IMAGES_PER_PROMPT=4 \
  bash scripts/run_five_model_dreambooth_benchmark.sh
```

This is sequential by design. Five methods × 30 subjects × 25 prompts is an expensive multi-day run; each model is fully prepared per subject before inference.

## Saved outputs and metrics

Every image is normalized to:

```text
benchmark_outputs/<model>/<subject>/<prompt-slug>/seed_000.png
benchmark_outputs/<model>/<subject>/<prompt-slug>/prompt.txt
benchmark_outputs/<model>/<subject>/timing.json
```

The evaluator writes:

```text
benchmark_reports/per_image_metrics.csv
benchmark_reports/per_subject_metrics.csv
benchmark_reports/summary.csv
```

`summary.csv` contains exactly the requested columns: `preparation_seconds_per_object`, `inference_seconds_per_image`, `DINO`, `CLIP-I` (`clip_i`), and `CLIP-T` (`clip_t`). DINO uses ViT-S/16; CLIP-I/CLIP-T use CLIP ViT-B/32, the standard DreamBooth metric encoders.

The metric encoders are shared across all backbones. This makes the metrics comparable as a common reference, though it does not eliminate inherent backbone differences between Infinity, Switti, Stable Diffusion and SDXL.
