# Auto-crop one ViCo object and run a five-prompt smoke test

This workflow runs **our Modulation-Edit method only**. Pass one of the 16 ViCo folder names, such as `alarm_clock`, `cat_statue`, or `elephant_statue`. It detects the object in every reference photo, saves tight crops, fits **one offset on scales 0–4**, generates one image for each of the first five official prompts, and reports DINO, CLIP-I, and CLIP-T. It does not train or run any of the five baseline models.

## 1. Start a Vast.ai instance

Choose a CUDA **development** image with enough GPU memory for Infinity-2B and a CUDA toolkit for FlashAttention. Open the instance terminal and verify:

```bash
nvidia-smi
nvcc --version
conda --version
```

If `nvcc` is absent, choose a CUDA development image. If `conda` is absent, install Miniconda or use an instance where `/venv/vico-modulation/bin/python` is already configured.

## 2. Get the correct repository branch

For a new server checkout:

```bash
cd /workspace
git clone --branch five-model-dreambooth-benchmark \
  https://github.com/LeeHoang2710/Modulation-Edit.git
cd Modulation-Edit
```

For an existing checkout:

```bash
cd /workspace/Modulation-Edit
git fetch origin
git switch five-model-dreambooth-benchmark
git pull origin five-model-dreambooth-benchmark
```

**These auto-crop files are local changes until they are pushed.** Before running on Vast.ai, either push this branch or copy `benchmark/prepare_vico_object_crops.py`, `scripts/run_vico_object_crop_smoke.sh`, and this guide into the same paths on the server. The updated `scripts/prepare_vico_assets.sh` supports either Conda or `/venv/vico-modulation`.

## 3. Install our method's environment

On a Conda-based CUDA development image:

```bash
bash scripts/setup_vico_vastai.sh
conda run --no-capture-output -n vico-modulation python -c \
  'import torch, transformers, PIL; print(torch.__version__, transformers.__version__, torch.cuda.is_available())'
```

The last value must be `True`. The environment installer also provides the OpenAI CLIP evaluator. The crop detector uses the pinned `transformers` package and Pillow already used by this environment.

If the instance already has the project environment at `/venv/vico-modulation`, use that Python for the checks instead:

```bash
/venv/vico-modulation/bin/python -c \
  'import torch, transformers, PIL; print(torch.__version__, transformers.__version__, torch.cuda.is_available())'
```

## 4. Download ViCo references and frozen Infinity weights

If Hugging Face requires access, create a **read** token on your Hugging Face account and export it in the terminal. Do not put the token in a repository file.

```bash
export HF_TOKEN='paste_your_read_token_here'
bash scripts/prepare_vico_assets.sh
```

This downloads the ViCo `inputs` tree to `data/vico/`, Infinity-2B and the VAE to `weights/infinity/`, and Flan-T5-XL to `weights/infinity/flan-t5-xl/`. It also generates `benchmark/manifests/vico.json`. Verify the assets:

```bash
ls data/vico/cat_statue
ls weights/infinity/infinity_2b_reg.pth weights/infinity/infinity_vae_d32reg.pth
ls weights/infinity/flan-t5-xl
```

The first auto-crop run downloads the public [Grounding DINO tiny checkpoint](https://huggingface.co/IDEA-Research/grounding-dino-tiny) into the Hugging Face cache. It uses the subject's ViCo noun, for example `cat statue`, to find a bounding box. Later runs reuse the cached detector.

The first metric run also downloads DINO ViT-S/16 and OpenAI CLIP ViT-B/32 into `weights/metrics/`. To fetch them before starting the experiment:

```bash
conda run --no-capture-output -n vico-modulation python - <<'PY'
from pathlib import Path
import torch, clip
root = Path('weights/metrics').resolve()
torch.hub.set_dir(str(root / 'torchhub'))
torch.hub.load('facebookresearch/dino:main', 'dino_vits16', trust_repo=True)
clip.load('ViT-B/32', device='cpu', download_root=str(root / 'clip'))
print('Evaluator weights ready:', root)
PY
```

## 5. Crop and review one object before training

To create crops for `cat_statue` without starting the expensive offset fit:

```bash
CROP_ONLY=1 bash scripts/run_vico_object_crop_smoke.sh cat_statue
```

Inspect these files:

```text
data/vico_auto_cropped/cat_statue/                  # cropped PNG reference images
data/vico_auto_cropped/cat_statue_crop_preview.jpg # originals with red boxes beside crops
data/vico_auto_cropped/cat_statue_detections.json  # bounding boxes and detection scores
benchmark/manifests/cat_statue_auto_cropped.json  # one-object training manifest
```

The detector must find the object in **every** reference photo. If it cannot, the script stops before saving training crops. For a hard object, try a clearer detector phrase or a lower detection threshold, then inspect the preview:

```bash
CROP_ONLY=1 CROP_LABEL='cat figurine' BOX_THRESHOLD=0.18 \
  bash scripts/run_vico_object_crop_smoke.sh cat_statue
```

The crop detector is automatic but cannot guarantee the correct instance, especially for toys, statues, or photos with multiple similar objects. Review the preview before using the result in a paper.

## 6. Run the one-object, five-prompt smoke test

Examples:

```bash
bash scripts/run_vico_object_crop_smoke.sh cat_statue
bash scripts/run_vico_object_crop_smoke.sh elephant_statue
bash scripts/run_vico_object_crop_smoke.sh alarm_clock
```

Each command re-crops that object's references and trains **one** new offset with 50 steps, batch size 5, gradient accumulation 1, learning rate `4e-3`, and no horizontal flip. The loss and learned offset cover scales `0–4` only. At inference it uses prompt cross-attention at all scales, early alpha schedule `0.25,0.20,0.15,0.10,0.05`, early CFG `1.2`, and `G_prompt` with late CFG `2.0`. It generates one image per prompt at inference seed 0.

You can vary the inference seed or alpha schedule:

```bash
INFER_SEED=3 bash scripts/run_vico_object_crop_smoke.sh cat_statue
ALPHA_SCHEDULE='0.15,0.15,0.15,0.15,0.15' \
  bash scripts/run_vico_object_crop_smoke.sh cat_statue
```

Use a different `WORK_DIR`, `RESULTS`, and `REPORT_DIR` for each ablation so runs remain separate. The smoke launcher refits the offset every time.

## 7. Find the outputs

For `cat_statue`, the default locations are:

```text
benchmark_work/vico_crop_smoke/cat_statue/offsets/cat_statue/train_seed000/offset.pth
benchmark_work/vico_crop_smoke/cat_statue/logs/
benchmark_outputs/vico_crop_smoke/cat_statue/ours_cat_statue_auto_crop_smoke_late_prompt/cat_statue/prompt_00/seed_000.png
benchmark_outputs/vico_crop_smoke/cat_statue/ours_cat_statue_auto_crop_smoke_late_prompt/cat_statue/prompt_01/seed_000.png
benchmark_reports/vico_crop_smoke/cat_statue/summary.csv
```

The remaining images are in `prompt_02` through `prompt_04`. The evaluator scores them against the **original, uncropped** ViCo references, so the crop changes only our fitting inputs. Five prompts and one generated image per prompt make this a smoke test, not the 31-prompt benchmark result.
