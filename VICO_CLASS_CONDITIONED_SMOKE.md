# Class-conditioned ViCo offset experiment

The optimizer uses the full subject name as its class caption, such as `a cat statue` or `a alarm clock`, for every reference image. It fits one shared per-scale offset on **scales 0–4**. Inference uses the ViCo prompt templates filled with the full subject name (`a cat statue ...`) for cross-attention and prompt modulation. The class caption is the base modulation and CFG partner; the learned offset is blended with the full prompt on scales 0–4. Scales 5–12 use full-prompt modulation.

This changes the conditioning during fitting, so the new offset must be trained again. It is saved in a separate work directory.

## Setup on Vast.ai

```bash
cd /workspace
git clone --branch five-model-dreambooth-benchmark https://github.com/LeeHoang2710/Modulation-Edit.git
cd Modulation-Edit
bash scripts/setup_vico_vastai.sh
bash scripts/prepare_vico_assets.sh
```

## Run one object and five prompts

```bash
bash scripts/run_vico_class_conditioned_smoke.sh cat_statue
```

Use another ViCo subject name as the argument. The script turns its underscores into spaces, trains for 50 steps with batch size 5 and learning rate `4e-3`, and generates one image for each of five prompts using inference seed 0.

```bash
MAX_PROMPTS=31 STEPS=100 bash scripts/run_vico_class_conditioned_smoke.sh cat_statue
```

The checkpoint and reconstruction are under `benchmark_work/vico_class_conditioned/cat_statue/`. Generated images are under `benchmark_outputs/vico_class_conditioned/cat_statue/method/cat_statue/`. The `fit.log` and `infer.log` files record both stages.
