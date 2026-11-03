# Class-conditioned ViCo offset experiment

The optimizer uses the full subject name as its class caption, such as `a cat statue` or `a alarm clock`, for every reference image. It learns a separate offset \(D_s\) on each of scales **0–4**. It does not use null-text conditioning during this optimization.

At inference, scales 0–4 use fixed \(\alpha=0.1\):

\[
G_s=0.9(G_{\text{class}}+D_s)+0.1G_{\text{prompt}}.
\]

Scales 5–12 use \(G_{\text{prompt}}\). The full prompt, such as `a cat statue on the beach`, is supplied to conditional cross-attention (`ca_kv`) at every scale. The CFG partner uses the class branch, so early guidance compares the blended full-prompt branch against \(G_{\text{class}}+D_s\).

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

The checkpoint and reconstruction are under `benchmark_work/vico_class_conditioned/cat_statue/`. Generated images are under `benchmark_outputs/vico_class_conditioned_alpha01/cat_statue/method/cat_statue/`. The `fit.log` and `infer.log` files record both stages.

The launcher fixes alpha to `0.1` and injects offsets on all scales 0–4. You can still change `MAX_PROMPTS`, `STEPS`, `BATCH_SIZE`, `GRAD_ACCUM`, `LR`, or `INFER_SEED` through environment variables.
