# ViCo: released ARBooth vs. eight independently optimized modulation offsets

This guide runs two separate ViCo benchmarks and keeps their results separate.

| Protocol | Subject representation | Images per concept-prompt pair |
| --- | --- | --- |
| ARBooth | one released checkpoint | eight inference seeds `0...7` |
| Our strategy | eight independently optimized offsets (`train_seed000...007`) | one shared inference seed (`g_seed=0`) per offset |

Both protocols save eight images per concept-prompt pair. They use the same
ARBooth-style DINO, CLIP-I, and CLIP-T evaluation implementation, but the two
rows should **not** be called the same sampling protocol: ours measures offset
optimization variation, whereas ARBooth measures generation-seed variation.

Our current strategy learns only scales `0...4`, with `batch_size=5`,
`grad_accum=1`, `steps=50`, and `lr=4e-3`. At inference it uses fixed
`alpha=[0.1]*5`, early CFG `1.0`, prompt CA-KV at all 13 scales, and late
`G_prompt` with CFG `2.0`.

## 1. Clone the branch on Vast.ai

Create a CUDA **development** instance; `nvcc --version` must work because
FlashAttention is compiled during setup. An A6000-class GPU or a GPU with at
least 24 GB VRAM is recommended.

```bash
cd /workspace
git clone --recurse-submodules --branch five-model-dreambooth-benchmark \
  https://github.com/LeeHoang2710/Modulation-Edit.git
cd Modulation-Edit
```

## 2. Install the two environments

```bash
bash scripts/setup_vico_arbooth_and_modulation_envs.sh
```

For a CUDA 11.8 development image:

```bash
CUDA_WHL=cu118 bash scripts/setup_vico_arbooth_and_modulation_envs.sh
```

This creates:

- `arbooth-vico` — released ARBooth inference and ARBooth-protocol scoring.
- `vico-modulation` — the five-scale offset optimizer and scoring.

The installer initializes ARBooth's Git submodule automatically. If an earlier
run was interrupted, simply rerun the same command and wait for the final
`Ready:` line. The pip warning about running as `root` is informational inside
the Vast container.

## 3. Download data and all weights once

Set a Hugging Face **read** token only in the active shell:

```bash
export HF_TOKEN='your_huggingface_read_token'
bash scripts/prepare_vico_arbooth_and_modulation_assets.sh
```

It downloads the ViCo references and 31 templates, Infinity-2B, Infinity VAE,
Flan-T5-XL, all 16 released ARBooth subject checkpoints, DINO ViT-S/16, and
OpenAI CLIP ViT-B/32.

## 4. Separate smoke tests

ARBooth: one subject, five prompts, one inference seed:

```bash
bash scripts/run_arbooth_vico_smoke.sh
```

Our strategy: one subject, five prompts, eight independently learned offsets,
each generated with exactly the same inference seed:

```bash
ALPHA_SCHEDULE='0.25,0.20,0.15,0.10,0.05' \
bash scripts/run_ours_vico_multi_offset_smoke.sh
```

Each smoke script saves its generated images plus `summary.csv`, which reports
I-DINO, I-CLIP, T-CLIP, and pure inference seconds/image.

## 5. Full 16 × 31 runs

ARBooth: 16 subjects × 31 prompts × eight inference seeds = 3,968 images.

```bash
bash scripts/run_arbooth_vico_paper_protocol.sh
```

Our strategy: 16 subjects × eight fitted offsets × 31 prompts × one shared
inference seed = 3,968 images.

```bash
ALPHA_SCHEDULE='0.25,0.20,0.15,0.10,0.05' \
bash scripts/run_ours_vico_multi_offset_fixed_seed.sh
```

The ARBooth results are saved to:

```text
benchmark_outputs_vico_arbooth/
benchmark_reports_vico_arbooth/summary.csv
```

The multi-offset results are saved to:

```text
benchmark_outputs_vico_multi_offset8_seed0/
benchmark_reports_vico_multi_offset8_seed0/summary.csv
```

## 6. Run remotely and download automatically to your Mac

Run the following commands from your **Mac checkout**, not inside Vast.ai.
They execute the selected server command, wait for it to finish, then copy its
image and metric directories to `~/Downloads/` using `rsync`.

```bash
export VAST_HOST='ssh4.vast.ai'
export VAST_PORT='your_vast_ssh_port'
export VAST_REPO_DIR='/workspace/Modulation-Edit'
```

Smoke tests:

```bash
bash scripts/run_vico_remote_and_download.sh arbooth smoke
bash scripts/run_vico_remote_and_download.sh ours smoke
```

Full benchmarks:

```bash
bash scripts/run_vico_remote_and_download.sh arbooth full
bash scripts/run_vico_remote_and_download.sh ours full
```

Optional overrides for our strategy preserve the same method but adjust cost:

```bash
STEPS=100 BATCH_SIZE=5 LR=4e-3 FIXED_INFERENCE_SEED=0 \
  bash scripts/run_ours_vico_multi_offset_fixed_seed.sh
```

Do not overwrite existing results with a different configuration. Set unique
`RESULTS`, `WORK_DIR`, and `REPORT_DIR` names for every independent run.
