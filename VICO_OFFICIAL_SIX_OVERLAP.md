# Official ViCo reproduction for the six exact overlaps

This reproduces the released **ViCo** model for the six objects whose reference
images are byte-identical to six objects in our current 16-object benchmark:

| Official ViCo | Current benchmark | ViCo class word |
|---|---|---|
| `barn` | `wooden_barn` | `barn` |
| `clock` | `alarm_clock` | `clock` |
| `dog7` | `brown_dog2` | `dog` |
| `monster_toy` | `monster_toy` | `toy` |
| `pink_sunglasses` | `pink_sunglasses` | `sunglasses` |
| `teddybear` | `brown_teddybear` | `teddybear` |

The runner uses the official ViCo inference code unchanged. It uses the first
matching reference image, the released checkpoint at `load_step=399`, the
official `*` placeholder, 50 DDIM steps, CFG 7.5, and four samples per
prompt. The result layout is normalized to:

```text
benchmark_outputs_vico_official_six_overlap/
  barn/prompt_00/seed_000.jpg ... seed_003.jpg
  ...
```

This keeps ViCo outputs separate from our Modulation-Edit outputs so they can
be placed side-by-side in the paper. It does not train ViCo again.

## 1. Prepare source, checkpoints, and references

Activate the existing ViCo/metrics environment on Vast.ai, then run:

```bash
cd /workspace/Modulation-Edit
bash scripts/prepare_vico_official_overlap_assets.sh
```

The script clones the official source into `models/ViCo`, downloads the public
ViCo checkpoint folder from the link used by the official repository, downloads
Stable Diffusion v1-4 from Hugging Face, and ensures `data/vico` plus the
manifest exist. Set `HF_TOKEN` only if the base checkpoint asks for Hugging
Face authentication.

## 2. Smoke test: one object and five prompts

```bash
cd /workspace/Modulation-Edit
SUBJECTS=dog7 MAX_PROMPTS=5 NUM_SAMPLES=4 \
bash scripts/run_vico_official_six_overlap.sh
```

Images are written to:

```text
benchmark_outputs_vico_official_six_overlap/dog7/
```

## 3. Run all six objects and all 31 prompts

```bash
cd /workspace/Modulation-Edit
SUBJECTS=all MAX_PROMPTS=0 NUM_SAMPLES=4 \
bash scripts/run_vico_official_six_overlap.sh
```

This creates `6 × 31 × 4 = 744` images. The same metrics evaluator used
for our ARBooth-protocol comparison can be run afterward if desired, but keep
the official ViCo images in their own report/output directory.

## Checkpoint matching

The runner searches `weights/vico` for a directory containing both
`checkpoints/cross_attention-399.pt` and `checkpoints/embeddings_gs-399.pt`,
then matches the directory name to the official object name. If the Drive
download creates an extra nesting level, no manual move is needed.

Official source: [haoosz/ViCo](https://github.com/haoosz/ViCo). Its README
documents the public checkpoint folder and the `vico_txt2img.py` inference
arguments.
