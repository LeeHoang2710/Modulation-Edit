# ARBooth / ViCo benchmark

This is the active personalization benchmark for this branch. It uses the 16
ViCo concepts bundled in `models/ARBooth/inputs/`, not the historical Google
DreamBooth dataset.

- 16 concepts: 5 live and 11 non-live.
- 31 official templates per concept.
- 8 samples per template for a full run: 3,968 images per method.
- References are the original 4--12 images inside each concept folder.

## Build the manifest

```bash
conda run --no-capture-output -n benchmark-arbooth python benchmark/prepare_vico.py
```

This writes a deterministic, ignored `benchmark/manifests/vico.json`. Templates
remain as `a {} ...` so every method can apply its own subject-token convention.

## Run Modulation-Edit

Smoke test one concept:

```bash
SUBJECTS=brown_dog IMAGES_PER_PROMPT=1 bash scripts/run_ours_vico_benchmark.sh
```

Full paper-sized run:

```bash
bash scripts/run_ours_vico_benchmark.sh
```

The runner fits one null-text per-scale AdaLN offset from **all** ViCo
references of each subject, then generates the 31 prompts with the selected
combined modulation/cross-attention configuration. It saves normalized images
under `benchmark_outputs_vico/modulation_offset/` and supplementary DINO,
CLIP-I, CLIP-T, preparation-time, and inference-time CSVs under
`benchmark_reports_vico/`.

For an exact ARBooth paper-table comparison, also use upstream ARBooth's
official evaluator on its required `concept/prompt_XX/` output layout; it adds
LPIPS diversity. Released baselines must remain inference-only: ARBooth
checkpoints, DreamMatcher's released TI/DreamBooth/Custom Diffusion artifacts,
ViCo's released eight-subject subset, and ELITE's mapper plus masks.
