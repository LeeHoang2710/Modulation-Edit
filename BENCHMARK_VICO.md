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
references of each subject. Its default inference is the current method:

\[
G_s=(1-\alpha_s)(G_{null}+D_s)+\alpha_sG_{prompt},\quad
\alpha_{0:4}=[.25,.20,.15,.10,.05],\quad cfg=1.2
\]

for scales 0--4; scales 5--12 use \(G_{prompt}\) with `cfg=2.0`. Prompt
cross-attention (`ca_kv`) is used at all scales. It saves normalized images
under `benchmark_outputs_vico/modulation_offset/` and supplementary DINO,
CLIP-I, CLIP-T, preparation-time, and inference-time CSVs under
`benchmark_reports_vico/`.

The default optimizer supervises and learns offsets only on scales `0-4`;
override `LOSS_SCALES` and `OFFSET_INFER_SCALES` together only for an explicit
ablation.

The late-modulation ablation preserves the same early schedule, CFG values,
and prompt `ca_kv`, but sets scales 5--12 to \(G_{null}\):

```bash
LATE_GSS=null bash scripts/run_ours_vico_benchmark.sh
```

The script automatically writes this to a distinct output model directory
(`modulation_offset_late_null`) so it cannot overwrite the `G_prompt` result.

For an exact ARBooth paper-table comparison, also use upstream ARBooth's
official evaluator on its required `concept/prompt_XX/` output layout; it adds
LPIPS diversity. Released baselines must remain inference-only: ARBooth
checkpoints, DreamMatcher's released TI/DreamBooth/Custom Diffusion artifacts,
ViCo's released eight-subject subset, and ELITE's mapper plus masks.
