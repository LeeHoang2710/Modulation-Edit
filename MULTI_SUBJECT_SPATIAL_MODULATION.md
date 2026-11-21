# Two-subject spatial SharedAdaLN inference

`tools/infer_two_offset_spatial.py` extends the main scale-scheduled variant to
two independently learned subject offsets. At each early scale, it constructs
one token-aligned SharedAdaLN tensor:

```text
[ left subject transformation | right subject transformation ]
```

All six vectors are placed spatially together. In the repository's internal
ordering they are `(SA gate, FFN gate, SA scale, FFN scale, SA shift, FFN
shift)`. The transformer then runs one global self-attention call over the full
token grid. The attention output keeps the same token order, so the left and
right gate vectors automatically multiply the matching output positions.

For odd-width scales, the token column crossed by the boundary receives an
overlap-weighted mixture. At the one-token coarsest scale, the default 50/50
split therefore uses the average of the two transformations; it is impossible
to represent two spatial halves with only one token.

## Example

```bash
/venv/vico-modulation/bin/python tools/infer_two_offset_spatial.py \
  --left-offset-path output/left_subject/offset.pth \
  --right-offset-path output/right_subject/offset.pth \
  --prompt "a dog and a cat sitting together in a garden" \
  --output-dir output/two_subjects/dog_cat \
  --model_path weights/infinity_2b_reg.pth \
  --vae_type 32 \
  --vae_path weights/infinity_vae_d32reg.pth \
  --text_encoder_ckpt weights/flan-t5-xl \
  --use_bit_label 1 \
  --pn 1M
```

Class-conditioned checkpoints provide their base prompts through
`train_prompt`. For older checkpoints, pass `--left-base-prompt` and
`--right-base-prompt`. `--split-ratio 0.5` is the default; values below or above
0.5 assign less or more width to the left subject.

The default prompt-weight schedule and CFG values match
`tools/infer_offset_vico_scale_schedule.py`. Use `--offset-scales` to restrict
injection further; `auto` never injects a checkpoint outside its recorded
training scales.
