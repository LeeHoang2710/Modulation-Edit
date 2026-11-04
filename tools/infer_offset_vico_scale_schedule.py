#!/usr/bin/env python3
"""Generate ViCo prompts with the current scale-scheduled modulation method.

For the default five learned early scales:
  s=0..4: (1-a_s) * (G_base + D_s) + a_s * G_prompt,
           a=[.1,.1,.1,.1,.1], CFG=1.2
  s=5..12: G_prompt (or the null-text ablation G_null), CFG=2.0

With --class-conditioned-base, G_base is the class prompt saved in the
checkpoint (for example, "a cat statue"). Otherwise G_base is G_null.

Conditional cross-attention ``ca_kv`` always uses the full prompt. The CFG
partner uses the base caption for both ``ca_kv`` and modulation: class text in
class-conditioned mode, or null text in the legacy mode.
"""
from __future__ import annotations

import argparse
import json
import os.path as osp
import sys
import time
from pathlib import Path

sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__))))

import cv2
import torch

from tools.infer_offset_modspace import NULL, PROMPT, encode_texts, encode_texts_with_base, modspace_infer
from benchmark.vico_prompt_format import render_prompt
from tools.optimize_ada_offset import get_scale_schedule, parse_scales
from tools.run_infinity import add_common_arguments, encode_prompt, load_tokenizer, load_transformer, load_visual_tokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser)
    parser.add_argument("--offset-path", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--subject", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--alpha-schedule", default="0.1,0.1,0.1,0.1,0.1")
    parser.add_argument("--early-composition", choices=["interpolate", "preserve-offset"], default="interpolate",
                        help="interpolate weights both base and offset by 1-alpha; preserve-offset keeps D_s at weight 1")
    parser.add_argument("--early-cfg", type=float, default=1.2)
    parser.add_argument("--late-cfg", type=float, default=2.0)
    parser.add_argument("--late-gss", choices=["prompt", "null"], default="prompt",
                        help="shared AdaLN modulation at scales after the alpha schedule; ca_kv remains prompt in both variants")
    parser.add_argument("--offset-scales", default="0-4",
                        help="scales where the learned offset is injected; supports '0-4' or '0,2-4'")
    parser.add_argument("--prompt-mode", choices=["class", "modifier-only"], default="class",
                        help="class renders 'a cat statue in the snow'; modifier-only renders 'in the snow'")
    parser.add_argument("--class-conditioned-base", action="store_true",
                        help="use the checkpoint's training class prompt as modulation/CA base")
    parser.add_argument("--subject-as-class", action="store_true",
                        help="render prompts using the full subject name, e.g. alarm_clock -> alarm clock")
    # The benchmark runner uses one image from each independently fitted
    # offset. Additional samples are therefore opt-in, not a paper default.
    parser.add_argument("--num-samples", type=int, default=1)
    # --seed is defined by add_common_arguments(parser).
    parser.add_argument("--output-seed-start", type=int, default=0,
                        help="first normalized output seed index; lets independently trained offsets share one subject folder")
    parser.add_argument("--max-prompts", type=int, default=0,
                        help="0 means all 31 official templates; use 5 for a smoke test")
    parser.add_argument("--prepend-class-prompt", action="store_true",
                        help='generate "a <class name>" before the selected benchmark templates')
    parser.add_argument("--timing-file", default="", help="optional JSON path for pure generation timing")
    parser.add_argument("--top-k", type=int, default=900)
    parser.add_argument("--top-p", type=float, default=0.97)
    parser.set_defaults(h_div_w_template=-1.)
    args = parser.parse_args()
    if not (args.use_bit_label and args.vae_type != 0 and not args.apply_spatial_patchify):
        parser.error("Requires bitwise, non-patchified Infinity-2B settings")
    alpha = tuple(float(value) for value in args.alpha_schedule.split(",") if value.strip())
    if not alpha or any(value < 0 or value > 1 for value in alpha):
        parser.error("--alpha-schedule must be comma-separated values in [0, 1]")
    if args.class_conditioned_base and alpha != (0.1,) * 5:
        parser.error("--class-conditioned-base requires alpha 0.1 on exactly five scales")

    manifest = json.loads(Path(args.manifest).read_text())
    item = next((entry for entry in manifest["subjects"] if entry["subject"] == args.subject), None)
    if item is None:
        parser.error(f"Unknown ViCo subject: {args.subject}")
    class_name = args.subject.replace("_", " ") if args.subject_as_class else item["class_name"]
    checkpoint = torch.load(args.offset_path, map_location="cpu")
    if args.class_conditioned_base and not checkpoint.get("train_prompt"):
        parser.error("--class-conditioned-base requires an offset trained with --train_prompt")

    vae = load_visual_tokenizer(args)
    vae.requires_grad_(False)
    infinity = load_transformer(vae, args)
    infinity.requires_grad_(False)
    infinity.eval()
    tokenizer, encoder = load_tokenizer(t5_path=args.text_encoder_ckpt)
    base_prompt = checkpoint["train_prompt"] if args.class_conditioned_base else ""
    base_cond = encode_prompt(tokenizer, encoder, base_prompt) if base_prompt else None
    if args.class_conditioned_base and checkpoint["train_prompt"] != f"a {class_name}":
        parser.error("Checkpoint training prompt does not match the manifest class")
    offset = checkpoint["ada_offset"].reshape(-1, 1, 6, infinity.C).cuda()
    null_len = checkpoint.get("null_len", 32)
    h_div_w = args.h_div_w_template if args.h_div_w_template > 0 else checkpoint.get("h_div_w_template", 1.0)
    _, scale_schedule, _ = get_scale_schedule(args, h_div_w)
    early_count = len(alpha)
    if early_count >= len(scale_schedule):
        parser.error(f"alpha schedule has {early_count} scales but generation has only {len(scale_schedule)}")
    learned = set(checkpoint.get("loss_scales", range(offset.shape[0])))
    early_scales = set(range(early_count))
    offset_scales = set(parse_scales(args.offset_scales, len(scale_schedule)))
    if args.class_conditioned_base and (learned != set(range(5)) or offset_scales != set(range(5))):
        parser.error("Class-base inference requires offsets trained and injected on exactly scales 0-4")
    if not offset_scales.issubset(learned):
        parser.error(f"Offset does not contain all requested injection scales: {sorted(offset_scales - learned)}")
    if not early_scales.issubset(learned):
        parser.error(f"Offset does not contain all required learned early scales: {sorted(early_scales - learned)}")
    if offset.shape[0] != 1 and offset.shape[0] < early_count:
        parser.error(f"Offset contains {offset.shape[0]} scale slots; need at least {early_count}")

    # modspace_infer starts at G_base and adds w_D*D_s plus
    # a*(G_prompt-G_base). The two supported formulations are:
    #   interpolate:    (1-a)(G_base+D_s) + a*G_prompt
    #   preserve-offset: (1-a)G_base + D_s + a*G_prompt
    coefficients = {
        scale: ((1.0 - value) if args.early_composition == "interpolate" else 1.0, value)
        for scale, value in enumerate(alpha)
    }
    cfg_by_scale = [args.early_cfg] * early_count + [args.late_cfg] * (len(scale_schedule) - early_count)
    # The CFG partner remains the base branch. In class-conditioned mode,
    # NULL indexes the encoded class prompt instead of the model null text.
    gss_sources = ({scale: PROMPT for scale in range(early_count, len(scale_schedule))}
                   if args.late_gss == "prompt" else {})
    rows = [(NULL, PROMPT, True, 0.0), (NULL, NULL, True, 0.0)]
    out_root = Path(args.out_dir) / args.subject
    out_root.mkdir(parents=True, exist_ok=True)
    prompt_cache = {}
    templates = item["prompt_templates"][:args.max_prompts or None]
    prompts = [render_prompt(template, class_name, args.prompt_mode) for template in templates]
    if args.prepend_class_prompt:
        if args.prompt_mode != "class":
            parser.error("--prepend-class-prompt requires --prompt-mode class")
        prompts.insert(0, f"a {class_name}")
    generation_seconds = 0.0
    written = 0
    for index, prompt in enumerate(prompts):
        if prompt not in prompt_cache:
            prompt_cache[prompt] = encode_prompt(tokenizer, encoder, prompt)
        texts = (encode_texts_with_base(infinity, base_cond, prompt_cache[prompt])
                 if base_cond is not None else encode_texts(infinity, null_len, prompt_cache[prompt]))
        prompt_dir = out_root / f"prompt_{index:02d}"
        prompt_dir.mkdir(parents=True, exist_ok=True)
        (prompt_dir / "prompt.txt").write_text(prompt + "\n")
        for sample in range(args.num_samples):
            started = time.perf_counter()
            image = modspace_infer(infinity, vae, texts, offset, rows, scale_schedule,
                                   offset_scales, early_scales, args.early_cfg, 1.0,
                                   args.seed + sample, args, scale_coefficients=coefficients,
                                   cfg_by_scale=cfg_by_scale,
                                   conditional_gss_source_by_scale=gss_sources)
            generation_seconds += time.perf_counter() - started
            path = prompt_dir / f"seed_{args.output_seed_start + sample:03d}.png"
            if not cv2.imwrite(str(path), image):
                raise OSError(f"Could not write {path}")
            written += 1
            # Consumed by the benchmark parent process to maintain its global
            # image-level tqdm bar. Full paths remain in its .log file.
            print(f"[image] {path}", flush=True)
    if args.timing_file:
        timing_path = Path(args.timing_file)
        timing_path.parent.mkdir(parents=True, exist_ok=True)
        timing_path.write_text(json.dumps({
            "subject": args.subject, "late_gss": args.late_gss,
            "class_conditioned_base": args.class_conditioned_base,
            "base_prompt": base_prompt,
            "train_prompt": checkpoint.get("train_prompt", ""),
            "alpha_schedule": alpha, "early_composition": args.early_composition,
            "offset_scales": sorted(offset_scales),
            "images": written, "generation_seconds": generation_seconds,
            "generation_seconds_per_image": generation_seconds / max(written, 1),
            "prompts": len(prompts),
            "benchmark_prompts": len(templates),
            "prepended_class_prompt": args.prepend_class_prompt,
        }, indent=2) + "\n")


if __name__ == "__main__":
    main()
