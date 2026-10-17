#!/usr/bin/env python3
"""Arrange released ARBooth checkpoint files in the layout its evaluator expects."""
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="benchmark/manifests/vico.json")
    parser.add_argument("--flat-dir", default="weights/arbooth/released/ckpts")
    parser.add_argument("--output", default="weights/arbooth/by_subject")
    parser.add_argument("--copy", action="store_true", help="copy instead of creating relative symlinks")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    subjects = json.loads((root / args.manifest).read_text())["subjects"]
    flat_dir = (root / args.flat_dir).resolve()
    output = (root / args.output).resolve()
    if not flat_dir.is_dir():
        raise SystemExit(f"ARBooth checkpoint directory is missing: {flat_dir}")

    for item in subjects:
        subject = item["subject"]
        source = flat_dir / f"{subject}.pth"
        if not source.is_file():
            raise SystemExit(f"Missing released ARBooth checkpoint: {source}")
        destination = output / subject / source.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.is_symlink() or destination.exists():
            if destination.resolve() == source.resolve():
                continue
            destination.unlink()
        if args.copy:
            shutil.copy2(source, destination)
        else:
            destination.symlink_to(os.path.relpath(source, destination.parent))
        print(f"[ready] {subject}: {destination}")
    print(f"[ready] {len(subjects)} ARBooth checkpoints under {output}")


if __name__ == "__main__":
    main()
