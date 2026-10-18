#!/usr/bin/env python3
"""Download Hugging Face assets without placing credentials in CLI history.

``--include`` accepts exact filenames or Hugging Face glob patterns.  Omit it
to download an entire repository.  Authentication is read only from HF_TOKEN.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--local-dir", required=True)
    parser.add_argument("--include", nargs="*", default=None,
                        help="Exact files or glob patterns; omit for the full repository")
    parser.add_argument("--revision", default=None)
    args = parser.parse_args()

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise SystemExit("huggingface_hub is missing. Run the environment setup script first.") from exc

    destination = Path(args.local_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    token = os.environ.get("HF_TOKEN") or None
    print(f"[download] {args.repo_id} -> {destination}")
    snapshot_download(
        repo_id=args.repo_id,
        local_dir=str(destination),
        allow_patterns=args.include or None,
        revision=args.revision,
        token=token,
    )
    print("[download] complete")


if __name__ == "__main__":
    main()
