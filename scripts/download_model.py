from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from huggingface_hub import snapshot_download


MODEL_REPOS = {
    "tiny": "Systran/faster-whisper-tiny",
    "base": "Systran/faster-whisper-base",
    "small": "Systran/faster-whisper-small",
    "medium": "Systran/faster-whisper-medium",
    "large-v2": "Systran/faster-whisper-large-v2",
    "large-v3": "Systran/faster-whisper-large-v3",
}


def resolve_repo(model: str) -> str:
    return MODEL_REPOS.get(model, model)


def copy_snapshot(snapshot_path: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for source in snapshot_path.iterdir():
        target = output_dir / source.name
        if source.is_dir():
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(source, target)
        else:
            shutil.copy2(source, target)


def main() -> int:
    parser = argparse.ArgumentParser(description="Download a faster-whisper CTranslate2 model for offline use.")
    parser.add_argument("model", nargs="?", default="large-v3", help="Model alias or Hugging Face repo id.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/app/data/models/faster-whisper-large-v3"),
        help="Directory where the model files should be copied.",
    )
    parser.add_argument("--revision", default=None, help="Optional Hugging Face revision.")
    args = parser.parse_args()

    repo_id = resolve_repo(args.model)
    print(f"Downloading {repo_id} ...", flush=True)
    try:
        snapshot = Path(snapshot_download(repo_id=repo_id, revision=args.revision))
    except Exception as exc:
        print(f"Model download failed: {exc}", file=sys.stderr)
        return 1

    copy_snapshot(snapshot, args.output_dir)
    print(f"Model is ready at: {args.output_dir}")
    print(f"Set WHISPER_MODEL={args.output_dir} in .env to use it without downloading at runtime.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
