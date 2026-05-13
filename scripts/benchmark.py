from __future__ import annotations

import argparse
import shutil
import sys
import uuid
from pathlib import Path
from time import perf_counter

from app.config import get_settings
from app.logging_config import configure_logging
from app.transcription.audio import probe_duration_seconds
from app.transcription.pipeline import TranscriptionPipeline


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark faster-whisper transcription on one local file.")
    parser.add_argument("audio_file", type=Path, help="Path to an audio/video file readable by ffmpeg.")
    parser.add_argument("--profile", default=None, help="Transcription profile: accuracy_first or speed_balanced.")
    args = parser.parse_args()

    settings = get_settings()
    configure_logging(settings)

    if not args.audio_file.exists():
        print(f"File not found: {args.audio_file}", file=sys.stderr)
        return 2

    job_id = f"benchmark_{uuid.uuid4().hex[:12]}"
    upload_copy = settings.upload_dir / f"{job_id}_{args.audio_file.name}"
    shutil.copy2(args.audio_file, upload_copy)

    params = {
        "profile": args.profile or settings.default_profile,
        "chunk_seconds": settings.chunk_seconds,
        "chunk_overlap_seconds": settings.chunk_overlap_seconds,
        "model": settings.whisper_model,
        "compute_type": settings.whisper_compute_type,
        "cpu_threads": settings.whisper_cpu_threads,
    }
    duration = probe_duration_seconds(upload_copy)
    pipeline = TranscriptionPipeline(settings)

    started = perf_counter()

    def progress(value: float, stage: str) -> None:
        print(f"{value * 100:5.1f}% {stage}", flush=True)

    result = pipeline.run(
        job_id=job_id,
        input_path=upload_copy,
        original_filename=args.audio_file.name,
        params=params,
        progress_callback=progress,
    )
    elapsed = perf_counter() - started
    rtf = elapsed / duration if duration else 0

    print()
    print(f"Audio duration: {duration:.2f} seconds")
    print(f"Elapsed time:    {elapsed:.2f} seconds")
    print(f"Real-time factor: {rtf:.3f}x")
    print(f"Transcript dir:  {result['transcript_dir']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
