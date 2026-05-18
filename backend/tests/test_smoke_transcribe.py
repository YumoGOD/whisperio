"""Smoke-test полного pipeline: probe → wav 16k mono → VAD → faster-whisper → форматы.

Запуск:
    cd backend
    # без аргументов берёт storage/_smoke_pipeline/speech.wav (Задача 4):
    python -m tests.test_smoke_transcribe
    # либо явный путь:
    python -m tests.test_smoke_transcribe path\to\audio.mp3

Использует settings.WHISPER_MODEL (по умолчанию large-v3). На CPU прогон
9-сек файла занимает 2–3 минуты при первом запуске (плюс ~1.5 GB загрузки весов).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from app.config import settings
from app.pipeline.audio import probe_duration, to_wav_16k_mono
from app.pipeline.formats import to_json, to_srt, to_txt
from app.pipeline.transcribe import transcriber
from app.pipeline.vad import split_into_segments

SCRATCH = settings.STORAGE_DIR / "_smoke_pipeline"


def _resolve_source(argv: list[str]) -> Path:
    if len(argv) > 1:
        p = Path(argv[1]).resolve()
        if not p.exists():
            print(f"ERR: file not found: {p}", file=sys.stderr)
            sys.exit(2)
        return p
    default = SCRATCH / "speech.wav"
    if not default.exists():
        print("ERR: no argument and storage/_smoke_pipeline/speech.wav not found.\n"
              "Run task 4 smoke test first, or pass an audio path.", file=sys.stderr)
        sys.exit(2)
    return default


def main(argv: list[str]) -> int:
    src = _resolve_source(argv)
    print(f"source: {src}  ({src.stat().st_size} bytes)")

    dur = probe_duration(src)
    print(f"  probe_duration: {dur:.3f}s")

    SCRATCH.mkdir(parents=True, exist_ok=True)
    wav = SCRATCH / "converted_16k_mono.wav"
    to_wav_16k_mono(src, wav)
    print(f"  wav: {wav.name} ({wav.stat().st_size} bytes)")

    print(f"  VAD: enabled={settings.VAD_ENABLED}, "
          f"min_silence={settings.VAD_MIN_SILENCE_MS}ms, "
          f"pad={settings.VAD_SPEECH_PAD_MS}ms, "
          f"max_seg={settings.VAD_MAX_SEGMENT_SEC}s")
    segs = split_into_segments(wav)
    print(f"  VAD segments: {len(segs)}")
    for i, (s, e) in enumerate(segs):
        print(f"    [{i}] {s:7.3f} -> {e:7.3f}  ({e - s:.3f}s)")

    print(f"\n  faster-whisper: model={settings.WHISPER_MODEL}, "
          f"device={settings.WHISPER_DEVICE}, compute={settings.WHISPER_COMPUTE_TYPE}, "
          f"batch={settings.WHISPER_BATCH_SIZE}, beam={settings.BEAM_SIZE}")
    print("  loading model + running transcribe (first run pulls weights ~1.5 GB)...")
    t0 = time.perf_counter()
    result = transcriber.transcribe_segments(wav, segs, language=None)
    elapsed = time.perf_counter() - t0
    print(f"  done in {elapsed:.1f}s on {transcriber.device}/{transcriber.compute_type}")
    print(f"  detected_language: {result.detected_language!r}")
    print(f"  segments transcribed: {len(result.segments)}")

    print("\n  first 3 segments:")
    for i, s in enumerate(result.segments[:3]):
        print(f"    [{i}] {s.start:7.3f} -> {s.end:7.3f}  {s.text!r}")

    print("\n  to_txt():")
    txt = to_txt(result.segments)
    for line in txt.splitlines():
        print(f"    {line}")

    print("\n  to_srt() (first ~12 lines):")
    srt = to_srt(result.segments)
    for line in srt.splitlines()[:12]:
        print(f"    {line}")

    print("\n  to_json():")
    js = to_json(result.segments, {
        "filename": src.name,
        "model": settings.WHISPER_MODEL,
        "language": None,
        "detected_language": result.detected_language,
        "duration_sec": dur,
    })
    parsed = json.loads(js)
    print(f"    meta keys: {sorted(parsed['meta'].keys())}")
    print(f"    segments count: {len(parsed['segments'])}")
    if parsed["segments"]:
        print(f"    first segment keys: {sorted(parsed['segments'][0].keys())}")

    assert len(result.segments) > 0, "expected at least one transcribed segment"
    assert all(s.end >= s.start for s in result.segments), "non-monotonic timestamps"

    print("\nsmoke transcribe ok")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
