"""Smoke-test пайплайна препроцессинга: probe → wav 16k mono → Silero VAD.

Запуск:
    cd backend
    # 1) с реальным файлом:
    python -m tests.test_smoke_pipeline path/to/audio.mp3
    # 2) без файла — будет сгенерирован синтетический MP3 (шум + тишина + шум):
    python -m tests.test_smoke_pipeline

Без воркера, без faster-whisper — только препроцессинг.
"""

from __future__ import annotations

import sys
from pathlib import Path

import ffmpeg

from app.config import settings
from app.pipeline.audio import probe_duration, to_wav_16k_mono
from app.pipeline.vad import split_into_segments

SCRATCH = settings.STORAGE_DIR / "_smoke_pipeline"


def _generate_synthetic_mp3() -> Path:
    """Соберём короткий MP3: 2с шума + 1с тишины + 3с шума.

    Цель — проверить, что pipeline работает. VAD на шуме может и не найти речь,
    это нормально: smoke-test проверяет отсутствие исключений, а не качество.
    """
    SCRATCH.mkdir(parents=True, exist_ok=True)
    out = SCRATCH / "synth.mp3"
    if out.exists():
        return out

    noise1 = ffmpeg.input("anoisesrc=duration=2:color=pink:amplitude=0.4", f="lavfi")
    silence = ffmpeg.input("anullsrc=channel_layout=mono:sample_rate=16000:duration=1", f="lavfi")
    noise2 = ffmpeg.input("anoisesrc=duration=3:color=pink:amplitude=0.4", f="lavfi")
    (
        ffmpeg
        .concat(noise1, silence, noise2, v=0, a=1)
        .output(str(out), ac=1, ar=16000)
        .overwrite_output()
        .run(capture_stdout=True, capture_stderr=True, quiet=True)
    )
    return out


def _resolve_source(argv: list[str]) -> Path:
    if len(argv) > 1:
        p = Path(argv[1]).resolve()
        if not p.exists():
            print(f"  ERR: file not found: {p}", file=sys.stderr)
            sys.exit(2)
        return p
    print("  no file argument given; generating synthetic mp3 (noise+silence+noise)")
    return _generate_synthetic_mp3()


def main(argv: list[str]) -> int:
    src = _resolve_source(argv)
    print(f"source: {src}  ({src.stat().st_size} bytes)")

    dur = probe_duration(src)
    print(f"  probe_duration: {dur:.3f}s")

    SCRATCH.mkdir(parents=True, exist_ok=True)
    wav = SCRATCH / "converted_16k_mono.wav"
    to_wav_16k_mono(src, wav)
    wav_dur = probe_duration(wav)
    print(f"  wav: {wav.name} ({wav.stat().st_size} bytes, {wav_dur:.3f}s)")
    assert abs(wav_dur - dur) < 0.5, f"duration drift: {dur:.3f} vs {wav_dur:.3f}"

    print(f"  VAD_ENABLED={settings.VAD_ENABLED}, "
          f"min_silence={settings.VAD_MIN_SILENCE_MS}ms, "
          f"pad={settings.VAD_SPEECH_PAD_MS}ms, "
          f"max_seg={settings.VAD_MAX_SEGMENT_SEC}s")
    segs = split_into_segments(wav)
    print(f"  segments ({len(segs)}):")
    for i, (s, e) in enumerate(segs[:20]):
        print(f"    [{i:02d}] {s:7.3f} -> {e:7.3f}  ({e - s:.3f}s)")
    if len(segs) > 20:
        print(f"    ... and {len(segs) - 20} more")

    # Каждый сегмент не длиннее VAD_MAX_SEGMENT_SEC (при включённом VAD).
    if settings.VAD_ENABLED:
        too_long = [(s, e) for s, e in segs if (e - s) > settings.VAD_MAX_SEGMENT_SEC + 1.0]
        assert not too_long, f"segments exceed max_segment_sec: {too_long}"

    print("\nsmoke pipeline ok")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
