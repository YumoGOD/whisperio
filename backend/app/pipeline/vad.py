"""Silero VAD: разбиение аудио на речевые сегменты.

Модель грузится один раз (lazy singleton). Параметры берутся из settings.
Если VAD_ENABLED=false — возвращаем один сегмент во весь файл.
При любом сбое VAD откатываемся к одному сегменту во весь файл и пишем warning,
чтобы не валить весь пайплайн транскрипции из-за VAD.
"""

from __future__ import annotations

import logging
from pathlib import Path
from threading import Lock
from typing import Any

from app.config import settings

log = logging.getLogger(__name__)

SAMPLE_RATE = 16000

_lock = Lock()
_model: Any = None
_read_audio: Any = None
_get_speech_timestamps: Any = None


def _load() -> tuple[Any, Any, Any]:
    """Лениво грузит модель Silero VAD. Идемпотентно, thread-safe."""
    global _model, _read_audio, _get_speech_timestamps
    if _model is not None:
        return _model, _read_audio, _get_speech_timestamps
    with _lock:
        if _model is None:
            try:
                from silero_vad import (
                    get_speech_timestamps, load_silero_vad, read_audio,
                )
            except ImportError as e:
                raise RuntimeError(
                    "silero-vad is not installed; run pip install -r requirements.txt"
                ) from e
            log.info("loading Silero VAD model ...")
            _model = load_silero_vad()
            _read_audio = read_audio
            _get_speech_timestamps = get_speech_timestamps
            log.info("Silero VAD loaded")
    return _model, _read_audio, _get_speech_timestamps


def _whole_file_segment(wav_path: Path) -> list[tuple[float, float]]:
    from app.pipeline.audio import probe_duration
    return [(0.0, probe_duration(wav_path))]


def split_into_segments(wav_path: Path | str) -> list[tuple[float, float]]:
    """Разбить wav на (start_sec, end_sec) речевые сегменты.

    Ожидается, что файл уже 16 kHz mono PCM 16-bit (см. audio.to_wav_16k_mono).
    Длительность каждого сегмента ограничена settings.VAD_MAX_SEGMENT_SEC.
    """
    p = Path(wav_path)
    if not p.exists():
        raise FileNotFoundError(p)

    if not settings.VAD_ENABLED:
        return _whole_file_segment(p)

    try:
        model, read_audio, get_speech_timestamps = _load()
        wav = read_audio(str(p), sampling_rate=SAMPLE_RATE)
        ts = get_speech_timestamps(
            wav,
            model,
            sampling_rate=SAMPLE_RATE,
            min_silence_duration_ms=settings.VAD_MIN_SILENCE_MS,
            speech_pad_ms=settings.VAD_SPEECH_PAD_MS,
            max_speech_duration_s=float(settings.VAD_MAX_SEGMENT_SEC),
            return_seconds=True,
        )
    except Exception:
        log.exception("VAD failed; falling back to single segment for %s", p.name)
        return _whole_file_segment(p)

    segments = [(float(t["start"]), float(t["end"])) for t in ts]
    if not segments:
        log.info("VAD returned 0 speech segments for %s", p.name)
    return segments
