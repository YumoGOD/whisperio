"""ffmpeg / ffprobe обёртки.

Тонкая прослойка над `ffmpeg-python` с понятными исключениями.
Любая ошибка ffmpeg/ffprobe оборачивается в AudioError с человекочитаемым
последним хвостом stderr — чтобы worker мог положить её в Job.error.
"""

from __future__ import annotations

import logging
from pathlib import Path

import ffmpeg  # ffmpeg-python

log = logging.getLogger(__name__)


class AudioError(RuntimeError):
    """Ошибка препроцессинга аудио (ffmpeg/ffprobe)."""


def _ff_error_text(err: ffmpeg.Error) -> str:
    raw = err.stderr or b""
    msg = raw.decode("utf-8", errors="ignore").strip()
    return msg[-500:] if msg else str(err)


def probe_duration(path: Path | str) -> float:
    """Длительность файла в секундах (через ffprobe)."""
    p = Path(path)
    if not p.exists():
        raise AudioError(f"file does not exist: {p}")
    try:
        info = ffmpeg.probe(str(p))
    except ffmpeg.Error as e:
        raise AudioError(f"ffprobe failed for {p.name}: {_ff_error_text(e)}") from e
    except FileNotFoundError as e:
        raise AudioError("ffprobe not found in PATH (install ffmpeg)") from e

    try:
        return float(info["format"]["duration"])
    except (KeyError, TypeError, ValueError) as e:
        raise AudioError(f"missing duration in probe output for {p.name}") from e


def to_wav_16k_mono(input_path: Path | str, output_path: Path | str) -> Path:
    """Конвертировать произвольный аудио/видео в WAV 16 kHz mono PCM 16-bit.

    Это стандартный вход для Silero VAD и faster-whisper. Видеодорожка
    отбрасывается, аудио ремаплируется в 1 канал @ 16 kHz.
    """
    inp = Path(input_path)
    out = Path(output_path)
    if not inp.exists():
        raise AudioError(f"input does not exist: {inp}")
    out.parent.mkdir(parents=True, exist_ok=True)

    try:
        (
            ffmpeg
            .input(str(inp))
            .output(
                str(out),
                ac=1,                # mono
                ar=16000,            # 16 kHz
                acodec="pcm_s16le",  # 16-bit little-endian PCM
                vn=None,             # drop video stream
            )
            .overwrite_output()
            .run(capture_stdout=True, capture_stderr=True, quiet=True)
        )
    except ffmpeg.Error as e:
        raise AudioError(f"ffmpeg conversion failed for {inp.name}: {_ff_error_text(e)}") from e
    except FileNotFoundError as e:
        raise AudioError("ffmpeg not found in PATH") from e

    if not out.exists() or out.stat().st_size == 0:
        raise AudioError(f"ffmpeg succeeded but produced no output: {out}")
    return out
