"""Сериализация транскрипта в TXT / SRT / JSON.

Все функции принимают итерабельный поток TranscribedSegment'ов и возвращают str.
"""

from __future__ import annotations

import json
from typing import Any, Iterable

from app.pipeline.transcribe import TranscribedSegment


def to_txt(segments: Iterable[TranscribedSegment]) -> str:
    """Один сегмент = одна строка. Без таймкодов."""
    return "\n".join(s.text.strip() for s in segments if s.text.strip())


def _fmt_srt_time(seconds: float) -> str:
    """Секунды → HH:MM:SS,mmm (формат SRT)."""
    if seconds < 0 or seconds != seconds:  # NaN-safe
        seconds = 0.0
    total_ms = int(round(seconds * 1000))
    h, rem = divmod(total_ms, 3_600_000)
    m, rem = divmod(rem, 60_000)
    s, ms = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def to_srt(segments: Iterable[TranscribedSegment]) -> str:
    lines: list[str] = []
    for idx, seg in enumerate(segments, start=1):
        lines.append(str(idx))
        lines.append(f"{_fmt_srt_time(seg.start)} --> {_fmt_srt_time(seg.end)}")
        lines.append(seg.text.strip())
        lines.append("")
    return "\n".join(lines)


def to_json(segments: Iterable[TranscribedSegment], meta: dict[str, Any]) -> str:
    """`{"meta": {...}, "segments": [{start, end, text}, ...]}`. UTF-8, indent=2."""
    payload = {
        "meta": meta,
        "segments": [
            {"start": s.start, "end": s.end, "text": s.text}
            for s in segments
        ],
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)
