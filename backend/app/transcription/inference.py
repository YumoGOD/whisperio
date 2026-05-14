import inspect
from dataclasses import dataclass
from typing import Any

from faster_whisper import WhisperModel

from app.transcription.segmenter import AudioWindow
from app.transcription.settings import DecodeSettings


@dataclass(frozen=True, slots=True)
class InferenceResult:
    segments: list[dict[str, float | str]]
    duration_sec: float


def _windows_to_clip_timestamps(windows: list[AudioWindow]) -> list[float]:
    clip_timestamps: list[float] = []
    for window in windows:
        clip_timestamps.extend((round(window.start_sec, 3), round(window.end_sec, 3)))
    return clip_timestamps


def _filter_supported_kwargs(model: WhisperModel, kwargs: dict[str, Any]) -> dict[str, Any]:
    supported = set(inspect.signature(model.transcribe).parameters)
    return {key: value for key, value in kwargs.items() if key in supported}


def transcribe_windows(
    model: WhisperModel,
    *,
    audio_path: str,
    windows: list[AudioWindow],
    decode: DecodeSettings,
    language: str,
    task: str,
) -> InferenceResult:
    if not windows:
        return InferenceResult(segments=[], duration_sec=0.0)
    clip_timestamps = _windows_to_clip_timestamps(windows)
    transcribe_kwargs: dict[str, Any] = {
        "language": language if language != "auto" else None,
        "task": task,
        "beam_size": decode.beam_size,
        "best_of": decode.best_of,
        "temperature": decode.temperature,
        "condition_on_previous_text": decode.condition_on_previous_text,
        "no_speech_threshold": decode.no_speech_threshold,
        "log_prob_threshold": decode.log_prob_threshold,
        "compression_ratio_threshold": decode.compression_ratio_threshold,
        "without_timestamps": False,
        "vad_filter": False,
        "clip_timestamps": clip_timestamps,
    }
    transcribe_kwargs = _filter_supported_kwargs(model, transcribe_kwargs)
    segments, info = model.transcribe(audio_path, **transcribe_kwargs)

    rows: list[dict[str, float | str]] = []
    for segment in segments:
        rows.append(
            {
                "start_sec": float(segment.start),
                "end_sec": float(segment.end),
                "text": segment.text.strip(),
                "avg_logprob": float(getattr(segment, "avg_logprob", 0.0)),
                "no_speech_prob": float(getattr(segment, "no_speech_prob", 0.0)),
                "compression_ratio": float(getattr(segment, "compression_ratio", 0.0)),
            }
        )
    inferred_duration = max(window.end_sec for window in windows)
    duration_sec = float(getattr(info, "duration", inferred_duration) or inferred_duration)
    return InferenceResult(segments=rows, duration_sec=duration_sec)

