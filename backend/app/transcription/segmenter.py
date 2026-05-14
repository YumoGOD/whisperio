from dataclasses import dataclass

from faster_whisper.audio import decode_audio
from faster_whisper.vad import VadOptions, get_speech_timestamps

from app.transcription.settings import VadSettings


@dataclass(frozen=True, slots=True)
class AudioWindow:
    start_sec: float
    end_sec: float

    @property
    def duration_sec(self) -> float:
        return max(0.0, self.end_sec - self.start_sec)


def _split_long_window(window: AudioWindow, *, max_duration_sec: float) -> list[AudioWindow]:
    if window.duration_sec <= max_duration_sec:
        return [window]
    chunks: list[AudioWindow] = []
    cursor = window.start_sec
    while cursor < window.end_sec:
        end_sec = min(window.end_sec, cursor + max_duration_sec)
        chunks.append(AudioWindow(start_sec=cursor, end_sec=end_sec))
        cursor = end_sec
    return chunks


def _merge_neighbors(
    windows: list[AudioWindow],
    *,
    max_duration_sec: float,
    merge_gap_sec: float,
) -> list[AudioWindow]:
    if not windows:
        return []
    merged: list[AudioWindow] = [windows[0]]
    for window in windows[1:]:
        prev = merged[-1]
        gap = max(0.0, window.start_sec - prev.end_sec)
        merged_duration = max(0.0, window.end_sec - prev.start_sec)
        if gap <= merge_gap_sec and merged_duration <= max_duration_sec:
            merged[-1] = AudioWindow(start_sec=prev.start_sec, end_sec=max(prev.end_sec, window.end_sec))
        else:
            merged.append(window)
    return merged


def build_vad_windows(
    audio_path: str,
    *,
    sample_rate: int,
    vad: VadSettings,
) -> tuple[list[AudioWindow], float]:
    audio = decode_audio(audio_path, sampling_rate=sample_rate)
    total_duration_sec = max(0.0, float(audio.shape[0]) / float(sample_rate))
    if total_duration_sec <= 0:
        return [AudioWindow(start_sec=0.0, end_sec=0.0)], 0.0
    options = VadOptions(
        threshold=vad.threshold,
        min_speech_duration_ms=vad.min_speech_duration_ms,
        min_silence_duration_ms=vad.min_silence_duration_ms,
        speech_pad_ms=vad.speech_pad_ms,
        max_speech_duration_s=vad.max_segment_duration_sec,
    )
    try:
        speech_timestamps = get_speech_timestamps(
            audio,
            vad_options=options,
            sampling_rate=sample_rate,
        )
    except Exception:
        # Fail-open: for difficult or broken VAD states process full audio.
        return [AudioWindow(start_sec=0.0, end_sec=total_duration_sec)], total_duration_sec
    if not speech_timestamps:
        return [AudioWindow(start_sec=0.0, end_sec=total_duration_sec)], total_duration_sec

    windows: list[AudioWindow] = []
    for segment in speech_timestamps:
        start_sec = max(0.0, float(segment["start"]) / float(sample_rate))
        end_sec = max(start_sec, float(segment["end"]) / float(sample_rate))
        if end_sec <= start_sec:
            continue
        windows.extend(
            _split_long_window(
                AudioWindow(start_sec=start_sec, end_sec=end_sec),
                max_duration_sec=vad.max_segment_duration_sec,
            )
        )
    if not windows:
        return [AudioWindow(start_sec=0.0, end_sec=total_duration_sec)], total_duration_sec
    windows.sort(key=lambda item: (item.start_sec, item.end_sec))
    windows = _merge_neighbors(
        windows,
        max_duration_sec=vad.max_segment_duration_sec,
        merge_gap_sec=vad.merge_gap_sec,
    )
    return windows, total_duration_sec

