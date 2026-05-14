import asyncio
from dataclasses import dataclass
from typing import Awaitable, Callable

from faster_whisper import WhisperModel

from app.transcription.inference import InferenceResult
from app.transcription.quality import (
    choose_best_window_segments,
    decorate_segments,
    find_rescue_windows,
    replace_rows_in_window,
)
from app.transcription.segmenter import AudioWindow, build_vad_windows
from app.transcription.settings import DecodeSettings, TranscriptionSettings

TranscribeFn = Callable[
    [WhisperModel, str, list[AudioWindow], DecodeSettings],
    Awaitable[InferenceResult],
]


@dataclass(frozen=True, slots=True)
class PipelineResult:
    segment_rows: list[dict[str, object]]
    quality_payload: dict[str, object]
    duration_sec: float
    vad_window_count: int
    rescue_window_count: int


def _rows_in_window(
    rows: list[dict[str, float | str]],
    *,
    window: AudioWindow,
) -> list[dict[str, float | str]]:
    selected: list[dict[str, float | str]] = []
    for row in rows:
        start_sec = float(row.get("start_sec", 0.0))
        end_sec = float(row.get("end_sec", start_sec))
        overlaps = start_sec < window.end_sec and end_sec > window.start_sec
        if overlaps:
            selected.append(row)
    return selected


class QualityFirstPipeline:
    def __init__(self, *, settings: TranscriptionSettings, transcribe_fn: TranscribeFn) -> None:
        self.settings = settings
        self._transcribe_fn = transcribe_fn

    async def run(
        self,
        *,
        model: WhisperModel,
        prepared_audio_path: str,
        fallback_duration_sec: float,
    ) -> PipelineResult:
        windows, vad_duration_sec = await asyncio.to_thread(
            build_vad_windows,
            prepared_audio_path,
            sample_rate=self.settings.preprocess.sample_rate,
            vad=self.settings.profile.vad,
        )
        primary = await self._transcribe_fn(
            model,
            prepared_audio_path,
            windows,
            self.settings.profile.primary_decode,
        )
        segment_rows = primary.segments
        rescue_windows, rescue_reasons = find_rescue_windows(
            segment_rows,
            quality=self.settings.profile.quality,
        )
        rescue_evaluations: list[dict[str, object]] = []
        rescue_applied = 0
        for idx, window in enumerate(rescue_windows):
            original_rows = _rows_in_window(segment_rows, window=window)
            rescue_result = await self._transcribe_fn(
                model,
                prepared_audio_path,
                [window],
                self.settings.profile.rescue_decode,
            )
            chosen_rows, used_rescue, score_info = choose_best_window_segments(
                original_rows=original_rows,
                rescue_rows=rescue_result.segments,
                min_gain=self.settings.profile.quality.rescue_min_score_gain,
            )
            rescue_evaluations.append(
                {
                    "window_index": idx,
                    "window_start_sec": round(window.start_sec, 3),
                    "window_end_sec": round(window.end_sec, 3),
                    "selected_rescue": used_rescue,
                    "original_segments": len(original_rows),
                    "rescue_segments": len(rescue_result.segments),
                    **score_info,
                }
            )
            if not used_rescue:
                continue
            rescue_applied += 1
            segment_rows = replace_rows_in_window(
                all_rows=segment_rows,
                window=window,
                replacement_rows=chosen_rows,
            )

        decorated_rows, tagging_stats = decorate_segments(
            segment_rows,
            quality=self.settings.profile.quality,
        )
        decorated_duration = 0.0
        if decorated_rows:
            decorated_duration = max(float(item["end_sec"]) for item in decorated_rows)
        duration_sec = max(fallback_duration_sec, vad_duration_sec, primary.duration_sec, decorated_duration)
        quality_payload = {
            "profile": self.settings.profile.name,
            "vad_window_count": len(windows),
            "rescue_window_count": len(rescue_windows),
            "rescue_applied_count": rescue_applied,
            "rescue_reason_counts": rescue_reasons,
            "rescue_windows": rescue_evaluations,
            "tagging": tagging_stats,
        }
        return PipelineResult(
            segment_rows=decorated_rows,
            quality_payload=quality_payload,
            duration_sec=duration_sec,
            vad_window_count=len(windows),
            rescue_window_count=len(rescue_windows),
        )

