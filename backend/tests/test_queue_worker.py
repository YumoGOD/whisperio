import asyncio
import threading
import time

import numpy as np

from app.transcription import segmenter
from app.transcription.inference import InferenceResult
from app.transcription.pipeline import QualityFirstPipeline
from app.transcription.quality import (
    choose_best_window_segments,
    find_rescue_windows,
)
from app.transcription.segmenter import AudioWindow, build_vad_windows
from app.transcription.settings import load_transcription_settings
from app.transcription_worker import TranscriptionWorker


def test_settings_load_profile_and_language(monkeypatch):
    monkeypatch.setenv("TRANSCRIBE_PROFILE", "balanced")
    monkeypatch.setenv("TRANSCRIBE_LANGUAGE", "auto")

    settings = load_transcription_settings()

    assert settings.profile.name == "balanced"
    assert settings.language == "auto"
    assert settings.profile.primary_decode.condition_on_previous_text is False


def test_build_vad_windows_merges_short_gaps(monkeypatch):
    sample_rate = 16000
    monkeypatch.setattr(
        segmenter,
        "decode_audio",
        lambda *_args, **_kwargs: np.zeros(sample_rate * 45, dtype=np.float32),
    )
    monkeypatch.setattr(
        segmenter,
        "get_speech_timestamps",
        lambda *_args, **_kwargs: [
            {"start": 0, "end": sample_rate * 10},
            {"start": sample_rate * 10 + 1000, "end": sample_rate * 20},
            {"start": sample_rate * 35, "end": sample_rate * 40},
        ],
    )
    settings = load_transcription_settings()

    windows, duration_sec = build_vad_windows("dummy.wav", sample_rate=sample_rate, vad=settings.profile.vad)

    assert round(duration_sec, 1) == 45.0
    assert len(windows) == 2
    assert windows[0].start_sec == 0.0
    assert round(windows[0].end_sec, 1) == 20.0


def test_build_vad_windows_falls_back_to_full_audio(monkeypatch):
    sample_rate = 16000
    monkeypatch.setattr(
        segmenter,
        "decode_audio",
        lambda *_args, **_kwargs: np.zeros(sample_rate * 30, dtype=np.float32),
    )

    def _raise(*_args, **_kwargs):
        raise RuntimeError("vad failed")

    monkeypatch.setattr(segmenter, "get_speech_timestamps", _raise)
    settings = load_transcription_settings()

    windows, duration_sec = build_vad_windows("dummy.wav", sample_rate=sample_rate, vad=settings.profile.vad)

    assert len(windows) == 1
    assert windows[0] == AudioWindow(start_sec=0.0, end_sec=duration_sec)
    assert round(duration_sec, 1) == 30.0


def test_quality_helpers_choose_rescue_segments():
    quality = load_transcription_settings().profile.quality
    raw_rows = [
        {
            "start_sec": 2.0,
            "end_sec": 4.0,
            "text": "неразборчиво",
            "avg_logprob": -4.0,
            "no_speech_prob": 0.99,
            "compression_ratio": 1.2,
        }
    ]
    rescue_rows = [
        {
            "start_sec": 2.0,
            "end_sec": 4.0,
            "text": "улучшенный фрагмент речи",
            "avg_logprob": -0.2,
            "no_speech_prob": 0.1,
            "compression_ratio": 1.1,
        }
    ]

    windows, reasons = find_rescue_windows(raw_rows, quality=quality)
    chosen, selected_rescue, scores = choose_best_window_segments(
        original_rows=raw_rows,
        rescue_rows=rescue_rows,
        min_gain=quality.rescue_min_score_gain,
    )

    assert windows
    assert "LOW_CONFIDENCE" in reasons
    assert selected_rescue is True
    assert chosen == rescue_rows
    assert scores["rescue_score"] > scores["original_score"]


def test_pipeline_applies_segment_rescue(monkeypatch):
    settings = load_transcription_settings()
    monkeypatch.setattr(
        "app.transcription.pipeline.build_vad_windows",
        lambda *_args, **_kwargs: ([AudioWindow(0.0, 12.0)], 12.0),
    )

    async def fake_transcribe(_model, _path, windows, decode):
        if decode == settings.profile.primary_decode:
            return InferenceResult(
                segments=[
                    {
                        "start_sec": windows[0].start_sec,
                        "end_sec": windows[0].end_sec,
                        "text": "очень шумно",
                        "avg_logprob": -4.3,
                        "no_speech_prob": 0.99,
                        "compression_ratio": 2.5,
                    }
                ],
                duration_sec=12.0,
            )
        return InferenceResult(
            segments=[
                {
                    "start_sec": windows[0].start_sec,
                    "end_sec": windows[0].end_sec,
                    "text": "улучшенная расшифровка",
                    "avg_logprob": -0.4,
                    "no_speech_prob": 0.08,
                    "compression_ratio": 1.2,
                }
            ],
            duration_sec=12.0,
        )

    pipeline = QualityFirstPipeline(settings=settings, transcribe_fn=fake_transcribe)
    result = asyncio.run(
        pipeline.run(
            model=object(),
            prepared_audio_path="unused.wav",
            fallback_duration_sec=12.0,
        )
    )

    assert result.quality_payload["rescue_window_count"] == 1
    assert result.quality_payload["rescue_applied_count"] == 1
    assert result.segment_rows[0]["text"] == "улучшенная расшифровка"


def test_transcribe_calls_are_serialized(monkeypatch):
    worker = TranscriptionWorker()
    state_lock = threading.Lock()
    state = {"active": 0, "max_active": 0}

    def fake_transcribe_windows(*_args, **_kwargs):
        with state_lock:
            state["active"] += 1
            state["max_active"] = max(state["max_active"], state["active"])
        time.sleep(0.05)
        with state_lock:
            state["active"] -= 1
        return InferenceResult(segments=[], duration_sec=0.0)

    monkeypatch.setattr("app.transcription_worker.transcribe_windows", fake_transcribe_windows)
    decode = worker.settings.profile.primary_decode
    windows = [AudioWindow(start_sec=0.0, end_sec=5.0)]

    async def run_parallel_calls():
        await asyncio.gather(
            worker._run_transcribe(object(), "a.wav", windows, decode),
            worker._run_transcribe(object(), "b.wav", windows, decode),
        )

    asyncio.run(run_parallel_calls())
    assert state["max_active"] == 1

"""Legacy pre-refactor tests intentionally disabled.

import asyncio
import threading
import time
from types import SimpleNamespace

import pytest

from app.queue_worker import QualityGuardError, TranscriptionWorker


def test_invalid_env_values_use_safe_defaults(monkeypatch):
    monkeypatch.setenv("WHISPER_TASK", "invalid-task")
    monkeypatch.setenv("WHISPER_LANGUAGE", " ")
    monkeypatch.setenv("WHISPER_BEAM_SIZE", "not-an-int")
    monkeypatch.setenv("WHISPER_BEST_OF", "-3")
    monkeypatch.setenv("WHISPER_TEMPERATURE", "2.4")
    monkeypatch.setenv("WHISPER_VAD_THRESHOLD", "-1")
    monkeypatch.setenv("WHISPER_VAD_SPEECH_PAD_MS", "-100")
    monkeypatch.setenv("WHISPER_COMPRESSION_RATIO_THRESHOLD", "-2")

    worker = TranscriptionWorker()

    assert worker.task == "transcribe"
    assert worker.language == "ru"
    assert worker.beam_size == 1
    assert worker.best_of == 1
    assert worker.temperature == 1.0
    assert worker.vad_threshold == 0.0
    assert worker.vad_speech_pad_ms == 300
    assert worker.compression_ratio_threshold == 2.4


def test_transcribe_kwargs_include_chunk_length_only_when_supported():
    worker = TranscriptionWorker()

    class ModelWithChunkLength:
        def __init__(self):
            self.kwargs = None

        def transcribe(self, audio_path, chunk_length=None, **kwargs):
            self.kwargs = dict(kwargs)
            self.kwargs["chunk_length"] = chunk_length
            return [], SimpleNamespace(duration=0.0)

    class ModelWithoutChunkLength:
        def __init__(self):
            self.kwargs = None

        def transcribe(self, audio_path, **kwargs):
            self.kwargs = dict(kwargs)
            return [], SimpleNamespace(duration=0.0)

    with_chunk = ModelWithChunkLength()
    worker._transcribe_audio(with_chunk, "dummy.wav", None, True)
    assert with_chunk.kwargs["chunk_length"] == worker.chunk_length_s

    without_chunk = ModelWithoutChunkLength()
    worker._transcribe_audio(without_chunk, "dummy.wav", None, True)
    assert "chunk_length" not in without_chunk.kwargs


def test_lecture_enhance_profile_uses_dedicated_filters(monkeypatch):
    monkeypatch.setenv("WHISPER_ENHANCE_PROFILE", "lecture")
    worker = TranscriptionWorker()

    filters = worker._build_enhance_filters()

    assert any(item.startswith("highpass") for item in filters)
    assert any(item.startswith("afftdn") for item in filters)
    assert any(item.startswith("dynaudnorm") for item in filters)


def test_run_transcribe_is_serialized_across_parallel_calls():
    worker = TranscriptionWorker()
    state_lock = threading.Lock()
    state = {"active": 0, "max_active": 0}

    def fake_transcribe(model, audio_path, initial_prompt, condition_on_previous_text):
        with state_lock:
            state["active"] += 1
            state["max_active"] = max(state["max_active"], state["active"])
        time.sleep(0.05)
        with state_lock:
            state["active"] -= 1
        return [], SimpleNamespace(duration=0.0)

    worker._transcribe_audio = fake_transcribe

    async def run_parallel_calls():
        await asyncio.gather(
            worker._run_transcribe(None, "audio-1.wav", None, True),
            worker._run_transcribe(None, "audio-2.wav", None, True),
        )

    asyncio.run(run_parallel_calls())
    assert state["max_active"] == 1


def test_quality_analysis_can_trigger_and_clear_anomaly():
    worker = TranscriptionWorker()
    repeated_segments = [
        {
            "start_sec": 0.0,
            "end_sec": 1.0,
            "text": "повтор",
            "avg_logprob": -0.2,
            "no_speech_prob": 0.0,
            "compression_ratio": 1.0,
        }
        for _ in range(5)
    ]
    anomaly = worker._analyze_quality(repeated_segments, "повтор")
    assert anomaly["is_anomaly"] is True
    assert "LOW_UNIQUE_SEGMENT_RATIO" in anomaly["reasons"]
    assert anomaly["near_duplicate_ratio"] >= 0.0

    diverse_segments = [
        {
            "start_sec": float(idx),
            "end_sec": float(idx) + 1.0,
            "text": f"фраза {idx}",
            "avg_logprob": -0.2,
            "no_speech_prob": 0.0,
            "compression_ratio": 1.0,
        }
        for idx in range(5)
    ]
    clean = worker._analyze_quality(diverse_segments, None)
    assert clean["is_anomaly"] is False


def test_quality_analysis_detects_near_duplicate_runs(monkeypatch):
    monkeypatch.setenv("WHISPER_NEAR_DUPLICATE_SIMILARITY_THRESHOLD", "0.7")
    monkeypatch.setenv("WHISPER_MAX_NEAR_DUPLICATE_RATIO", "0.3")
    monkeypatch.setenv("WHISPER_MAX_NEAR_DUPLICATE_RUN", "2")
    worker = TranscriptionWorker()
    near_duplicate_segments = [
        {
            "start_sec": float(idx),
            "end_sec": float(idx) + 1.0,
            "text": text,
            "avg_logprob": -0.4,
            "no_speech_prob": 0.05,
            "compression_ratio": 1.1,
        }
        for idx, text in enumerate(
            [
                "Новый продукт доступен клиентам",
                "Новый продукт доступен клиентам сегодня",
                "Новый продукт доступен клиентам уже сейчас",
                "Новый продукт доступен клиентам",
            ]
        )
    ]

    anomaly = worker._analyze_quality(near_duplicate_segments, None)

    assert anomaly["is_anomaly"] is True
    assert "HIGH_NEAR_DUPLICATE_RATIO" in anomaly["reasons"]
    assert "LONG_NEAR_DUPLICATE_RUN" in anomaly["reasons"]


def test_quality_guard_anomaly_is_soft_by_default():
    worker = TranscriptionWorker()
    quality_payload: dict[str, object] = {}

    worker._handle_quality_guard_anomaly(
        source="fallback",
        reasons=["LONG_NEAR_DUPLICATE_RUN"],
        quality_payload=quality_payload,
    )

    assert quality_payload["soft_fail_applied"] is True
    assert quality_payload["soft_fail_source"] == "fallback"
    assert quality_payload["soft_fail_reasons"] == ["LONG_NEAR_DUPLICATE_RUN"]


def test_quality_guard_anomaly_raises_when_strict(monkeypatch):
    monkeypatch.setenv("WHISPER_STRICT_QUALITY_GUARD", "true")
    worker = TranscriptionWorker()

    with pytest.raises(QualityGuardError, match="quality guard fallback failed"):
        worker._handle_quality_guard_anomaly(
            source="fallback",
            reasons=["LONG_NEAR_DUPLICATE_RUN"],
            quality_payload={},
        )


def test_decorate_segments_returns_speech_only_segments():
    worker = TranscriptionWorker()
    raw_segments = [
        {
            "start_sec": 0.0,
            "end_sec": 1.2,
            "text": "первая фраза",
            "avg_logprob": -2.0,
            "no_speech_prob": 0.9,
            "compression_ratio": 3.0,
        },
        {
            "start_sec": 1.2,
            "end_sec": 2.0,
            "text": "  ",
            "avg_logprob": -2.0,
            "no_speech_prob": 0.9,
            "compression_ratio": 3.0,
        },
        {
            "start_sec": 2.0,
            "end_sec": 3.1,
            "text": "вторая фраза",
            "avg_logprob": -2.0,
            "no_speech_prob": 0.9,
            "compression_ratio": 3.0,
        },
    ]

    decorated_rows, tagging_stats = worker._decorate_segments(raw_segments, "unused.wav", 3.1)

    assert decorated_rows == [
        {
            "start_sec": 0.0,
            "end_sec": 1.2,
            "label": "speech",
            "text": "первая фраза",
            "confidence": 0.1353,
            "quality_flags": {
                "avg_logprob": -2.0,
                "no_speech_prob": 0.9,
                "compression_ratio": 3.0,
            },
        },
        {
            "start_sec": 2.0,
            "end_sec": 3.1,
            "label": "speech",
            "text": "вторая фраза",
            "confidence": 0.1353,
            "quality_flags": {
                "avg_logprob": -2.0,
                "no_speech_prob": 0.9,
                "compression_ratio": 3.0,
            },
        },
    ]
    assert tagging_stats["inserted_labels"] == {
        "silence": 0,
        "music": 0,
        "unclear": 0,
        "speech": 2,
    }
    assert tagging_stats["gap_metrics"] == []
    assert tagging_stats["filtered_segments"] == 0
    assert tagging_stats["filtered_reason_counts"] == {}


def test_decorate_segments_never_inserts_marker_texts(monkeypatch):
    monkeypatch.setenv("WHISPER_DROP_FILTERED_SEGMENTS", "false")
    worker = TranscriptionWorker()
    raw_segments = [
        {
            "start_sec": 0.0,
            "end_sec": 1.0,
            "text": "тихая фраза",
            "avg_logprob": -3.0,
            "no_speech_prob": 0.99,
            "compression_ratio": 4.0,
        }
    ]

    decorated_rows, _ = worker._decorate_segments(raw_segments, "unused.wav", 1.0)

    assert len(decorated_rows) == 1
    assert decorated_rows[0]["text"] not in {"<Тишина>", "<Музыка>", "<Неразборчиво>"}
    assert decorated_rows[0]["label"] == "speech"


def test_decorate_segments_filters_known_artifacts():
    worker = TranscriptionWorker()
    raw_segments = [
        {
            "start_sec": 0.0,
            "end_sec": 1.0,
            "text": "Субтитры сделал DimaTorzok",
            "avg_logprob": -0.1,
            "no_speech_prob": 0.05,
            "compression_ratio": 1.1,
        },
        {
            "start_sec": 1.0,
            "end_sec": 2.0,
            "text": "Продолжение следует...",
            "avg_logprob": -0.1,
            "no_speech_prob": 0.05,
            "compression_ratio": 1.1,
        },
        {
            "start_sec": 2.0,
            "end_sec": 3.0,
            "text": "Основная часть лекции",
            "avg_logprob": -0.3,
            "no_speech_prob": 0.02,
            "compression_ratio": 1.1,
        },
    ]

    decorated_rows, tagging_stats = worker._decorate_segments(raw_segments, "unused.wav", 3.0)

    assert [row["text"] for row in decorated_rows] == ["Основная часть лекции"]
    assert tagging_stats["filtered_segments"] == 2
    assert tagging_stats["filtered_reason_counts"]["ARTIFACT_TEXT_PATTERN"] == 2


def test_decorate_segments_keeps_filtered_rows_when_disabled(monkeypatch):
    monkeypatch.setenv("WHISPER_DROP_FILTERED_SEGMENTS", "false")
    worker = TranscriptionWorker()
    raw_segments = [
        {
            "start_sec": 0.0,
            "end_sec": 1.0,
            "text": "Субтитры сделал DimaTorzok",
            "avg_logprob": -0.1,
            "no_speech_prob": 0.05,
            "compression_ratio": 1.1,
        }
    ]

    decorated_rows, tagging_stats = worker._decorate_segments(raw_segments, "unused.wav", 1.0)

    assert len(decorated_rows) == 1
    assert decorated_rows[0]["quality_flags"]["filter_reasons"] == ["ARTIFACT_TEXT_PATTERN"]
    assert tagging_stats["kept_filtered_segments"] == 1


def test_chunk_plan_for_long_audio(monkeypatch):
    monkeypatch.setenv("WHISPER_LONG_AUDIO_MIN_SEC", "120")
    monkeypatch.setenv("WHISPER_LONG_AUDIO_CHUNK_SEC", "60")
    monkeypatch.setenv("WHISPER_LONG_AUDIO_OVERLAP_SEC", "5")
    worker = TranscriptionWorker()

    assert worker._build_chunk_plan(100.0) == [(0.0, 100.0)]
    assert worker._build_chunk_plan(130.0) == [(0.0, 60.0), (55.0, 115.0), (110.0, 130.0)]


def test_transcribe_with_chunking_deduplicates_boundary_and_uses_chunk_decode_policy():
    worker = TranscriptionWorker()
    worker._build_chunk_plan = lambda _: [(0.0, 40.0), (32.0, 70.0)]
    worker._extract_chunk = lambda *args, **kwargs: None
    calls: list[tuple[str | None, bool]] = []

    async def fake_run_transcribe(model, audio_path, initial_prompt, condition_on_previous_text):
        calls.append((initial_prompt, condition_on_previous_text))
        if len(calls) == 1:
            return (
                [
                    {
                        "start_sec": 30.0,
                        "end_sec": 38.0,
                        "text": "ключевая фраза",
                        "avg_logprob": -0.6,
                        "no_speech_prob": 0.2,
                        "compression_ratio": 1.2,
                    },
                    {
                        "start_sec": 38.5,
                        "end_sec": 39.8,
                        "text": "дальше",
                        "avg_logprob": -0.4,
                        "no_speech_prob": 0.1,
                        "compression_ratio": 1.0,
                    },
                ],
                SimpleNamespace(duration=70.0),
            )
        return (
            [
                {
                    "start_sec": 0.0,
                    "end_sec": 6.5,
                    "text": "ключевая фраза",
                    "avg_logprob": -0.2,
                    "no_speech_prob": 0.05,
                    "compression_ratio": 1.0,
                },
                {
                    "start_sec": 7.0,
                    "end_sec": 15.0,
                    "text": "новый тезис",
                    "avg_logprob": -0.4,
                    "no_speech_prob": 0.05,
                    "compression_ratio": 1.0,
                },
            ],
            SimpleNamespace(duration=70.0),
        )

    worker._run_transcribe = fake_run_transcribe

    rows, info = asyncio.run(
        worker._transcribe_with_chunking(
            model=None,
            prepared_audio_path="unused.wav",
            prompt="стартовый контекст",
            condition_on_previous_text=True,
            audio_duration_sec=70.0,
        )
    )

    assert info["duration"] == 70.0
    assert [row["text"] for row in rows].count("ключевая фраза") == 1
    assert calls == [("стартовый контекст", True), (None, False)]
"""
