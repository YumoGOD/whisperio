import asyncio
import threading
import time
from types import SimpleNamespace

from app.queue_worker import TranscriptionWorker


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


def test_decorate_segments_skips_tagging_when_disabled(monkeypatch):
    monkeypatch.setenv("WHISPER_ENABLE_TAGS", "false")
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
            "label": "unclear",
            "text": "<Неразборчиво>",
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
            "label": "unclear",
            "text": "<Неразборчиво>",
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
        "speech": 0,
    }
    assert tagging_stats["gap_metrics"] == []


def test_chunk_plan_for_long_audio(monkeypatch):
    monkeypatch.setenv("WHISPER_LONG_AUDIO_MIN_SEC", "120")
    monkeypatch.setenv("WHISPER_LONG_AUDIO_CHUNK_SEC", "60")
    monkeypatch.setenv("WHISPER_LONG_AUDIO_OVERLAP_SEC", "5")
    worker = TranscriptionWorker()

    assert worker._build_chunk_plan(100.0) == [(0.0, 100.0)]
    assert worker._build_chunk_plan(130.0) == [(0.0, 60.0), (55.0, 115.0), (110.0, 130.0)]
