"""faster-whisper обёртка с lazy-init и CPU-fallback.

Один Transcriber на процесс воркера (модель ~1.5 GB на диске и в RAM).
Возвращает абсолютные таймкоды относительно исходного WAV — faster-whisper
с clip_timestamps уже отдаёт их в абсолютной системе координат файла.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any, Callable

from app.config import settings

log = logging.getLogger(__name__)

# WAV для VAD/whisper всегда 16 kHz mono (см. audio.to_wav_16k_mono).
# BatchedInferencePipeline ожидает clip_timestamps в сэмплах, а не секундах.
_SAMPLE_RATE = 16000


@dataclass
class TranscribedSegment:
    start: float
    end: float
    text: str


@dataclass
class TranscriptionResult:
    segments: list[TranscribedSegment]
    detected_language: str | None


class Transcriber:
    """Тонкая обёртка над faster-whisper. Безопасна к повторному вызову."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._model: Any = None  # faster_whisper.WhisperModel
        self._batched: Any = None  # faster_whisper.BatchedInferencePipeline | None
        self._device: str | None = None
        self._compute_type: str | None = None

    # ------------------------- init -------------------------

    @staticmethod
    def _pick_device_and_compute() -> tuple[str, str]:
        """Согласовать device/compute с реальностью железа.

        - cuda/auto + CUDA доступна → cuda + settings.WHISPER_COMPUTE_TYPE
        - cuda/auto + CUDA нет → CPU/int8 (warning)
        - cpu + compute несовместим с CPU (float16) → CPU/int8 (warning)
        """
        device = settings.WHISPER_DEVICE.lower().strip()
        compute = settings.WHISPER_COMPUTE_TYPE.strip()

        if device in ("cuda", "auto"):
            try:
                import torch  # noqa: PLC0415
                if torch.cuda.is_available():
                    return "cuda", compute
                log.warning("CUDA requested (%s) but not available; falling back to cpu/int8",
                            device)
            except ImportError:
                log.warning("torch not importable; falling back to cpu/int8")
            return "cpu", "int8"

        if compute in ("float16", "int8_float16"):
            log.warning("compute_type=%s not supported on CPU; using int8 instead", compute)
            return "cpu", "int8"
        return device, compute

    def _load(self) -> None:
        if self._model is not None:
            return
        with self._lock:
            if self._model is not None:
                return
            try:
                from faster_whisper import WhisperModel  # noqa: PLC0415
            except ImportError as e:
                raise RuntimeError(
                    "faster-whisper is not installed; run pip install -r requirements.txt"
                ) from e

            device, compute = self._pick_device_and_compute()
            log.info("loading faster-whisper model=%s device=%s compute=%s",
                     settings.WHISPER_MODEL, device, compute)
            try:
                self._model = WhisperModel(
                    settings.WHISPER_MODEL,
                    device=device,
                    compute_type=compute,
                    download_root=str(settings.WHISPER_MODELS_DIR),
                )
            except Exception:
                # Самый частый кейс — CUDA OOM или несовместимый compute_type.
                # Откатываемся на CPU/int8 без падения процесса.
                log.exception("WhisperModel init failed on %s/%s; retrying on cpu/int8",
                              device, compute)
                self._model = WhisperModel(
                    settings.WHISPER_MODEL,
                    device="cpu",
                    compute_type="int8",
                    download_root=str(settings.WHISPER_MODELS_DIR),
                )
                device, compute = "cpu", "int8"
            self._device, self._compute_type = device, compute
            log.info("faster-whisper loaded on %s/%s", device, compute)

            try:
                from faster_whisper import BatchedInferencePipeline  # noqa: PLC0415
                self._batched = BatchedInferencePipeline(model=self._model)
                log.info("BatchedInferencePipeline ready (batch_size=%d)",
                         settings.WHISPER_BATCH_SIZE)
            except (ImportError, AttributeError):
                self._batched = None
                log.info("BatchedInferencePipeline unavailable; using WhisperModel directly")

    # ------------------------- public API -------------------------

    @property
    def device(self) -> str | None:
        return self._device

    @property
    def compute_type(self) -> str | None:
        return self._compute_type

    def transcribe_segments(
        self,
        wav_path: Path | str,
        segments: list[tuple[float, float]],
        language: str | None,
        on_progress: Callable[[float], None] | None = None,
        total_duration: float | None = None,
    ) -> TranscriptionResult:
        """Транскрибировать речевые сегменты.

        wav_path — путь к 16 kHz mono WAV (см. audio.to_wav_16k_mono).
        segments — список (start, end) из VAD; пустой ⇒ транскрибируем весь файл.
        language — 'ru'/'en'/...; None ⇒ авто-определение.
        on_progress — опциональный callback, вызывается после каждого готового
            сегмента whisper'а со значением 0..1. Прогресс оценивается как
            `segment.end / total_duration` (если total_duration не задан —
            используем end последнего VAD-сегмента, иначе callback не вызываем).
        """
        self._load()

        if on_progress is not None and total_duration is None:
            total_duration = segments[-1][1] if segments else None

        if not segments:
            log.info("transcribe_segments: empty VAD segments — transcribing whole file")

        common: dict[str, Any] = {
            "beam_size": settings.BEAM_SIZE,
            "language": language,
            # Наша Silero VAD — primary; не даём faster-whisper резать ещё раз.
            "vad_filter": False,
        }
        path_str = str(wav_path)

        try:
            if self._batched is not None:
                # BatchedInferencePipeline ожидает clip_timestamps как
                # list[{"start": int_samples, "end": int_samples}] — индексы
                # в audio-массиве при 16 kHz (см. faster_whisper/vad.collect_chunks).
                batched_kwargs = dict(common)
                if segments:
                    batched_kwargs["clip_timestamps"] = [
                        {"start": int(s * _SAMPLE_RATE), "end": int(e * _SAMPLE_RATE)}
                        for s, e in segments
                    ]
                segs_iter, info = self._batched.transcribe(
                    path_str,
                    batch_size=settings.WHISPER_BATCH_SIZE,
                    **batched_kwargs,
                )
            else:
                # WhisperModel.transcribe принимает flat list[float] либо CSV-строку.
                model_kwargs = dict(common)
                if segments:
                    model_kwargs["clip_timestamps"] = [
                        t for pair in segments for t in pair
                    ]
                segs_iter, info = self._model.transcribe(path_str, **model_kwargs)
        except Exception as e:
            raise RuntimeError(f"faster-whisper transcribe failed: {e}") from e

        out: list[TranscribedSegment] = []
        for s in segs_iter:
            end_t = float(s.end)
            text = (s.text or "").strip()
            if text:
                out.append(TranscribedSegment(
                    start=float(s.start),
                    end=end_t,
                    text=text,
                ))
            if on_progress is not None and total_duration and total_duration > 0:
                try:
                    on_progress(max(0.0, min(1.0, end_t / total_duration)))
                except Exception:  # noqa: BLE001 — прогресс не должен ронять транскрипцию
                    log.exception("on_progress callback failed; ignoring")

        detected = getattr(info, "language", None)
        return TranscriptionResult(segments=out, detected_language=detected)


# Process-wide singleton — воркер импортирует его и переиспользует.
transcriber = Transcriber()
