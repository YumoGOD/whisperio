from __future__ import annotations

import json
import logging
from inspect import signature
from pathlib import Path
from time import perf_counter
from typing import Any, Callable

import torch

from faster_whisper import WhisperModel

from app.config import Settings
from app.transcription.audio import extract_chunk, prepare_audio, probe_duration_seconds
from app.transcription.chunking import build_chunks, merge_segments, normalize_text, replace_transcript_artifacts
from app.transcription.exports import write_exports
from app.transcription.glossary import (
    GlossaryContext,
    apply_hard_normalization,
    build_glossary_context,
    should_drop_glossary_repetition,
)
from app.transcription.profiles import resolve_profile

logger = logging.getLogger(__name__)

# Только NVIDIA CUDA (см. Dockerfile и docker-compose).
_WHISPER_DEVICE = "cuda"

ProgressCallback = Callable[[float, str], None]


def elapsed_since(started_at: float) -> float:
    return round(perf_counter() - started_at, 3)


def _build_context_tail(segments: list[dict[str, Any]], max_chars: int = 400) -> str | None:
    """Return the trailing text of processed segments to use as inter-chunk context."""
    if not segments:
        return None
    parts: list[str] = []
    total = 0
    for seg in reversed(segments[-10:]):
        text = seg.get("text", "").strip()
        if not text:
            continue
        if total + len(text) + 1 > max_chars:
            break
        parts.insert(0, text)
        total += len(text) + 1
    return " ".join(parts) if parts else None


def _merge_prompts(tail: str | None, glossary_prompt: str, max_chars: int = 1400) -> str:
    """Combine prev-chunk tail (priority) and glossary prompt within max_chars."""
    if not tail:
        return glossary_prompt[:max_chars]
    tail = tail[:max_chars]
    remaining = max_chars - len(tail) - 1
    if glossary_prompt and remaining > 20:
        return f"{tail} {glossary_prompt[:remaining]}"
    return tail


class TranscriptionPipeline:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._model: WhisperModel | None = None

    @property
    def model(self) -> WhisperModel:
        if self._model is None:
            model_name = self.settings.whisper_model
            logger.info(
                "Загрузка модели faster-whisper: model=%s device=%s compute_type=%s",
                model_name,
                _WHISPER_DEVICE,
                self.settings.whisper_compute_type,
            )
            try:
                self._model = WhisperModel(
                    model_name,
                    device=_WHISPER_DEVICE,
                    compute_type=self.settings.whisper_compute_type,
                    download_root=str(self.settings.whisper_download_root)
                    if self.settings.whisper_download_root
                    else None,
                )
            except Exception as exc:
                download_root = self.settings.whisper_download_root or Path("./data/models")
                raise RuntimeError(
                    "Не удалось загрузить модель faster-whisper. Если это первый запуск, worker должен скачать "
                    f"'{model_name}' с Hugging Face, либо нужно указать локальный путь к CTranslate2-модели. "
                    f"Текущие настройки: WHISPER_MODEL={model_name!r}, WHISPER_DOWNLOAD_ROOT={str(download_root)!r}. "
                    "Для офлайн-режима выполните `docker compose run --rm worker python scripts/download_model.py "
                    "large-v3 --output-dir /app/data/models/faster-whisper-large-v3`, затем укажите в .env "
                    "WHISPER_MODEL=/app/data/models/faster-whisper-large-v3. "
                    f"Исходная ошибка: {exc}"
                ) from exc
        return self._model

    def run(
        self,
        *,
        job_id: str,
        input_path: Path,
        original_filename: str,
        params: dict[str, Any],
        progress_callback: ProgressCallback | None = None,
    ) -> dict[str, Any]:
        started = perf_counter()
        profile_name = params.get("profile") or self.settings.default_profile
        profile = resolve_profile(profile_name, self.settings)
        glossary_context = build_glossary_context(self.settings, params)
        job_work_dir = self.settings.work_dir / job_id
        chunk_dir = job_work_dir / "chunks"
        transcript_dir = self.settings.transcript_dir / job_id
        prepared_path = job_work_dir / "prepared_16k_mono.wav"

        stage_started = perf_counter()
        self._progress(progress_callback, 0.05, "анализ длительности")
        duration_seconds = probe_duration_seconds(input_path)
        probe_elapsed = elapsed_since(stage_started)

        stage_started = perf_counter()
        self._progress(progress_callback, 0.10, "подготовка аудио")
        prepare_audio(input_path, prepared_path, self.settings)
        preprocess_elapsed = elapsed_since(stage_started)

        stage_started = perf_counter()
        chunks = build_chunks(
            duration_seconds=duration_seconds,
            chunk_seconds=int(params.get("chunk_seconds") or self.settings.chunk_seconds),
            overlap_seconds=int(params.get("chunk_overlap_seconds") or self.settings.chunk_overlap_seconds),
        )
        if not chunks:
            raise RuntimeError("Не удалось определить длительность аудиофайла")
        chunk_plan_elapsed = elapsed_since(stage_started)

        all_segments: list[dict[str, Any]] = []
        diagnostics: dict[str, Any] = {
            "job_id": job_id,
            "original_filename": original_filename,
            "profile_name": profile_name,
            "profile": profile,
            "job_context": {
                "audio_type": params.get("audio_type") or "",
                "audio_context": params.get("audio_context") or "",
                "expected_content": params.get("expected_content") or "",
                "dynamic_terms": params.get("dynamic_terms") or "",
            },
            "glossary": glossary_context.diagnostics(),
            "duration_seconds": duration_seconds,
            "chunks": [],
            "settings": {
                "model": self.settings.whisper_model,
                "device": _WHISPER_DEVICE,
                "compute_type": self.settings.whisper_compute_type,
                "language": self.settings.whisper_language or None,
                "task": self.settings.whisper_task,
                "target_sample_rate": self.settings.target_sample_rate,
                "enable_loudnorm": self.settings.enable_loudnorm,
                "chunk_seconds": int(params.get("chunk_seconds") or self.settings.chunk_seconds),
                "chunk_overlap_seconds": int(params.get("chunk_overlap_seconds") or self.settings.chunk_overlap_seconds),
            },
            "stage_timings": {
                "probe_seconds": probe_elapsed,
                "preprocess_seconds": preprocess_elapsed,
                "chunk_plan_seconds": chunk_plan_elapsed,
                "chunk_extract_seconds": 0.0,
                "chunk_transcribe_seconds": 0.0,
                "postprocess_seconds": 0.0,
                "export_seconds": 0.0,
            },
        }

        prev_tail_text: str | None = None
        for chunk in chunks:
            chunk_progress_start = 0.15 + 0.75 * (chunk.index / len(chunks))
            self._progress(progress_callback, chunk_progress_start, f"фрагмент {chunk.index + 1}/{len(chunks)}")
            chunk_path = chunk_dir / f"chunk_{chunk.index:04d}_{int(chunk.start)}_{int(chunk.end)}.wav"
            extract_started = perf_counter()
            extract_chunk(prepared_path, chunk_path, chunk.start, chunk.duration)
            extract_elapsed = elapsed_since(extract_started)
            transcribe_started = perf_counter()
            profile_attempt = dict(profile)
            for attempt in range(3):
                try:
                    segments, info = self._transcribe_chunk(
                        chunk_path, chunk.start, profile_attempt, glossary_context, prev_tail_text
                    )
                    break
                except Exception as exc:
                    logger.warning(
                        "Чанк %s: попытка %d/3 не удалась: %s", chunk_path.name, attempt + 1, exc
                    )
                    if attempt == 2:
                        raise
                    torch.cuda.empty_cache()
                    if attempt == 0:
                        profile_attempt["beam_size"] = max(1, profile["beam_size"] - 2)
                        profile_attempt["best_of"] = max(1, profile["best_of"] - 2)
                    else:
                        profile_attempt["beam_size"] = 1
                        profile_attempt["best_of"] = 1
                        profile_attempt["temperature"] = [0.0]
            prev_tail_text = _build_context_tail(segments)
            transcribe_elapsed = elapsed_since(transcribe_started)
            diagnostics["stage_timings"]["chunk_extract_seconds"] = round(
                diagnostics["stage_timings"]["chunk_extract_seconds"] + extract_elapsed,
                3,
            )
            diagnostics["stage_timings"]["chunk_transcribe_seconds"] = round(
                diagnostics["stage_timings"]["chunk_transcribe_seconds"] + transcribe_elapsed,
                3,
            )
            all_segments.extend(segments)
            diagnostics["chunks"].append(
                {
                    "index": chunk.index,
                    "start": chunk.start,
                    "end": chunk.end,
                    "path": str(chunk_path),
                    "segments": len(segments),
                    "extract_seconds": extract_elapsed,
                    "transcribe_seconds": transcribe_elapsed,
                    "audio_seconds": round(chunk.duration, 3),
                    "real_time_factor": round(transcribe_elapsed / chunk.duration, 4) if chunk.duration else None,
                    "language": getattr(info, "language", None),
                    "language_probability": getattr(info, "language_probability", None),
                }
            )

        stage_started = perf_counter()
        self._progress(progress_callback, 0.92, "постобработка")
        merged_segments = merge_segments(all_segments, int(params.get("chunk_overlap_seconds") or self.settings.chunk_overlap_seconds))
        merged_segments, artifact_stats = replace_transcript_artifacts(merged_segments)
        full_text = " ".join(segment["text"] for segment in merged_segments)
        diagnostics["stage_timings"]["postprocess_seconds"] = elapsed_since(stage_started)
        diagnostics["merged_segments"] = len(merged_segments)
        diagnostics["artifact_postprocessing"] = artifact_stats
        diagnostics["glossary"] = glossary_context.diagnostics()

        stage_started = perf_counter()
        exports = write_exports(
            transcript_dir=transcript_dir,
            job_id=job_id,
            text=full_text,
            segments=merged_segments,
            metadata=diagnostics,
        )
        diagnostics["stage_timings"]["export_seconds"] = elapsed_since(stage_started)
        elapsed_seconds = perf_counter() - started
        diagnostics["elapsed_seconds"] = round(elapsed_seconds, 3)
        diagnostics["real_time_factor"] = round(elapsed_seconds / duration_seconds, 4) if duration_seconds else None
        exports["json"].write_text(
            json.dumps({"text": full_text, "segments": merged_segments, "metadata": diagnostics}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (transcript_dir / "diagnostics.json").write_text(
            json.dumps(diagnostics, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        self._progress(progress_callback, 0.98, "сохранение результатов")
        return {
            "text": full_text,
            "segments": merged_segments,
            "transcript_dir": str(transcript_dir),
            "prepared_path": str(prepared_path),
            "duration_seconds": duration_seconds,
            "exports": {key: str(path) for key, path in exports.items()},
            "diagnostics": diagnostics,
        }

    def _transcribe_chunk(
        self,
        chunk_path: Path,
        offset_seconds: float,
        profile: dict[str, Any],
        glossary_context: GlossaryContext,
        prev_tail_text: str | None = None,
    ):
        vad_filter = bool(profile.get("vad_filter", False))
        vad_parameters = profile.get("vad_parameters") if vad_filter else None
        transcribe_kwargs: dict[str, Any] = {
            "beam_size": int(profile["beam_size"]),
            "best_of": int(profile["best_of"]),
            "patience": float(profile["patience"]),
            "temperature": profile["temperature"],
            "compression_ratio_threshold": float(profile["compression_ratio_threshold"]),
            "log_prob_threshold": float(profile["log_prob_threshold"]),
            "no_speech_threshold": float(profile["no_speech_threshold"]),
            "condition_on_previous_text": bool(profile["condition_on_previous_text"]),
            "word_timestamps": bool(profile["word_timestamps"]),
            "language": self.settings.whisper_language or None,
            "task": self.settings.whisper_task,
            "vad_filter": vad_filter,
            "vad_parameters": vad_parameters,
        }
        if prev_tail_text or glossary_context.initial_prompt:
            transcribe_kwargs["initial_prompt"] = _merge_prompts(
                prev_tail_text,
                glossary_context.initial_prompt,
                self.settings.glossary_prompt_max_chars,
            )
        if self.settings.glossary_enable_hotwords and glossary_context.hotwords and self._supports_hotwords():
            if glossary_context.initial_prompt:
                logger.debug("Hotwords skipped because initial_prompt already carries glossary context")
            else:
                transcribe_kwargs["hotwords"] = ",".join(glossary_context.hotwords)
        segments_iter, info = self.model.transcribe(str(chunk_path), **transcribe_kwargs)
        segments: list[dict[str, Any]] = []
        for segment in segments_iter:
            text = normalize_text(segment.text)
            compression_ratio = getattr(segment, "compression_ratio", None)
            if should_drop_glossary_repetition(
                text,
                glossary_context,
                compression_ratio,
                self.settings.glossary_repetition_compression_threshold,
            ):
                logger.warning(
                    "Сегмент пропущен как вероятная словарная галлюцинация: start=%.2f end=%.2f compression_ratio=%s text=%r",
                    float(segment.start) + offset_seconds,
                    float(segment.end) + offset_seconds,
                    compression_ratio,
                    text[:220],
                )
                continue
            text = apply_hard_normalization(
                text,
                glossary_context,
                enabled=self.settings.glossary_enable_hard_normalization,
            )
            segments.append(
                {
                    "id": len(segments),
                    "start": float(segment.start) + offset_seconds,
                    "end": float(segment.end) + offset_seconds,
                    "text": text,
                    "avg_logprob": getattr(segment, "avg_logprob", None),
                    "no_speech_prob": getattr(segment, "no_speech_prob", None),
                    "compression_ratio": compression_ratio,
                }
            )
        return segments, info

    def _supports_hotwords(self) -> bool:
        try:
            return "hotwords" in signature(self.model.transcribe).parameters
        except (TypeError, ValueError):
            return False

    @staticmethod
    def _progress(callback: ProgressCallback | None, progress: float, stage: str) -> None:
        logger.info("Прогресс pipeline %.2f: %s", progress, stage)
        if callback:
            callback(progress, stage)
