import asyncio
import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path

try:
    import psutil
except ImportError:  # pragma: no cover
    psutil = None

try:
    import resource
except ImportError:  # pragma: no cover - not available on Windows
    resource = None

from faster_whisper import WhisperModel

from app.db import get_connection, get_queue_stats
from app.logging_utils import bind_log_context, log_event, reset_log_context
from app.models import JobStage, JobStatus
from app.transcription.inference import InferenceResult, transcribe_windows
from app.transcription.pipeline import QualityFirstPipeline
from app.transcription.preprocess import (
    AudioPreprocessError,
    cleanup_prepared_audio,
    prepare_audio,
    probe_audio,
)
from app.transcription.segmenter import AudioWindow
from app.transcription.settings import DecodeSettings, load_transcription_settings


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_env_int(name: str, default: int, *, min_value: int = 0) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        parsed = int(raw)
    except ValueError:
        return default
    return max(min_value, parsed)


class TranscriptionWorker:
    def __init__(self) -> None:
        self.logger = logging.getLogger("whisperio.worker")
        self.settings = load_transcription_settings()
        self.model: WhisperModel | None = None
        self._model_lock = asyncio.Lock()
        self._transcribe_lock = asyncio.Lock()
        self._pipeline = QualityFirstPipeline(settings=self.settings, transcribe_fn=self._run_transcribe)
        self.sla_rtf_threshold = max(0.01, float(os.getenv("WHISPER_SLA_RTF_THRESHOLD", "0.25")))
        self.resource_snapshot_interval_sec = _safe_env_int("LOG_SNAPSHOT_INTERVAL_SEC", 10, min_value=1)
        self._last_resource_snapshot_ts = 0.0

    def process_claimed_job(self, job_id: str) -> None:
        asyncio.run(self._process_job(job_id))

    async def _get_model(self) -> WhisperModel:
        if self.model is not None:
            return self.model
        async with self._model_lock:
            if self.model is None:
                self.model = await asyncio.to_thread(
                    WhisperModel,
                    self.settings.model_name,
                    device=self.settings.model_device,
                    compute_type=self.settings.model_compute_type,
                    cpu_threads=self.settings.model_cpu_threads,
                    num_workers=self.settings.model_workers,
                )
        return self.model

    async def _run_transcribe(
        self,
        model: WhisperModel,
        audio_path: str,
        windows: list[AudioWindow],
        decode: DecodeSettings,
    ) -> InferenceResult:
        async with self._transcribe_lock:
            return await asyncio.to_thread(
                transcribe_windows,
                model,
                audio_path=audio_path,
                windows=windows,
                decode=decode,
                language=self.settings.language,
                task=self.settings.task,
            )

    def _is_delete_requested(self, conn, job_id: str) -> bool:
        row = conn.execute(
            "SELECT delete_requested FROM transcription_jobs WHERE id = ?",
            (job_id,),
        ).fetchone()
        if not row:
            return False
        return bool(row["delete_requested"])

    def _finalize_delete(self, job_id: str, prepared_audio_path: str | None) -> None:
        with get_connection() as conn:
            row = conn.execute(
                "SELECT stored_path, prepared_audio_path FROM transcription_jobs WHERE id = ?",
                (job_id,),
            ).fetchone()
            if not row:
                return
            stored_path = row["stored_path"]
            kept_prepared_path = row["prepared_audio_path"]
            conn.execute("DELETE FROM transcription_jobs WHERE id = ?", (job_id,))

        for candidate in (stored_path, kept_prepared_path, prepared_audio_path):
            if not candidate:
                continue
            path = Path(candidate)
            if path.exists() and path.is_file():
                path.unlink(missing_ok=True)

        self._log_event(
            event="job_deleted_after_processing_step",
            job_id=job_id,
            stage=JobStage.COMPLETED,
            status=JobStatus.DONE,
            progress=100.0,
        )

    async def _process_job(self, job_id: str) -> None:
        started_wall_time = time.perf_counter()
        prepared_audio_path: str | None = None
        source_duration_sec: float = 0.0
        decode_duration_ms: int | None = None
        preprocess_duration_ms: int | None = None
        transcribe_duration_ms: int | None = None
        persist_duration_ms: int | None = None
        quality_flags_json: str | None = None
        request_id: str | None = None
        audio_size_bytes: int | None = None

        log_context_token = bind_log_context(worker_pid=os.getpid(), job_id=job_id)
        try:
            with get_connection() as conn:
                job = conn.execute(
                    "SELECT stored_path, request_id, delete_requested FROM transcription_jobs WHERE id = ?",
                    (job_id,),
                ).fetchone()
            if not job:
                self._log_event(
                    event="job_not_found",
                    job_id=job_id,
                    stage=JobStage.FAILED,
                    error_code="JOB_NOT_FOUND",
                )
                return

            request_id = job["request_id"]
            reset_log_context(log_context_token)
            log_context_token = bind_log_context(
                worker_pid=os.getpid(),
                job_id=job_id,
                request_id=request_id,
            )

            with get_connection() as conn:
                self._update_job(
                    conn=conn,
                    job_id=job_id,
                    status=JobStatus.PROCESSING,
                    stage=JobStage.PREPARING,
                    progress=5.0,
                    status_message="Подготовка к распознаванию",
                    started_at=utc_now_iso(),
                    clear_error=True,
                )

            audio_path = str(job["stored_path"])
            if bool(job["delete_requested"]):
                self._finalize_delete(job_id, prepared_audio_path)
                return
            if Path(audio_path).exists():
                audio_size_bytes = Path(audio_path).stat().st_size

            self._log_event(
                event="job_started",
                job_id=job_id,
                stage=JobStage.PREPARING,
                status=JobStatus.PROCESSING,
                progress=5.0,
                extra_fields={
                    "audio_size_bytes": audio_size_bytes,
                    "profile": self.settings.profile.name,
                    "model_name": self.settings.model_name,
                    "language": self.settings.language,
                },
            )
            self._log_resource_snapshot(job_id=job_id, stage=JobStage.PREPARING, force=True)

            decode_started = time.perf_counter()
            source_probe = await asyncio.to_thread(
                probe_audio,
                audio_path,
                timeout_sec=self.settings.preprocess.timeout_sec,
            )
            decode_duration_ms = int((time.perf_counter() - decode_started) * 1000)
            source_duration_sec = source_probe.duration_sec

            with get_connection() as conn:
                self._update_job(
                    conn=conn,
                    job_id=job_id,
                    status=JobStatus.PROCESSING,
                    stage=JobStage.PREPROCESSING,
                    progress=12.0,
                    status_message="Шумоподавление и нормализация аудио",
                    decode_duration_ms=decode_duration_ms,
                )

            preprocess_started = time.perf_counter()
            prepared_audio_path = await asyncio.to_thread(
                prepare_audio,
                audio_path,
                sample_rate=self.settings.preprocess.sample_rate,
                timeout_sec=self.settings.preprocess.timeout_sec,
                filters=self.settings.preprocess.ffmpeg_filters,
            )
            preprocess_duration_ms = int((time.perf_counter() - preprocess_started) * 1000)

            with get_connection() as conn:
                self._update_job(
                    conn=conn,
                    job_id=job_id,
                    status=JobStatus.PROCESSING,
                    stage=JobStage.PREPROCESSING,
                    progress=22.0,
                    status_message="Аудио подготовлено",
                    preprocess_duration_ms=preprocess_duration_ms,
                    decode_duration_ms=decode_duration_ms,
                    prepared_audio_path=(
                        prepared_audio_path if self.settings.preprocess.keep_prepared_audio else None
                    ),
                )
                if self._is_delete_requested(conn, job_id):
                    self._finalize_delete(job_id, prepared_audio_path)
                    return

            with get_connection() as conn:
                self._update_job(
                    conn=conn,
                    job_id=job_id,
                    status=JobStatus.PROCESSING,
                    stage=JobStage.TRANSCRIBING,
                    progress=30.0,
                    status_message="Транскрибация сегментов речи",
                )

            model = await self._get_model()
            transcribe_started = time.perf_counter()
            pipeline_result = await self._pipeline.run(
                model=model,
                prepared_audio_path=prepared_audio_path or audio_path,
                fallback_duration_sec=source_duration_sec,
            )
            transcribe_duration_ms = int((time.perf_counter() - transcribe_started) * 1000)

            quality_flags_json = json.dumps(pipeline_result.quality_payload, ensure_ascii=False)
            segment_rows = [
                (
                    job_id,
                    idx,
                    float(segment["start_sec"]),
                    float(segment["end_sec"]),
                    str(segment["text"]),
                    str(segment["label"]),
                    float(segment["confidence"]),
                    json.dumps(segment["quality_flags"], ensure_ascii=False),
                )
                for idx, segment in enumerate(pipeline_result.segment_rows)
            ]

            self._log_event(
                event="stage_finished",
                job_id=job_id,
                stage=JobStage.TRANSCRIBING,
                status=JobStatus.PROCESSING,
                progress=82.0,
                duration_ms=transcribe_duration_ms,
                segment_count=len(segment_rows),
                audio_duration_sec=pipeline_result.duration_sec,
                extra_fields={
                    "vad_window_count": pipeline_result.vad_window_count,
                    "rescue_window_count": pipeline_result.rescue_window_count,
                    "rescue_applied_count": pipeline_result.quality_payload.get("rescue_applied_count", 0),
                },
            )

            with get_connection() as conn:
                self._update_job(
                    conn=conn,
                    job_id=job_id,
                    status=JobStatus.PROCESSING,
                    stage=JobStage.SAVING_SEGMENTS,
                    progress=88.0,
                    status_message=f"Сохранение сегментов: {len(segment_rows)}",
                    transcribe_duration_ms=transcribe_duration_ms,
                    preprocess_duration_ms=preprocess_duration_ms,
                    decode_duration_ms=decode_duration_ms,
                    quality_flags=quality_flags_json,
                    prepared_audio_path=(
                        prepared_audio_path if self.settings.preprocess.keep_prepared_audio else None
                    ),
                )
                if self._is_delete_requested(conn, job_id):
                    self._finalize_delete(job_id, prepared_audio_path)
                    return

            persist_started = time.perf_counter()
            with get_connection() as conn:
                conn.execute("DELETE FROM transcription_segments WHERE job_id = ?", (job_id,))
                if segment_rows:
                    conn.executemany(
                        """
                        INSERT INTO transcription_segments
                        (job_id, idx, start_sec, end_sec, text, label, confidence, quality_flags)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        segment_rows,
                    )
                wall_duration_ms = int((time.perf_counter() - started_wall_time) * 1000)
                self._update_job(
                    conn=conn,
                    job_id=job_id,
                    status=JobStatus.DONE,
                    stage=JobStage.COMPLETED,
                    progress=100.0,
                    status_message="Транскрибация завершена",
                    finished_at=utc_now_iso(),
                    duration_sec=pipeline_result.duration_sec,
                    processing_duration_ms=wall_duration_ms,
                    transcribe_duration_ms=transcribe_duration_ms,
                    preprocess_duration_ms=preprocess_duration_ms,
                    decode_duration_ms=decode_duration_ms,
                    prepared_audio_path=(
                        prepared_audio_path if self.settings.preprocess.keep_prepared_audio else None
                    ),
                    quality_flags=quality_flags_json,
                    clear_error=True,
                )
            persist_duration_ms = int((time.perf_counter() - persist_started) * 1000)
            with get_connection() as conn:
                self._update_job(
                    conn=conn,
                    job_id=job_id,
                    persist_duration_ms=persist_duration_ms,
                )

            rtf = None
            effective_speed_x = None
            if pipeline_result.duration_sec > 0:
                processing_sec = wall_duration_ms / 1000.0
                rtf = round(processing_sec / pipeline_result.duration_sec, 6)
                effective_speed_x = round(pipeline_result.duration_sec / max(processing_sec, 1e-6), 6)

            self._log_event(
                event="job_completed",
                job_id=job_id,
                stage=JobStage.COMPLETED,
                status=JobStatus.DONE,
                progress=100.0,
                duration_ms=wall_duration_ms,
                segment_count=len(segment_rows),
                persist_duration_ms=persist_duration_ms,
                audio_duration_sec=pipeline_result.duration_sec,
                extra_fields={
                    "decode_duration_ms": decode_duration_ms,
                    "preprocess_duration_ms": preprocess_duration_ms,
                    "transcribe_duration_ms": transcribe_duration_ms,
                    "rtf": rtf,
                    "effective_speed_x": effective_speed_x,
                },
            )
            self._emit_sla_event_if_needed(
                rtf=rtf,
                job_id=job_id,
                stage=JobStage.COMPLETED,
                rescue_used=pipeline_result.rescue_window_count > 0,
            )
        except Exception as exc:  # noqa: BLE001
            wall_duration_ms = int((time.perf_counter() - started_wall_time) * 1000)
            self._log_event(
                event="stage_failed",
                job_id=job_id,
                stage=JobStage.FAILED,
                status=JobStatus.FAILED,
                progress=100.0,
                error_code=self._infer_error_code(exc),
                error_message=self._safe_error_message(exc),
                extra_fields={"failed_stage": "pipeline"},
            )
            with get_connection() as conn:
                status_message = "Ошибка обработки"
                if isinstance(exc, AudioPreprocessError):
                    status_message = "Ошибка подготовки аудио"
                self._update_job(
                    conn=conn,
                    job_id=job_id,
                    status=JobStatus.FAILED,
                    stage=JobStage.FAILED,
                    progress=100.0,
                    status_message=status_message,
                    finished_at=utc_now_iso(),
                    error=str(exc),
                    error_code=self._infer_error_code(exc),
                    processing_duration_ms=wall_duration_ms,
                    preprocess_duration_ms=preprocess_duration_ms,
                    decode_duration_ms=decode_duration_ms,
                    transcribe_duration_ms=transcribe_duration_ms,
                    persist_duration_ms=persist_duration_ms,
                    prepared_audio_path=(
                        prepared_audio_path if self.settings.preprocess.keep_prepared_audio else None
                    ),
                    quality_flags=quality_flags_json,
                )
            self._log_event(
                event="job_failed",
                job_id=job_id,
                stage=JobStage.FAILED,
                status=JobStatus.FAILED,
                progress=100.0,
                duration_ms=wall_duration_ms,
                error_code=self._infer_error_code(exc),
                error_message=self._safe_error_message(exc),
            )
        finally:
            cleanup_prepared_audio(
                prepared_audio_path,
                keep_prepared_audio=self.settings.preprocess.keep_prepared_audio,
            )
            self._log_resource_snapshot(job_id=job_id, stage=JobStage.COMPLETED, force=True)
            reset_log_context(log_context_token)

    def _update_job(
        self,
        conn,
        job_id: str,
        *,
        status: JobStatus | None = None,
        stage: JobStage | None = None,
        progress: float | None = None,
        status_message: str | None = None,
        started_at: str | None = None,
        finished_at: str | None = None,
        error: str | None = None,
        error_code: str | None = None,
        duration_sec: float | None = None,
        processing_duration_ms: int | None = None,
        transcribe_duration_ms: int | None = None,
        preprocess_duration_ms: int | None = None,
        decode_duration_ms: int | None = None,
        enhance_duration_ms: int | None = None,
        segment_duration_ms: int | None = None,
        decorate_duration_ms: int | None = None,
        persist_duration_ms: int | None = None,
        prepared_audio_path: str | None = None,
        quality_flags: str | None = None,
        clear_error: bool = False,
    ) -> None:
        fields: list[str] = []
        values: list[object] = []
        updates = {
            "status": status,
            "stage": stage,
            "progress": progress,
            "status_message": status_message,
            "started_at": started_at,
            "finished_at": finished_at,
            "error": error,
            "error_code": error_code,
            "duration_sec": duration_sec,
            "processing_duration_ms": processing_duration_ms,
            "transcribe_duration_ms": transcribe_duration_ms,
            "preprocess_duration_ms": preprocess_duration_ms,
            "decode_duration_ms": decode_duration_ms,
            "enhance_duration_ms": enhance_duration_ms,
            "segment_duration_ms": segment_duration_ms,
            "decorate_duration_ms": decorate_duration_ms,
            "persist_duration_ms": persist_duration_ms,
            "prepared_audio_path": prepared_audio_path,
            "quality_flags": quality_flags,
        }
        for key, value in updates.items():
            if value is not None:
                fields.append(f"{key} = ?")
                values.append(value)
        if clear_error:
            fields.append("error = NULL")
            fields.append("error_code = NULL")
        if not fields:
            return
        values.append(job_id)
        conn.execute(
            f"""
            UPDATE transcription_jobs
            SET {", ".join(fields)}
            WHERE id = ?
            """,
            values,
        )

    def _infer_error_code(self, exc: Exception) -> str:
        if isinstance(exc, AudioPreprocessError):
            return "AUDIO_PREPROCESS_FAILED"
        message = str(exc).lower()
        if "cuda" in message or "out of memory" in message:
            return "MODEL_RUNTIME_ERROR"
        if "no such file" in message or "not found" in message:
            return "AUDIO_FILE_NOT_FOUND"
        if "sqlite" in message or "database" in message:
            return "DB_WRITE_FAILED"
        return "TRANSCRIPTION_FAILED"

    def _safe_error_message(self, exc: Exception) -> str:
        if isinstance(exc, AudioPreprocessError):
            return "Audio preprocessing failed"
        return type(exc).__name__

    def _emit_sla_event_if_needed(
        self,
        *,
        rtf: float | None,
        job_id: str,
        stage: JobStage,
        rescue_used: bool,
    ) -> None:
        if rtf is None or rtf <= self.sla_rtf_threshold:
            return
        reason_code = "decode_slowdown"
        queue_stats = get_queue_stats()
        if queue_stats["queued"] > 0:
            reason_code = "cpu_saturation"
        if rescue_used:
            reason_code = "audio_complexity_high"
        self._log_event(
            event="sla_drift_detected",
            job_id=job_id,
            stage=stage,
            status=JobStatus.DONE,
            extra_fields={
                "reason_code": reason_code,
                "rtf": rtf,
                "rtf_threshold": self.sla_rtf_threshold,
                "queue_backlog": queue_stats["queued"],
                "processing_jobs": queue_stats["processing"],
                "rescue_used": rescue_used,
            },
        )

    def _log_resource_snapshot(self, *, job_id: str, stage: JobStage, force: bool = False) -> None:
        if psutil is None:
            return
        now = time.monotonic()
        if not force and (now - self._last_resource_snapshot_ts) < self.resource_snapshot_interval_sec:
            return
        self._last_resource_snapshot_ts = now

        process = psutil.Process(os.getpid())
        mem = process.memory_info()
        cpu_percent = process.cpu_percent(interval=None)
        num_threads = process.num_threads()
        open_files_count = len(process.open_files())
        ctx_switches = process.num_ctx_switches()
        max_rss_kb = None
        if resource is not None:
            usage = resource.getrusage(resource.RUSAGE_SELF)
            max_rss_kb = int(getattr(usage, "ru_maxrss", 0))
        self._log_event(
            event="worker_resource_snapshot",
            job_id=job_id,
            stage=stage,
            extra_fields={
                "worker_pid": os.getpid(),
                "cpu_percent": cpu_percent,
                "rss_bytes": mem.rss,
                "vms_bytes": mem.vms,
                "threads": num_threads,
                "open_files_count": open_files_count,
                "ctx_switches_voluntary": ctx_switches.voluntary,
                "ctx_switches_involuntary": ctx_switches.involuntary,
                "max_rss_kb": max_rss_kb,
            },
        )
        high_watermark_mb = _safe_env_int("LOG_RESOURCE_RSS_WARN_MB", 12288, min_value=1)
        if mem.rss > high_watermark_mb * 1024 * 1024:
            self._log_event(
                event="resource_pressure_warning",
                job_id=job_id,
                stage=stage,
                status=JobStatus.PROCESSING,
                extra_fields={
                    "reason_code": "rss_high_watermark",
                    "rss_bytes": mem.rss,
                    "rss_warn_mb": high_watermark_mb,
                },
            )

    def _log_event(
        self,
        *,
        event: str,
        job_id: str,
        stage: JobStage,
        status: JobStatus | None = None,
        progress: float | None = None,
        duration_ms: int | None = None,
        error_code: str | None = None,
        error_message: str | None = None,
        queue_size: int | None = None,
        segment_count: int | None = None,
        audio_duration_sec: float | None = None,
        persist_duration_ms: int | None = None,
        extra_fields: dict[str, object] | None = None,
    ) -> None:
        payload = {
            "event": event,
            "job_id": job_id,
            "stage": str(stage),
            "status": str(status) if status else None,
            "progress": progress,
            "duration_ms": duration_ms,
            "error_code": error_code,
            "error_message": error_message,
            "queue_size": queue_size,
            "segment_count": segment_count,
            "audio_duration_sec": audio_duration_sec,
            "persist_duration_ms": persist_duration_ms,
        }
        if extra_fields:
            payload.update(extra_fields)
        clean_payload = {key: value for key, value in payload.items() if value is not None}
        clean_payload.pop("event", None)
        log_event(
            self.logger,
            event=event,
            component="worker.pipeline",
            **clean_payload,
        )

