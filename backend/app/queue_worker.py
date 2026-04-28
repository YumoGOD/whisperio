import asyncio
import json
import logging
import os
import subprocess
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from faster_whisper import WhisperModel

from app.db import get_connection
from app.models import JobStage, JobStatus


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def getenv_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    return int(value)


def getenv_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    return float(value)


def getenv_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    return normalized in {"1", "true", "yes", "on"}


def clamp_progress(value: float | None) -> float | None:
    if value is None:
        return None
    return max(0.0, min(100.0, float(value)))


class AudioPreprocessError(RuntimeError):
    pass


class QualityGuardError(RuntimeError):
    pass


class TranscriptionWorker:
    def __init__(self) -> None:
        self.queue: asyncio.Queue[str] = asyncio.Queue()
        self._shutdown = asyncio.Event()
        self.logger = logging.getLogger("whisperio.worker")
        self.model_name = os.getenv("WHISPER_MODEL_SIZE", "medium")
        self.model_device = os.getenv("WHISPER_DEVICE", "cpu")
        self.model_compute_type = os.getenv("WHISPER_COMPUTE_TYPE", "int8")
        self.model: WhisperModel | None = None
        self._model_lock = asyncio.Lock()
        self.language = os.getenv("WHISPER_LANGUAGE", "ru")
        self.task = os.getenv("WHISPER_TASK", "transcribe")
        self.beam_size = getenv_int("WHISPER_BEAM_SIZE", 3)
        self.best_of = getenv_int("WHISPER_BEST_OF", 3)
        self.temperature = getenv_float("WHISPER_TEMPERATURE", 0.0)
        self.initial_prompt = os.getenv("WHISPER_INITIAL_PROMPT", "").strip() or None
        self.condition_on_previous_text = getenv_bool("WHISPER_CONDITION_ON_PREVIOUS_TEXT", False)
        self.vad_filter = getenv_bool("WHISPER_VAD_FILTER", True)
        self.enable_quality_fallback = getenv_bool("WHISPER_ENABLE_QUALITY_FALLBACK", True)
        self.min_unique_ratio = getenv_float("WHISPER_MIN_UNIQUE_SEGMENT_RATIO", 0.15)
        self.max_top_repeat_ratio = getenv_float("WHISPER_MAX_TOP_REPEAT_RATIO", 0.8)
        self.max_prompt_match_ratio = getenv_float("WHISPER_MAX_PROMPT_MATCH_RATIO", 0.5)
        self.preprocess_sample_rate = getenv_int("WHISPER_PREPROCESS_SAMPLE_RATE", 16000)
        self.preprocess_timeout_sec = getenv_int("WHISPER_PREPROCESS_TIMEOUT_SEC", 1800)
        self.keep_prepared_audio = getenv_bool("WHISPER_KEEP_PREPARED_AUDIO", False)

    async def enqueue(self, job_id: str) -> None:
        await self.queue.put(job_id)
        self._log_event(
            event="job_enqueued",
            job_id=job_id,
            stage=JobStage.QUEUED,
            status=JobStatus.QUEUED,
            progress=0.0,
            queue_size=self.queue.qsize(),
        )

    async def recover_pending_jobs(self) -> int:
        with get_connection() as conn:
            rows = conn.execute(
                """
                SELECT id, status, stage, progress, status_message
                FROM transcription_jobs
                WHERE status IN (?, ?)
                ORDER BY created_at ASC
                """,
                (JobStatus.QUEUED, JobStatus.PROCESSING),
            ).fetchall()

            if not rows:
                return 0

            job_ids = [row["id"] for row in rows]
            now_label = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            for row in rows:
                existing_message = (row["status_message"] or "").strip()
                restored_message = (
                    f"{existing_message}. Перезапуск сервиса, задача возвращена в очередь ({now_label})."
                    if existing_message
                    else f"Перезапуск сервиса, задача возвращена в очередь ({now_label})."
                )
                conn.execute(
                    """
                    UPDATE transcription_jobs
                    SET status = ?, stage = COALESCE(stage, ?), progress = COALESCE(progress, ?),
                        status_message = ?, started_at = NULL, finished_at = NULL,
                        duration_sec = NULL, processing_duration_ms = NULL, transcribe_duration_ms = NULL,
                        preprocess_duration_ms = NULL, prepared_audio_path = NULL, quality_flags = NULL,
                        error = NULL, error_code = NULL
                    WHERE id = ?
                    """,
                    (
                        JobStatus.QUEUED,
                        JobStage.QUEUED,
                        0.0,
                        restored_message,
                        row["id"],
                    ),
                )

        for job_id in job_ids:
            await self.enqueue(job_id)

        return len(job_ids)

    async def run(self) -> None:
        while not self._shutdown.is_set():
            try:
                job_id = await asyncio.wait_for(self.queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                continue

            self._log_event(
                event="job_dequeued",
                job_id=job_id,
                stage=JobStage.PREPARING,
                queue_size=self.queue.qsize(),
            )
            try:
                await self._process_job(job_id)
            finally:
                self.queue.task_done()

    async def shutdown(self) -> None:
        self._shutdown.set()

    async def _get_model(self) -> WhisperModel:
        if self.model is not None:
            return self.model
        async with self._model_lock:
            if self.model is None:
                self.model = await asyncio.to_thread(
                    WhisperModel,
                    self.model_name,
                    device=self.model_device,
                    compute_type=self.model_compute_type,
                )
        return self.model

    def _transcribe_audio(
        self,
        model: WhisperModel,
        audio_path: str,
        initial_prompt: str | None,
        condition_on_previous_text: bool,
    ):
        segments, info = model.transcribe(
            audio_path,
            language=self.language,
            task=self.task,
            beam_size=self.beam_size,
            best_of=self.best_of,
            temperature=self.temperature,
            vad_filter=self.vad_filter,
            condition_on_previous_text=condition_on_previous_text,
            initial_prompt=initial_prompt,
        )
        segment_rows = [
            (float(segment.start), float(segment.end), segment.text.strip())
            for segment in segments
        ]
        return segment_rows, info

    def _prepare_audio(self, audio_path: str) -> str:
        source_path = Path(audio_path)
        prepared_path = source_path.with_name(f"{source_path.stem}.prepared.wav")
        command = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(source_path),
            "-ac",
            "1",
            "-ar",
            str(self.preprocess_sample_rate),
            "-c:a",
            "pcm_s16le",
            str(prepared_path),
        ]
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=False,
                timeout=self.preprocess_timeout_sec,
            )
        except subprocess.TimeoutExpired as exc:
            raise AudioPreprocessError("ffmpeg preprocessing timed out") from exc
        if result.returncode != 0:
            details = (result.stderr or result.stdout or "ffmpeg failed").strip()
            raise AudioPreprocessError(f"ffmpeg preprocessing failed: {details}")
        if not prepared_path.exists() or prepared_path.stat().st_size == 0:
            raise AudioPreprocessError("ffmpeg preprocessing produced an empty output file")
        return str(prepared_path)

    def _cleanup_prepared_audio(self, prepared_audio_path: str | None) -> None:
        if self.keep_prepared_audio or not prepared_audio_path:
            return
        path = Path(prepared_audio_path)
        if path.exists():
            path.unlink(missing_ok=True)

    def _analyze_quality(
        self,
        segment_rows: list[tuple[float, float, str]],
        prompt: str | None,
    ) -> dict[str, object]:
        if not segment_rows:
            return {
                "segment_count": 0,
                "unique_text_count": 0,
                "unique_ratio": 0.0,
                "top_repeat_ratio": 0.0,
                "prompt_match_ratio": 0.0,
                "top_repeat_text": None,
                "reasons": ["NO_SEGMENTS"],
                "is_anomaly": True,
            }

        texts = [text.strip() for _, _, text in segment_rows if text and text.strip()]
        if not texts:
            return {
                "segment_count": len(segment_rows),
                "unique_text_count": 0,
                "unique_ratio": 0.0,
                "top_repeat_ratio": 0.0,
                "prompt_match_ratio": 0.0,
                "top_repeat_text": None,
                "reasons": ["EMPTY_SEGMENTS"],
                "is_anomaly": True,
            }

        counts = Counter(texts)
        top_text, top_count = counts.most_common(1)[0]
        segment_count = len(texts)
        unique_count = len(counts)
        unique_ratio = unique_count / segment_count
        top_repeat_ratio = top_count / segment_count
        prompt_match_ratio = 0.0
        normalized_prompt = prompt.strip() if prompt else None
        if normalized_prompt:
            prompt_hits = sum(1 for text in texts if text == normalized_prompt)
            prompt_match_ratio = prompt_hits / segment_count

        reasons: list[str] = []
        if unique_ratio < self.min_unique_ratio:
            reasons.append("LOW_UNIQUE_SEGMENT_RATIO")
        if top_repeat_ratio > self.max_top_repeat_ratio:
            reasons.append("HIGH_TOP_REPEAT_RATIO")
        if normalized_prompt and prompt_match_ratio > self.max_prompt_match_ratio:
            reasons.append("HIGH_PROMPT_MATCH_RATIO")

        return {
            "segment_count": segment_count,
            "unique_text_count": unique_count,
            "unique_ratio": round(unique_ratio, 4),
            "top_repeat_ratio": round(top_repeat_ratio, 4),
            "prompt_match_ratio": round(prompt_match_ratio, 4),
            "top_repeat_text": top_text,
            "reasons": reasons,
            "is_anomaly": bool(reasons),
        }

    async def _process_job(self, job_id: str) -> None:
        started_wall_time = time.perf_counter()
        prepared_audio_path: str | None = None
        preprocess_duration_ms: int | None = None
        quality_flags_json: str | None = None
        with get_connection() as conn:
            job = conn.execute(
                "SELECT stored_path FROM transcription_jobs WHERE id = ?",
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

        audio_path = job["stored_path"]
        self._log_event(
            event="job_started",
            job_id=job_id,
            stage=JobStage.PREPARING,
            status=JobStatus.PROCESSING,
            progress=5.0,
        )
        try:
            with get_connection() as conn:
                self._update_job(
                    conn=conn,
                    job_id=job_id,
                    status=JobStatus.PROCESSING,
                    stage=JobStage.PREPROCESSING,
                    progress=10.0,
                    status_message="Подготовка аудио (mono/16kHz PCM)",
                )
            preprocess_started = time.perf_counter()
            self._log_event(
                event="preprocess_started",
                job_id=job_id,
                stage=JobStage.PREPROCESSING,
                status=JobStatus.PROCESSING,
                progress=10.0,
            )
            prepared_audio_path = await asyncio.to_thread(self._prepare_audio, audio_path)
            preprocess_duration_ms = int((time.perf_counter() - preprocess_started) * 1000)
            with get_connection() as conn:
                self._update_job(
                    conn=conn,
                    job_id=job_id,
                    status=JobStatus.PROCESSING,
                    stage=JobStage.PREPROCESSING,
                    progress=14.0,
                    status_message="Аудио подготовлено",
                    preprocess_duration_ms=preprocess_duration_ms,
                    prepared_audio_path=prepared_audio_path if self.keep_prepared_audio else None,
                )
            self._log_event(
                event="preprocess_finished",
                job_id=job_id,
                stage=JobStage.PREPROCESSING,
                status=JobStatus.PROCESSING,
                progress=14.0,
                duration_ms=preprocess_duration_ms,
                extra_fields={"prepared_audio_path": prepared_audio_path},
            )

            with get_connection() as conn:
                self._update_job(
                    conn=conn,
                    job_id=job_id,
                    status=JobStatus.PROCESSING,
                    stage=JobStage.TRANSCRIBING,
                    progress=20.0,
                    status_message="Выполняется распознавание речи",
                )
            model = await self._get_model()
            transcribe_started = time.perf_counter()
            self._log_event(
                event="transcribe_started",
                job_id=job_id,
                stage=JobStage.TRANSCRIBING,
                status=JobStatus.PROCESSING,
                progress=20.0,
            )
            primary_rows, info = await asyncio.to_thread(
                self._transcribe_audio,
                model,
                prepared_audio_path or audio_path,
                self.initial_prompt,
                self.condition_on_previous_text,
            )
            quality_primary = self._analyze_quality(primary_rows, self.initial_prompt)
            fallback_used = False
            raw_segment_rows = primary_rows
            quality_payload: dict[str, object] = {"primary": quality_primary, "fallback_used": False}

            if bool(quality_primary["is_anomaly"]):
                self._log_event(
                    event="quality_anomaly_detected",
                    job_id=job_id,
                    stage=JobStage.TRANSCRIBING,
                    status=JobStatus.PROCESSING,
                    progress=70.0,
                    extra_fields={
                        "quality_reasons": ",".join(quality_primary["reasons"]),
                        "unique_ratio": quality_primary["unique_ratio"],
                        "top_repeat_ratio": quality_primary["top_repeat_ratio"],
                        "prompt_match_ratio": quality_primary["prompt_match_ratio"],
                    },
                )
                if not self.enable_quality_fallback:
                    raise QualityGuardError(
                        f"quality guard failed: {','.join(quality_primary['reasons'])}"
                    )
                fallback_used = True
                fallback_rows, info = await asyncio.to_thread(
                    self._transcribe_audio,
                    model,
                    prepared_audio_path or audio_path,
                    None,
                    False,
                )
                quality_fallback = self._analyze_quality(fallback_rows, None)
                quality_payload["fallback_used"] = True
                quality_payload["fallback"] = quality_fallback
                if bool(quality_fallback["is_anomaly"]):
                    raise QualityGuardError(
                        f"quality guard fallback failed: {','.join(quality_fallback['reasons'])}"
                    )
                raw_segment_rows = fallback_rows
                self._log_event(
                    event="quality_fallback_succeeded",
                    job_id=job_id,
                    stage=JobStage.TRANSCRIBING,
                    status=JobStatus.PROCESSING,
                    progress=75.0,
                    extra_fields={
                        "unique_ratio": quality_fallback["unique_ratio"],
                        "top_repeat_ratio": quality_fallback["top_repeat_ratio"],
                        "prompt_match_ratio": quality_fallback["prompt_match_ratio"],
                    },
                )

            transcribe_duration_ms = int((time.perf_counter() - transcribe_started) * 1000)
            quality_flags_json = json.dumps(quality_payload, ensure_ascii=False)
            segment_rows = [
                (job_id, idx, start_sec, end_sec, text)
                for idx, (start_sec, end_sec, text) in enumerate(raw_segment_rows)
            ]
            with get_connection() as conn:
                self._update_job(
                    conn=conn,
                    job_id=job_id,
                    status=JobStatus.PROCESSING,
                    stage=JobStage.SAVING_SEGMENTS,
                    progress=85.0,
                    status_message=f"Сохранение сегментов: {len(segment_rows)}",
                    transcribe_duration_ms=transcribe_duration_ms,
                    preprocess_duration_ms=preprocess_duration_ms,
                    prepared_audio_path=prepared_audio_path if self.keep_prepared_audio else None,
                    quality_flags=quality_flags_json,
                )
            self._log_event(
                event="transcribe_finished",
                job_id=job_id,
                stage=JobStage.TRANSCRIBING,
                status=JobStatus.PROCESSING,
                progress=80.0,
                duration_ms=transcribe_duration_ms,
                segment_count=len(segment_rows),
                audio_duration_sec=float(info.duration or 0.0),
                extra_fields={
                    "fallback_used": fallback_used,
                    "preprocess_duration_ms": preprocess_duration_ms,
                },
            )
            persist_started = time.perf_counter()

            with get_connection() as conn:
                conn.execute(
                    "DELETE FROM transcription_segments WHERE job_id = ?",
                    (job_id,),
                )
                if segment_rows:
                    conn.executemany(
                        """
                        INSERT INTO transcription_segments
                        (job_id, idx, start_sec, end_sec, text)
                        VALUES (?, ?, ?, ?, ?)
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
                    duration_sec=float(info.duration or 0.0),
                    processing_duration_ms=wall_duration_ms,
                    transcribe_duration_ms=transcribe_duration_ms,
                    preprocess_duration_ms=preprocess_duration_ms,
                    prepared_audio_path=prepared_audio_path if self.keep_prepared_audio else None,
                    quality_flags=quality_flags_json,
                    clear_error=True,
                )
            persist_duration_ms = int((time.perf_counter() - persist_started) * 1000)
            self._log_event(
                event="job_completed",
                job_id=job_id,
                stage=JobStage.COMPLETED,
                status=JobStatus.DONE,
                progress=100.0,
                duration_ms=wall_duration_ms,
                persist_duration_ms=persist_duration_ms,
                segment_count=len(segment_rows),
                extra_fields={"preprocess_duration_ms": preprocess_duration_ms},
            )
        except Exception as exc:  # noqa: BLE001
            wall_duration_ms = int((time.perf_counter() - started_wall_time) * 1000)
            with get_connection() as conn:
                status_message = "Ошибка обработки"
                if isinstance(exc, AudioPreprocessError):
                    status_message = "Ошибка подготовки аудио"
                elif isinstance(exc, QualityGuardError):
                    status_message = "Результат не прошел проверку качества"
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
                    prepared_audio_path=prepared_audio_path if self.keep_prepared_audio else None,
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
                error_message=str(exc),
                extra_fields={"preprocess_duration_ms": preprocess_duration_ms},
            )
        finally:
            self._cleanup_prepared_audio(prepared_audio_path)

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
        prepared_audio_path: str | None = None,
        quality_flags: str | None = None,
        clear_error: bool = False,
    ) -> None:
        fields: list[str] = []
        values: list[object] = []
        updates = {
            "status": status,
            "stage": stage,
            "progress": clamp_progress(progress),
            "status_message": status_message,
            "started_at": started_at,
            "finished_at": finished_at,
            "error": error,
            "error_code": error_code,
            "duration_sec": duration_sec,
            "processing_duration_ms": processing_duration_ms,
            "transcribe_duration_ms": transcribe_duration_ms,
            "preprocess_duration_ms": preprocess_duration_ms,
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
        if isinstance(exc, QualityGuardError):
            return "LOW_QUALITY_TRANSCRIPTION"
        error_type = type(exc).__name__.lower()
        message = str(exc).lower()
        if "cuda" in message or "out of memory" in message:
            return "MODEL_RUNTIME_ERROR"
        if "no such file" in message or "not found" in message:
            return "AUDIO_FILE_NOT_FOUND"
        if "sqlite" in error_type or "database" in message:
            return "DB_WRITE_FAILED"
        return "TRANSCRIPTION_FAILED"

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
            "progress": clamp_progress(progress),
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
        formatted = " ".join(
            f"{key}={value}"
            for key, value in payload.items()
            if value is not None
        )
        self.logger.info(formatted)
