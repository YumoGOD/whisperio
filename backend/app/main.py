import logging
import mimetypes
import os
import statistics
import time
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from app.db import get_connection, init_db, requeue_unfinished_jobs
from app.logging_utils import bind_log_context, configure_logging, log_event, reset_log_context
from app.models import JobStage, JobStatus
from app.transcription_worker import utc_now_iso
from app.runtime import TranscriptionRuntime
from app.schemas import (
    ErrorResponse,
    JobCreateResponse,
    JobDeleteResponse,
    JobDetail,
    JobListItem,
    JobStatsResponse,
    SpeedBreakdown,
    SpeedExtreme,
)

UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", "data/uploads"))
MAX_UPLOAD_SIZE_MB = int(os.getenv("MAX_UPLOAD_SIZE_MB", "200"))
MAX_UPLOAD_SIZE_BYTES = MAX_UPLOAD_SIZE_MB * 1024 * 1024
UPLOAD_CHUNK_SIZE = 1024 * 1024
ALLOWED_AUDIO_EXTENSIONS = {
    ".mp3",
    ".wav",
    ".m4a",
    ".ogg",
    ".flac",
    ".aac",
    ".avi",
    ".mp4",
    ".mkv",
    ".mov",
    ".webm",
}
AUDIO_CONTAINER_EXTENSIONS = {".avi", ".mp4", ".mkv", ".mov", ".webm"}
AUDIO_MEDIA_TYPES = {
    ".mp3": "audio/mpeg",
    ".wav": "audio/wav",
    ".m4a": "audio/mp4",
    ".ogg": "audio/ogg",
    ".flac": "audio/flac",
    ".aac": "audio/aac",
    ".avi": "video/x-msvideo",
    ".mp4": "audio/mp4",
    ".mkv": "video/x-matroska",
    ".mov": "video/quicktime",
    ".webm": "audio/webm",
}

runtime = TranscriptionRuntime()

configure_logging()
logger = logging.getLogger("whisperio.api")


@asynccontextmanager
async def lifespan(_: FastAPI):
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    init_db()
    recovered_jobs = requeue_unfinished_jobs()
    if recovered_jobs:
        log_event(
            logger,
            event="pending_jobs_recovered",
            component="api.lifecycle",
            count=recovered_jobs,
        )
    await runtime.start()
    try:
        yield
    finally:
        await runtime.stop()


app = FastAPI(title="WhisperIO API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def request_context_middleware(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID") or str(uuid4())
    request.state.request_id = request_id
    token = bind_log_context(request_id=request_id)
    started = time.perf_counter()
    log_event(
        logger,
        event="api_request_started",
        component="api.http",
        request_id=request_id,
        method=request.method,
        path=request.url.path,
        client_ip=request.client.host if request.client else None,
        user_agent=request.headers.get("user-agent"),
    )
    try:
        response = await call_next(request)
    except Exception:
        duration_ms = int((time.perf_counter() - started) * 1000)
        log_event(
            logger,
            event="api_request_finished",
            level=logging.ERROR,
            component="api.http",
            request_id=request_id,
            method=request.method,
            path=request.url.path,
            status_code=500,
            duration_ms=duration_ms,
        )
        reset_log_context(token)
        raise
    duration_ms = int((time.perf_counter() - started) * 1000)
    response.headers["X-Request-ID"] = request_id
    log_event(
        logger,
        event="api_request_finished",
        component="api.http",
        request_id=request_id,
        method=request.method,
        path=request.url.path,
        status_code=response.status_code,
        duration_ms=duration_ms,
    )
    reset_log_context(token)
    return response


def _extract_error_message(detail: object) -> str:
    if isinstance(detail, str):
        return detail
    if isinstance(detail, list) and detail:
        first = detail[0]
        if isinstance(first, dict):
            return str(first.get("msg") or "Ошибка валидации запроса")
        return str(first)
    if isinstance(detail, dict):
        message = detail.get("message")
        if message:
            return str(message)
    return "Внутренняя ошибка сервера"


def _extract_error_details(detail: object) -> list[object] | None:
    if isinstance(detail, list):
        return detail
    if isinstance(detail, dict):
        nested = detail.get("details")
        if isinstance(nested, list):
            return nested
        return [detail]
    if isinstance(detail, str):
        return [detail]
    return None


def _build_error_payload(
    *, code: str, detail: object, fallback_message: str
) -> dict[str, object]:
    if isinstance(detail, dict) and "code" in detail and "message" in detail:
        return {
            "code": str(detail["code"]),
            "message": str(detail["message"]),
            "details": _extract_error_details(detail),
        }
    message = _extract_error_message(detail) if detail is not None else fallback_message
    return {
        "code": code,
        "message": message,
        "details": _extract_error_details(detail),
    }


@app.exception_handler(HTTPException)
async def handle_http_exception(request: Request, exc: HTTPException):
    code = "HTTP_ERROR"
    if isinstance(exc.detail, dict) and exc.detail.get("code"):
        code = str(exc.detail["code"])
    elif exc.status_code == 404:
        code = "NOT_FOUND"
    elif exc.status_code == 400:
        code = "BAD_REQUEST"
    elif exc.status_code == 413:
        code = "PAYLOAD_TOO_LARGE"
    log_event(
        logger,
        event="http_exception",
        level=logging.WARNING if exc.status_code < 500 else logging.ERROR,
        component="api.http",
        request_id=getattr(request.state, "request_id", None),
        method=request.method,
        path=request.url.path,
        status_code=exc.status_code,
        error_code=code,
    )
    payload = _build_error_payload(code=code, detail=exc.detail, fallback_message="Ошибка запроса")
    return JSONResponse(status_code=exc.status_code, content=payload)


@app.exception_handler(RequestValidationError)
async def handle_validation_exception(request: Request, exc: RequestValidationError):
    errors = exc.errors()
    log_event(
        logger,
        event="request_validation_failed",
        level=logging.WARNING,
        component="api.http",
        request_id=getattr(request.state, "request_id", None),
        method=request.method,
        path=request.url.path,
        status_code=422,
        validation_errors_count=len(errors),
    )
    payload = {
        "code": "VALIDATION_ERROR",
        "message": "Ошибка валидации запроса",
        "details": errors,
    }
    return JSONResponse(status_code=422, content=payload)


@app.exception_handler(Exception)
async def handle_unexpected_exception(request: Request, exc: Exception):  # noqa: BLE001
    log_event(
        logger,
        event="unhandled_exception",
        level=logging.ERROR,
        component="api.http",
        request_id=getattr(request.state, "request_id", None),
        method=request.method,
        path=request.url.path,
        error_type=type(exc).__name__,
        error_message=str(exc),
    )
    payload = {
        "code": "INTERNAL_SERVER_ERROR",
        "message": "Внутренняя ошибка сервера",
        "details": None,
    }
    return JSONResponse(status_code=500, content=payload)


def parse_dt(value: str | None) -> datetime | None:
    if not value:
        return None
    return datetime.fromisoformat(value)


def _safe_unlink(path_value: str | None) -> None:
    if not path_value:
        return
    path = Path(path_value)
    if path.exists() and path.is_file():
        path.unlink(missing_ok=True)


def _calc_hourly_ms(duration_sec: float | None, metric_ms: int | None) -> float | None:
    if duration_sec is None or metric_ms is None:
        return None
    if duration_sec <= 0:
        return None
    return (float(metric_ms) * 3600.0) / float(duration_sec)


def _avg_or_none(values: list[float]) -> float | None:
    if not values:
        return None
    return float(round(statistics.fmean(values), 2))


def _audio_url(job_id: str, variant: str) -> str:
    return f"/api/jobs/{job_id}/audio?variant={variant}"


@app.get("/health")
def health(request: Request):
    log_event(
        logger,
        event="health_check_ok",
        component="api.health",
        request_id=getattr(request.state, "request_id", None),
    )
    return {"status": "ok"}


@app.post("/api/jobs", response_model=JobCreateResponse, responses={400: {"model": ErrorResponse}})
async def create_job(request: Request, file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail={"code": "FILE_NAME_REQUIRED", "message": "Имя файла обязательно"},
        )

    job_id = str(uuid4())
    request_id = getattr(request.state, "request_id", None)
    suffix = Path(file.filename).suffix.lower()
    if suffix and suffix not in ALLOWED_AUDIO_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail={
                "code": "UNSUPPORTED_AUDIO_TYPE",
                "message": "Неподдерживаемый формат файла",
                "details": [{"extension": suffix, "allowed_extensions": sorted(ALLOWED_AUDIO_EXTENSIONS)}],
            },
        )
    if file.content_type:
        is_audio_mime = file.content_type.startswith("audio/")
        is_video_container_mime = file.content_type.startswith("video/") and suffix in AUDIO_CONTAINER_EXTENSIONS
        if not (is_audio_mime or is_video_container_mime):
            raise HTTPException(
                status_code=400,
                detail={
                    "code": "UNSUPPORTED_MEDIA_TYPE",
                    "message": "Ожидался аудиофайл или видео-контейнер с аудиодорожкой",
                    "details": [{"content_type": file.content_type}],
                },
            )
    stored_name = f"{job_id}{suffix}" if suffix else job_id
    stored_path = UPLOAD_DIR / stored_name
    total_size = 0
    has_content = False
    upload_started = time.perf_counter()
    try:
        with stored_path.open("wb") as destination:
            while True:
                chunk = await file.read(UPLOAD_CHUNK_SIZE)
                if not chunk:
                    break
                has_content = True
                total_size += len(chunk)
                if total_size > MAX_UPLOAD_SIZE_BYTES:
                    destination.close()
                    stored_path.unlink(missing_ok=True)
                    raise HTTPException(
                        status_code=413,
                        detail={
                            "code": "AUDIO_FILE_TOO_LARGE",
                            "message": f"Файл превышает лимит {MAX_UPLOAD_SIZE_MB} МБ",
                            "details": [{"max_size_mb": MAX_UPLOAD_SIZE_MB}],
                        },
                    )
                destination.write(chunk)
        if not has_content:
            stored_path.unlink(missing_ok=True)
            raise HTTPException(
                status_code=400,
                detail={"code": "EMPTY_AUDIO_FILE", "message": "Аудиофайл пуст"},
            )
    finally:
        await file.close()

    created_at = utc_now_iso()
    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO transcription_jobs (
                id, request_id, original_filename, stored_path, status, stage, progress,
                status_message, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                job_id,
                request_id,
                file.filename,
                str(stored_path),
                JobStatus.QUEUED,
                JobStage.QUEUED,
                0.0,
                "Ожидает обработки",
                created_at,
            ),
        )

    upload_duration_ms = int((time.perf_counter() - upload_started) * 1000)
    log_event(
        logger,
        event="job_created",
        component="api.jobs",
        job_id=job_id,
        request_id=request_id,
        status=JobStatus.QUEUED,
        stage=JobStage.QUEUED,
        upload_duration_ms=upload_duration_ms,
        upload_size_bytes=total_size,
        file_extension=suffix or None,
    )
    return {"job_id": job_id, "status": JobStatus.QUEUED}


@app.get("/api/jobs", response_model=list[JobListItem])
def list_jobs(request: Request):
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT id, original_filename, status, stage, progress, status_message,
                   error, error_code, created_at, started_at, finished_at, duration_sec,
                   processing_duration_ms, transcribe_duration_ms, preprocess_duration_ms,
                   prepared_audio_path, quality_flags, delete_requested
            FROM transcription_jobs
            ORDER BY created_at DESC
            """
        ).fetchall()
    log_event(
        logger,
        event="jobs_listed",
        component="api.jobs",
        request_id=getattr(request.state, "request_id", None),
        jobs_count=len(rows),
    )
    return [serialize_job_row(row) for row in rows]


@app.get("/api/jobs/{job_id}", response_model=JobDetail, responses={404: {"model": ErrorResponse}})
def get_job(job_id: str, request: Request):
    with get_connection() as conn:
        row = conn.execute(
            """
            SELECT id, original_filename, status, stage, progress, status_message,
                   error, error_code, created_at, started_at, finished_at, duration_sec,
                   processing_duration_ms, transcribe_duration_ms, preprocess_duration_ms,
                   prepared_audio_path, quality_flags, stored_path, delete_requested
            FROM transcription_jobs
            WHERE id = ?
            """,
            (job_id,),
        ).fetchone()

        if not row:
            log_event(
                logger,
                event="job_detail_not_found",
                level=logging.WARNING,
                component="api.jobs",
                request_id=getattr(request.state, "request_id", None),
                job_id=job_id,
            )
            raise HTTPException(
                status_code=404,
                detail={"code": "JOB_NOT_FOUND", "message": "Задача не найдена"},
            )

        segments = conn.execute(
            """
            SELECT idx, start_sec, end_sec, text, label, confidence, quality_flags
            FROM transcription_segments
            WHERE job_id = ?
            ORDER BY idx ASC
            """,
            (job_id,),
        ).fetchall()

    payload = serialize_job_row(row)
    payload["segments"] = [
        {
            "idx": seg["idx"],
            "start_sec": seg["start_sec"],
            "end_sec": seg["end_sec"],
            "text": seg["text"],
            "label": seg["label"],
            "confidence": seg["confidence"],
            "quality_flags": seg["quality_flags"],
        }
        for seg in segments
    ]
    log_event(
        logger,
        event="job_detail_fetched",
        component="api.jobs",
        request_id=getattr(request.state, "request_id", None),
        job_id=job_id,
        segment_count=len(segments),
        status=payload.get("status"),
        stage=payload.get("stage"),
    )
    return payload


@app.delete("/api/jobs/{job_id}", response_model=JobDeleteResponse, responses={404: {"model": ErrorResponse}})
def delete_job(job_id: str, request: Request):
    request_id = getattr(request.state, "request_id", None)
    with get_connection() as conn:
        row = conn.execute(
            """
            SELECT status, stored_path, prepared_audio_path
            FROM transcription_jobs
            WHERE id = ?
            """,
            (job_id,),
        ).fetchone()
        if not row:
            raise HTTPException(
                status_code=404,
                detail={"code": "JOB_NOT_FOUND", "message": "Задача не найдена"},
            )

        if row["status"] == JobStatus.PROCESSING:
            conn.execute(
                """
                UPDATE transcription_jobs
                SET delete_requested = 1,
                    delete_requested_at = ?,
                    status_message = ?
                WHERE id = ?
                """,
                (
                    utc_now_iso(),
                    "Удаление запрошено. Ожидается завершение текущего шага.",
                    job_id,
                ),
            )
            log_event(
                logger,
                event="job_delete_requested",
                component="api.jobs",
                request_id=request_id,
                job_id=job_id,
            )
            return {
                "job_id": job_id,
                "status": "pending_delete",
                "message": "Заявка будет удалена после завершения текущего шага обработки.",
            }

        conn.execute("DELETE FROM transcription_jobs WHERE id = ?", (job_id,))
        _safe_unlink(row["stored_path"])
        _safe_unlink(row["prepared_audio_path"])

    log_event(
        logger,
        event="job_deleted",
        component="api.jobs",
        request_id=request_id,
        job_id=job_id,
    )
    return {
        "job_id": job_id,
        "status": "deleted",
        "message": "Заявка удалена.",
    }


@app.get("/api/jobs/{job_id}/audio", responses={404: {"model": ErrorResponse}})
def get_job_audio(
    job_id: str,
    request: Request,
    variant: str = Query(default="original", pattern="^(original|prepared)$"),
):
    with get_connection() as conn:
        row = conn.execute(
            """
            SELECT stored_path, prepared_audio_path, original_filename
            FROM transcription_jobs
            WHERE id = ?
            """,
            (job_id,),
        ).fetchone()
    if not row:
        log_event(
            logger,
            event="job_audio_not_found",
            level=logging.WARNING,
            component="api.jobs",
            request_id=getattr(request.state, "request_id", None),
            job_id=job_id,
            reason_code="job_not_found",
        )
        raise HTTPException(
            status_code=404,
            detail={"code": "JOB_NOT_FOUND", "message": "Задача не найдена"},
        )

    raw_path = row["prepared_audio_path"] if variant == "prepared" else row["stored_path"]
    if variant == "prepared" and not raw_path:
        raise HTTPException(
            status_code=404,
            detail={"code": "PREPARED_AUDIO_NOT_FOUND", "message": "Обработанное аудио недоступно"},
        )

    audio_path = Path(raw_path)
    if not audio_path.exists() or not audio_path.is_file():
        log_event(
            logger,
            event="job_audio_not_found",
            level=logging.WARNING,
            component="api.jobs",
            request_id=getattr(request.state, "request_id", None),
            job_id=job_id,
            reason_code="audio_file_missing",
        )
        raise HTTPException(
            status_code=404,
            detail={"code": "AUDIO_FILE_NOT_FOUND", "message": "Аудиофайл задачи не найден"},
        )

    media_type, _ = mimetypes.guess_type(str(audio_path))
    if not media_type:
        media_type = AUDIO_MEDIA_TYPES.get(audio_path.suffix.lower())
    log_event(
        logger,
        event="job_audio_served",
        component="api.jobs",
        request_id=getattr(request.state, "request_id", None),
        job_id=job_id,
        variant=variant,
        media_type=media_type or "application/octet-stream",
        filename=row["original_filename"] or audio_path.name,
    )
    return FileResponse(
        path=audio_path,
        media_type=media_type or "application/octet-stream",
        filename=row["original_filename"] or audio_path.name,
    )


@app.get("/api/stats", response_model=JobStatsResponse)
def get_stats(
    request: Request,
    from_: datetime | None = Query(default=None, alias="from"),
    to: datetime | None = Query(default=None),
):
    if from_ is not None and to is not None and from_ > to:
        raise HTTPException(
            status_code=400,
            detail={"code": "INVALID_DATE_RANGE", "message": "Параметр from не может быть больше to"},
        )
    where_clauses = ["status = ?"]
    params: list[object] = [JobStatus.DONE]
    if from_ is not None:
        where_clauses.append("finished_at >= ?")
        params.append(from_.isoformat())
    if to is not None:
        where_clauses.append("finished_at <= ?")
        params.append(to.isoformat())

    with get_connection() as conn:
        rows = conn.execute(
            f"""
            SELECT id, original_filename, duration_sec, processing_duration_ms,
                   preprocess_duration_ms, transcribe_duration_ms
            FROM transcription_jobs
            WHERE {' AND '.join(where_clauses)}
            ORDER BY finished_at DESC
            """,
            params,
        ).fetchall()

    enriched: list[dict[str, object]] = []
    for row in rows:
        total_ms_hour = _calc_hourly_ms(row["duration_sec"], row["processing_duration_ms"])
        preprocess_ms_hour = _calc_hourly_ms(row["duration_sec"], row["preprocess_duration_ms"])
        transcribe_ms_hour = _calc_hourly_ms(row["duration_sec"], row["transcribe_duration_ms"])
        if total_ms_hour is None:
            continue
        enriched.append(
            {
                "job_id": row["id"],
                "original_filename": row["original_filename"],
                "total_ms": total_ms_hour,
                "preprocess_ms": preprocess_ms_hour,
                "transcribe_ms": transcribe_ms_hour,
            }
        )

    def to_breakdown(item: dict[str, object] | None) -> SpeedBreakdown:
        if not item:
            return SpeedBreakdown()
        return SpeedBreakdown(
            total_ms=round(float(item["total_ms"]), 2),
            preprocess_ms=(
                round(float(item["preprocess_ms"]), 2) if item.get("preprocess_ms") is not None else None
            ),
            transcribe_ms=(
                round(float(item["transcribe_ms"]), 2) if item.get("transcribe_ms") is not None else None
            ),
        )

    if enriched:
        fastest = min(enriched, key=lambda item: float(item["total_ms"]))
        slowest = max(enriched, key=lambda item: float(item["total_ms"]))
        avg = {
            "total_ms": _avg_or_none([float(item["total_ms"]) for item in enriched]),
            "preprocess_ms": _avg_or_none(
                [float(item["preprocess_ms"]) for item in enriched if item.get("preprocess_ms") is not None]
            ),
            "transcribe_ms": _avg_or_none(
                [float(item["transcribe_ms"]) for item in enriched if item.get("transcribe_ms") is not None]
            ),
        }
    else:
        fastest = None
        slowest = None
        avg = {"total_ms": None, "preprocess_ms": None, "transcribe_ms": None}

    payload = JobStatsResponse(
        range_from=from_,
        range_to=to,
        completed_jobs=len(enriched),
        average=to_breakdown(avg),
        fastest=SpeedExtreme(
            job_id=fastest["job_id"] if fastest else None,
            original_filename=fastest["original_filename"] if fastest else None,
            **to_breakdown(fastest).model_dump(),
        ),
        slowest=SpeedExtreme(
            job_id=slowest["job_id"] if slowest else None,
            original_filename=slowest["original_filename"] if slowest else None,
            **to_breakdown(slowest).model_dump(),
        ),
    )
    log_event(
        logger,
        event="stats_fetched",
        component="api.jobs",
        request_id=getattr(request.state, "request_id", None),
        completed_jobs=payload.completed_jobs,
    )
    return payload


def serialize_job_row(row) -> dict:
    job_id = row["id"]
    has_prepared_audio = bool(row["prepared_audio_path"])
    return {
        "id": job_id,
        "original_filename": row["original_filename"],
        "status": row["status"],
        "stage": row["stage"],
        "progress": row["progress"],
        "status_message": row["status_message"],
        "error": row["error"],
        "error_code": row["error_code"],
        "created_at": parse_dt(row["created_at"]),
        "started_at": parse_dt(row["started_at"]),
        "finished_at": parse_dt(row["finished_at"]),
        "duration_sec": row["duration_sec"],
        "processing_duration_ms": row["processing_duration_ms"],
        "transcribe_duration_ms": row["transcribe_duration_ms"],
        "preprocess_duration_ms": row["preprocess_duration_ms"],
        "original_audio_url": _audio_url(job_id, "original"),
        "prepared_audio_url": _audio_url(job_id, "prepared") if has_prepared_audio else None,
        "delete_requested": bool(row["delete_requested"]),
        "quality_flags": row["quality_flags"],
    }
