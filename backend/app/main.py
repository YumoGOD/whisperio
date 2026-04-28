import asyncio
import logging
import mimetypes
import os
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from app.db import get_connection, init_db
from app.models import JobStage, JobStatus
from app.queue_worker import TranscriptionWorker, utc_now_iso
from app.schemas import ErrorResponse, JobCreateResponse, JobDetail, JobListItem

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
    ".mp4",
    ".webm",
}
AUDIO_MEDIA_TYPES = {
    ".mp3": "audio/mpeg",
    ".wav": "audio/wav",
    ".m4a": "audio/mp4",
    ".ogg": "audio/ogg",
    ".flac": "audio/flac",
    ".aac": "audio/aac",
    ".mp4": "audio/mp4",
    ".webm": "audio/webm",
}

worker = TranscriptionWorker()
worker_tasks: list[asyncio.Task[None]] = []

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("whisperio.api")


def resolve_worker_count() -> int:
    raw = os.getenv("WHISPER_WORKERS", "1").strip()
    try:
        parsed = int(raw)
    except ValueError:
        logger.warning("event=invalid_worker_count raw_value=%s fallback=1", raw)
        return 1
    if parsed < 1:
        logger.warning("event=invalid_worker_count raw_value=%s fallback=1", raw)
        return 1
    return parsed


@asynccontextmanager
async def lifespan(_: FastAPI):
    global worker_tasks
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    init_db()
    recovered_jobs = await worker.recover_pending_jobs()
    if recovered_jobs:
        logger.info("event=pending_jobs_recovered count=%s", recovered_jobs)
    worker_count = resolve_worker_count()
    worker_tasks = [
        asyncio.create_task(worker.run(), name=f"transcription-worker-{idx + 1}")
        for idx in range(worker_count)
    ]
    logger.info("event=workers_started count=%s", worker_count)
    try:
        yield
    finally:
        await worker.shutdown()
        for task in worker_tasks:
            task.cancel()
        if worker_tasks:
            await asyncio.gather(*worker_tasks, return_exceptions=True)
        logger.info("event=workers_stopped count=%s", len(worker_tasks))
        worker_tasks = []


app = FastAPI(title="WhisperIO API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


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
async def handle_http_exception(_: Request, exc: HTTPException):
    code = "HTTP_ERROR"
    if isinstance(exc.detail, dict) and exc.detail.get("code"):
        code = str(exc.detail["code"])
    elif exc.status_code == 404:
        code = "NOT_FOUND"
    elif exc.status_code == 400:
        code = "BAD_REQUEST"
    elif exc.status_code == 413:
        code = "PAYLOAD_TOO_LARGE"
    payload = _build_error_payload(code=code, detail=exc.detail, fallback_message="Ошибка запроса")
    return JSONResponse(status_code=exc.status_code, content=payload)


@app.exception_handler(RequestValidationError)
async def handle_validation_exception(_: Request, exc: RequestValidationError):
    payload = {
        "code": "VALIDATION_ERROR",
        "message": "Ошибка валидации запроса",
        "details": exc.errors(),
    }
    return JSONResponse(status_code=422, content=payload)


@app.exception_handler(Exception)
async def handle_unexpected_exception(_: Request, exc: Exception):  # noqa: BLE001
    logger.exception("event=unhandled_exception error=%s", str(exc))
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


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/api/jobs", response_model=JobCreateResponse, responses={400: {"model": ErrorResponse}})
async def create_job(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail={"code": "FILE_NAME_REQUIRED", "message": "Имя файла обязательно"},
        )

    job_id = str(uuid4())
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
    if file.content_type and not file.content_type.startswith("audio/"):
        raise HTTPException(
            status_code=400,
            detail={
                "code": "UNSUPPORTED_MEDIA_TYPE",
                "message": "Ожидался аудиофайл",
                "details": [{"content_type": file.content_type}],
            },
        )
    stored_name = f"{job_id}{suffix}" if suffix else job_id
    stored_path = UPLOAD_DIR / stored_name
    total_size = 0
    has_content = False
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
                id, original_filename, stored_path, status, stage, progress,
                status_message, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                job_id,
                file.filename,
                str(stored_path),
                JobStatus.QUEUED,
                JobStage.QUEUED,
                0.0,
                "Ожидает обработки",
                created_at,
            ),
        )

    await worker.enqueue(job_id)
    logger.info("event=job_created job_id=%s status=%s stage=%s", job_id, JobStatus.QUEUED, JobStage.QUEUED)
    return {"job_id": job_id, "status": JobStatus.QUEUED}


@app.get("/api/jobs", response_model=list[JobListItem])
def list_jobs():
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT id, original_filename, status, stage, progress, status_message,
                   error, error_code, created_at, started_at, finished_at, duration_sec,
                   processing_duration_ms, transcribe_duration_ms, preprocess_duration_ms,
                   prepared_audio_path, quality_flags
            FROM transcription_jobs
            ORDER BY created_at DESC
            """
        ).fetchall()

    return [serialize_job_row(row) for row in rows]


@app.get("/api/jobs/{job_id}", response_model=JobDetail, responses={404: {"model": ErrorResponse}})
def get_job(job_id: str):
    with get_connection() as conn:
        row = conn.execute(
            """
            SELECT id, original_filename, status, stage, progress, status_message,
                   error, error_code, created_at, started_at, finished_at, duration_sec,
                   processing_duration_ms, transcribe_duration_ms, preprocess_duration_ms,
                   prepared_audio_path, quality_flags, stored_path
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

        segments = conn.execute(
            """
            SELECT idx, start_sec, end_sec, text
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
        }
        for seg in segments
    ]
    return payload


@app.get("/api/jobs/{job_id}/audio", responses={404: {"model": ErrorResponse}})
def get_job_audio(job_id: str):
    with get_connection() as conn:
        row = conn.execute(
            """
            SELECT stored_path, original_filename
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

    audio_path = Path(row["stored_path"])
    if not audio_path.exists() or not audio_path.is_file():
        raise HTTPException(
            status_code=404,
            detail={"code": "AUDIO_FILE_NOT_FOUND", "message": "Аудиофайл задачи не найден"},
        )

    media_type, _ = mimetypes.guess_type(str(audio_path))
    if not media_type:
        media_type = AUDIO_MEDIA_TYPES.get(audio_path.suffix.lower())
    return FileResponse(
        path=audio_path,
        media_type=media_type or "application/octet-stream",
        filename=row["original_filename"] or audio_path.name,
    )


def serialize_job_row(row) -> dict:
    return {
        "id": row["id"],
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
        "prepared_audio_path": row["prepared_audio_path"],
        "quality_flags": row["quality_flags"],
    }
