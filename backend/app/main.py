"""FastAPI application: HTTP API.

Эндпоинты — см. docs/03_API_CONTRACT.md.
ML здесь не подключается — модели грузятся только в worker'е (Задача 5–6).
"""

import json
import logging
import shutil
from collections.abc import Generator
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

from fastapi import (
    APIRouter, Depends, FastAPI, File, Form, HTTPException, Query, Response,
    UploadFile,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.config import settings
from app.db import DB_URL, Base, SessionLocal, engine
from app.models import JOB_STATUSES, Job, WorkerHeartbeat
from app.schemas import (
    HealthResponse, JobCreateResponse, JobDetail, JobSummary, JobsList, Segment,
)
from app.storage import cleanup_job, get_input_path, get_result_dir

# Импорт пакета моделей нужен, чтобы Base.metadata знал о таблицах
# до вызова create_all. Сам объект не используется напрямую.
from app import models  # noqa: F401


logging.basicConfig(
    level=settings.LOG_LEVEL.upper(),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger(__name__)

# Доступные модели whisper — список из .env.example. Валидируем на входе,
# чтобы пользователь сразу получил 400, а не падение воркера через минуту.
ALLOWED_MODELS = frozenset({"large-v3", "large-v3-turbo", "medium", "small", "base"})
ALLOWED_FORMATS = ("txt", "srt", "json")
WORKER_ALIVE_WINDOW_SEC = 30


@asynccontextmanager
async def lifespan(_: FastAPI):
    from app.storage import ensure_dirs  # локальный импорт во избежание циклов
    ensure_dirs()
    Base.metadata.create_all(engine)
    log.info("storage ready; schema ensured at %s", DB_URL)
    yield


app = FastAPI(title="Whisper Transcription Service", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


router = APIRouter(prefix="/api")


# ============================ helpers ============================


def _save_upload(upload: UploadFile, dest: Path, max_bytes: int) -> int:
    """Потоково копирует UploadFile в dest, обрывая запись при превышении max_bytes."""
    total = 0
    chunk_size = 1024 * 1024
    try:
        with dest.open("wb") as out:
            while True:
                data = upload.file.read(chunk_size)
                if not data:
                    break
                total += len(data)
                if total > max_bytes:
                    out.close()
                    dest.unlink(missing_ok=True)
                    raise HTTPException(
                        status_code=400,
                        detail=f"file exceeds limit of {max_bytes // (1024 * 1024)} MB",
                    )
                out.write(data)
    finally:
        upload.file.close()
    return total


def _load_segments(job_id: str) -> list[Segment] | None:
    """Прочитать сегменты из result_dir/transcript.json, если он есть."""
    p = get_result_dir(job_id) / "transcript.json"
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        log.exception("failed to read transcript.json for job %s", job_id)
        return None
    raw = data.get("segments") if isinstance(data, dict) else data
    if not isinstance(raw, list):
        return None
    out: list[Segment] = []
    for seg in raw:
        try:
            out.append(Segment(start=float(seg["start"]), end=float(seg["end"]),
                               text=str(seg["text"])))
        except (KeyError, TypeError, ValueError):
            log.warning("skipping malformed segment in job %s: %r", job_id, seg)
    return out


def _as_aware_utc(dt: datetime) -> datetime:
    """SQLite не сохраняет tzinfo — приводим naive datetime к UTC."""
    return dt if dt.tzinfo is not None else dt.replace(tzinfo=timezone.utc)


# ============================ endpoints ============================


@router.post("/jobs", status_code=201, response_model=JobCreateResponse)
def create_job(
    file: UploadFile = File(...),
    language: str | None = Form(default=None),
    model: str | None = Form(default=None),
    diarize: bool = Form(default=False),  # ignored in v1
    db: Session = Depends(get_db),
) -> Job:
    del diarize

    if not file.filename:
        raise HTTPException(400, "filename is required")
    ext = Path(file.filename).suffix.lstrip(".").lower()
    if not ext:
        raise HTTPException(400, "file has no extension")
    if ext not in {e.lower() for e in settings.ALLOWED_EXTENSIONS}:
        raise HTTPException(400, f"unsupported extension: .{ext}")

    chosen_model = (model or settings.WHISPER_MODEL).strip()
    if chosen_model not in ALLOWED_MODELS:
        raise HTTPException(400, f"unknown model: {chosen_model}")

    lang = language.strip() if language else None
    if lang == "":
        lang = None

    job_id = str(uuid4())
    input_dir = get_input_path(job_id)
    input_dir.mkdir(parents=True, exist_ok=True)
    input_file = input_dir / f"input.{ext}"

    max_bytes = settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024
    try:
        file_size = _save_upload(file, input_file, max_bytes)
    except HTTPException:
        shutil.rmtree(input_dir, ignore_errors=True)
        raise
    except Exception as exc:
        shutil.rmtree(input_dir, ignore_errors=True)
        log.exception("failed to save upload for %s", file.filename)
        raise HTTPException(500, f"failed to save upload: {exc}") from exc

    job = Job(
        id=job_id,
        filename=file.filename,
        file_size=file_size,
        language=lang,
        model=chosen_model,
        status="queued",
        progress=0.0,
        input_path=str(input_file),
    )
    db.add(job)
    db.commit()
    db.refresh(job)
    log.info("job created: %s (%s, %d bytes, model=%s)",
             job.id, file.filename, file_size, chosen_model)
    return job


@router.get("/jobs", response_model=JobsList)
def list_jobs(
    limit: int = Query(default=100, ge=1, le=settings.MAX_JOBS_IN_LIST),
    status: str | None = Query(default=None),
    db: Session = Depends(get_db),
) -> JobsList:
    if status is not None and status not in JOB_STATUSES:
        raise HTTPException(400, f"unknown status: {status}")
    stmt = select(Job).order_by(Job.created_at.desc()).limit(limit)
    if status is not None:
        stmt = stmt.where(Job.status == status)
    rows = list(db.execute(stmt).scalars())
    return JobsList(jobs=[JobSummary.model_validate(j) for j in rows])


@router.get("/jobs/{job_id}", response_model=JobDetail)
def get_job(job_id: str, db: Session = Depends(get_db)) -> JobDetail:
    job = db.get(Job, job_id)
    if job is None:
        raise HTTPException(404, "job not found")
    segments = _load_segments(job_id) if job.status == "done" else None
    return JobDetail(
        id=job.id,
        filename=job.filename,
        status=job.status,
        progress=job.progress,
        duration_sec=job.duration_sec,
        model=job.model,
        language=job.language,
        detected_language=job.detected_language,
        transcript_text=job.transcript_text,
        segments=segments,
        error=job.error,
        created_at=job.created_at,
        started_at=job.started_at,
        finished_at=job.finished_at,
    )


@router.get("/jobs/{job_id}/download")
def download_job(
    job_id: str,
    format: str = Query(default="txt"),
    db: Session = Depends(get_db),
) -> FileResponse:
    if format not in ALLOWED_FORMATS:
        raise HTTPException(400, f"unknown format: {format} (allowed: {ALLOWED_FORMATS})")
    job = db.get(Job, job_id)
    if job is None or job.status != "done":
        raise HTTPException(404, "job not found or not done")
    path = get_result_dir(job_id) / f"transcript.{format}"
    if not path.exists():
        raise HTTPException(404, "result file not found")

    media_type = {
        "txt": "text/plain; charset=utf-8",
        "srt": "application/x-subrip",
        "json": "application/json",
    }[format]
    stem = Path(job.filename).stem or "transcript"
    return FileResponse(path, media_type=media_type, filename=f"{stem}.{format}")


@router.delete("/jobs/{job_id}", status_code=204)
def delete_job(job_id: str, db: Session = Depends(get_db)) -> Response:
    job = db.get(Job, job_id)
    if job is None:
        raise HTTPException(404, "job not found")
    cleanup_job(job_id)
    db.delete(job)
    db.commit()
    log.info("job deleted: %s", job_id)
    return Response(status_code=204)


@router.get("/health", response_model=HealthResponse)
def health(db: Session = Depends(get_db)) -> HealthResponse:
    hb = db.get(WorkerHeartbeat, 1)
    worker_alive = False
    gpu_available = False
    if hb is not None:
        age = datetime.now(timezone.utc) - _as_aware_utc(hb.last_seen)
        worker_alive = age <= timedelta(seconds=WORKER_ALIVE_WINDOW_SEC)
        # Доверяем gpu_available только если воркер жив (stale значение бесполезно).
        if worker_alive:
            gpu_available = bool(hb.gpu_available)
    queue_size = db.execute(
        select(func.count()).select_from(Job).where(Job.status == "queued")
    ).scalar_one()
    return HealthResponse(
        status="ok",
        worker_alive=worker_alive,
        gpu_available=gpu_available,
        queue_size=int(queue_size),
    )


app.include_router(router)
