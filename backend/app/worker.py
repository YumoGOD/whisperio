"""Worker-процесс: опрашивает БД, обрабатывает задачи транскрипции.

Запуск:
    cd backend
    python -m app.worker

Один процесс на одну GPU. Никаких очередей-брокеров: БД — единственный источник истины.
"""

from __future__ import annotations

import logging
import signal
import sys
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import settings
from app.db import Base, DB_URL, SessionLocal, engine
from app.models import Job, WorkerHeartbeat
from app.pipeline.audio import probe_duration, to_wav_16k_mono
from app.pipeline.formats import to_json, to_srt, to_txt
from app.pipeline.transcribe import TranscriptionResult, transcriber
from app.pipeline.vad import split_into_segments
from app.storage import ensure_dirs, get_input_path, get_result_dir

# Импорт пакета моделей: чтобы Base.metadata знал о таблицах перед create_all.
from app import models  # noqa: F401

log = logging.getLogger("app.worker")

ACTIVE_STATUSES = ("preprocessing", "transcribing")
# Сколько единиц прогресса пропускать между коммитами в БД (rate limit).
PROGRESS_COMMIT_STEP = 0.02


# ============================ utils ============================


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _as_aware_utc(dt: datetime | None) -> datetime | None:
    if dt is None:
        return None
    return dt if dt.tzinfo is not None else dt.replace(tzinfo=timezone.utc)


def _gpu_available() -> bool:
    try:
        import torch  # noqa: PLC0415
        return bool(torch.cuda.is_available())
    except Exception:  # noqa: BLE001
        return False


def _gpu_name() -> str | None:
    try:
        import torch  # noqa: PLC0415
        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
    except Exception:  # noqa: BLE001
        pass
    return None


def _setup_logging() -> None:
    logging.basicConfig(
        level=settings.LOG_LEVEL.upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


# ============================ DB ops ============================


def heartbeat(gpu_avail: bool) -> None:
    """Атомарно обновить worker_heartbeat (id=1)."""
    with SessionLocal() as db:
        hb = db.get(WorkerHeartbeat, 1)
        now = _utcnow()
        if hb is None:
            db.add(WorkerHeartbeat(id=1, last_seen=now, gpu_available=gpu_avail))
        else:
            hb.last_seen = now
            hb.gpu_available = gpu_avail
        db.commit()


def recover_stale_jobs() -> int:
    """Задачи в preprocessing/transcribing старше STALE_JOB_TIMEOUT_MIN → queued."""
    threshold = _utcnow() - timedelta(minutes=settings.STALE_JOB_TIMEOUT_MIN)
    recovered = 0
    with SessionLocal() as db:
        stale = list(
            db.execute(
                select(Job).where(
                    Job.status.in_(ACTIVE_STATUSES),
                    Job.started_at.is_not(None),
                )
            ).scalars()
        )
        for j in stale:
            started = _as_aware_utc(j.started_at)
            if started is None or started >= threshold:
                continue
            log.warning("recovering stale job %s (status=%s, started %s)",
                        j.id, j.status, started)
            j.status = "queued"
            j.started_at = None
            j.progress = 0.0
            j.error = None
            recovered += 1
        if recovered:
            db.commit()
    if recovered:
        log.info("recovered %d stale job(s) to queued", recovered)
    return recovered


def pick_queued_job(db: Session) -> Job | None:
    """Захватить одну queued-задачу, перевести в preprocessing.

    Атомарность здесь нестрогая (SQLite, один воркер). Коммитим сразу,
    чтобы API увидел status=preprocessing.
    """
    job = db.execute(
        select(Job).where(Job.status == "queued").order_by(Job.created_at).limit(1)
    ).scalar_one_or_none()
    if job is None:
        return None
    job.status = "preprocessing"
    job.started_at = _utcnow()
    job.progress = 0.0
    job.error = None
    db.commit()
    db.refresh(job)
    return job


# ============================ processing ============================


def _write_results(job: Job, result: TranscriptionResult) -> Path:
    result_dir = get_result_dir(job.id)
    result_dir.mkdir(parents=True, exist_ok=True)

    (result_dir / "transcript.txt").write_text(to_txt(result.segments), encoding="utf-8")
    (result_dir / "transcript.srt").write_text(to_srt(result.segments), encoding="utf-8")

    meta: dict[str, Any] = {
        "filename": job.filename,
        "model": job.model,
        "language": job.language,
        "detected_language": result.detected_language,
        "duration_sec": job.duration_sec,
    }
    (result_dir / "transcript.json").write_text(
        to_json(result.segments, meta), encoding="utf-8"
    )
    return result_dir


def _process(db: Session, job: Job) -> None:
    """Полный pipeline для одной задачи. Любая ошибка — выше по стеку."""
    input_path = Path(job.input_path)
    if not input_path.exists():
        raise RuntimeError(f"input file missing: {input_path}")

    # --- preprocessing ---
    log.info("[%s] preprocessing %s", job.id, input_path.name)
    wav_path = get_input_path(job.id) / "converted_16k_mono.wav"
    to_wav_16k_mono(input_path, wav_path)
    duration = probe_duration(wav_path)
    job.duration_sec = duration
    job.progress = 0.10
    db.commit()

    vad_segs = split_into_segments(wav_path)
    log.info("[%s] VAD: %d segments, duration=%.2fs", job.id, len(vad_segs), duration)
    job.progress = 0.20
    db.commit()

    # --- transcribing ---
    job.status = "transcribing"
    job.progress = 0.25
    db.commit()

    last_committed = {"p": 0.25}

    def on_progress(p_within: float) -> None:
        # 0.25 → 0.95 коридор. Коммитим только при заметном изменении.
        target = 0.25 + 0.70 * max(0.0, min(1.0, p_within))
        if target - last_committed["p"] >= PROGRESS_COMMIT_STEP:
            job.progress = round(target, 4)
            db.commit()
            last_committed["p"] = target

    result = transcriber.transcribe_segments(
        wav_path,
        vad_segs,
        language=job.language,
        on_progress=on_progress,
        total_duration=duration,
    )
    log.info("[%s] transcribed: %d segments, lang=%s",
             job.id, len(result.segments), result.detected_language)

    # --- finalize ---
    job.progress = 0.95
    db.commit()

    result_dir = _write_results(job, result)
    job.result_dir = str(result_dir)
    job.transcript_text = "\n".join(s.text for s in result.segments)
    job.detected_language = result.detected_language
    job.status = "done"
    job.progress = 1.0
    job.finished_at = _utcnow()
    db.commit()
    log.info("[%s] done in result_dir=%s", job.id, result_dir)


def process_one_job(db: Session, job: Job) -> None:
    """Обработать одну задачу. Любое исключение → status=failed, не падаем дальше."""
    try:
        _process(db, job)
    except Exception as e:  # noqa: BLE001
        log.exception("[%s] job failed", job.id)
        try:
            db.rollback()
            # Перечитываем job (rollback мог отвязать его).
            j2 = db.get(Job, job.id)
            target = j2 if j2 is not None else job
            target.status = "failed"
            target.error = str(e)[:1000]
            target.finished_at = _utcnow()
            db.commit()
        except Exception:  # noqa: BLE001
            log.exception("[%s] failed to mark job as failed", job.id)


# ============================ main loop ============================


def _on_signal(stop: threading.Event, signum: int) -> None:
    log.info("signal %s received; stopping worker after current job", signum)
    stop.set()


def main() -> int:
    _setup_logging()
    log.info("=== whisper worker starting ===")
    log.info("DB: %s", DB_URL)
    log.info("storage: %s", settings.STORAGE_DIR)
    log.info("model: %s (device=%s, compute=%s, batch=%d, beam=%d)",
             settings.WHISPER_MODEL, settings.WHISPER_DEVICE,
             settings.WHISPER_COMPUTE_TYPE, settings.WHISPER_BATCH_SIZE,
             settings.BEAM_SIZE)

    gpu_avail = _gpu_available()
    gpu_name = _gpu_name()
    log.info("GPU available: %s%s", gpu_avail, f" ({gpu_name})" if gpu_name else "")

    ensure_dirs()
    Base.metadata.create_all(engine)

    recover_stale_jobs()

    stop = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, lambda s, _f: _on_signal(stop, s))
        except (ValueError, OSError):
            # signal не работает в дочерних потоках (например, в смок-тесте)
            pass

    log.info("entering main loop (poll interval=%ds, heartbeat=%ds)",
             settings.WORKER_POLL_INTERVAL_SEC, settings.WORKER_HEARTBEAT_SEC)

    last_hb = 0.0
    while not stop.is_set():
        try:
            now = _utcnow().timestamp()
            if now - last_hb >= settings.WORKER_HEARTBEAT_SEC:
                heartbeat(gpu_avail)
                last_hb = now

            with SessionLocal() as db:
                job = pick_queued_job(db)
                if job is None:
                    stop.wait(settings.WORKER_POLL_INTERVAL_SEC)
                    continue
                log.info("picked job %s (%s, model=%s)", job.id, job.filename, job.model)
                process_one_job(db, job)
        except Exception:  # noqa: BLE001
            log.exception("worker loop iteration failed; backing off")
            stop.wait(settings.WORKER_POLL_INTERVAL_SEC)

    log.info("=== whisper worker stopped ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
