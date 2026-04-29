import logging
import os
import sqlite3
from contextlib import contextmanager
from pathlib import Path

from app.logging_utils import log_event
from app.models import JobStage, JobStatus

DATABASE_PATH = Path(os.getenv("DATABASE_PATH", "data/app.db"))
SQLITE_BUSY_TIMEOUT_MS = int(os.getenv("SQLITE_BUSY_TIMEOUT_MS", "30000"))
logger = logging.getLogger("whisperio.db")


def ensure_data_dirs() -> None:
    DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)


def init_db() -> None:
    ensure_data_dirs()
    log_event(logger, event="db_init_started", component="db.schema", db_path=str(DATABASE_PATH))
    try:
        with get_connection() as conn:
            conn.execute("PRAGMA journal_mode = WAL;")
            conn.execute("PRAGMA synchronous = NORMAL;")
            conn.execute("PRAGMA wal_autocheckpoint = 1000;")
            conn.executescript(
                """
            CREATE TABLE IF NOT EXISTS transcription_jobs (
                id TEXT PRIMARY KEY,
                request_id TEXT,
                original_filename TEXT NOT NULL,
                stored_path TEXT NOT NULL,
                whisper_model_size TEXT,
                whisper_cpu_threads INTEGER,
                whisper_beam_size INTEGER,
                status TEXT NOT NULL,
                stage TEXT,
                progress REAL,
                status_message TEXT,
                error TEXT,
                error_code TEXT,
                created_at TEXT NOT NULL,
                started_at TEXT,
                finished_at TEXT,
                duration_sec REAL,
                processing_duration_ms INTEGER,
                transcribe_duration_ms INTEGER,
                preprocess_duration_ms INTEGER,
                prepared_audio_path TEXT,
                quality_flags TEXT
            );

            CREATE TABLE IF NOT EXISTS transcription_segments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                job_id TEXT NOT NULL,
                idx INTEGER NOT NULL,
                start_sec REAL NOT NULL,
                end_sec REAL NOT NULL,
                text TEXT NOT NULL,
                FOREIGN KEY(job_id) REFERENCES transcription_jobs(id) ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_segments_job_id
            ON transcription_segments(job_id);

            CREATE INDEX IF NOT EXISTS idx_jobs_status_created
            ON transcription_jobs(status, created_at);
            """
            )
            added_columns = _ensure_job_columns(conn)
            conn.execute(
                """
            CREATE INDEX IF NOT EXISTS idx_jobs_request_id
            ON transcription_jobs(request_id);
            """
            )
        log_event(
            logger,
            event="db_init_completed",
            component="db.schema",
            added_columns=",".join(added_columns) if added_columns else None,
        )
    except sqlite3.Error as exc:
        log_event(
            logger,
            event="db_init_failed",
            level=logging.ERROR,
            component="db.schema",
            error_type=type(exc).__name__,
            error_message=str(exc),
        )
        raise


def _ensure_job_columns(conn: sqlite3.Connection) -> list[str]:
    columns = {
        row["name"]
        for row in conn.execute("PRAGMA table_info(transcription_jobs)").fetchall()
    }
    missing_columns = [
        ("request_id", "TEXT"),
        ("whisper_model_size", "TEXT"),
        ("whisper_cpu_threads", "INTEGER"),
        ("whisper_beam_size", "INTEGER"),
        ("stage", "TEXT"),
        ("progress", "REAL"),
        ("status_message", "TEXT"),
        ("error_code", "TEXT"),
        ("processing_duration_ms", "INTEGER"),
        ("transcribe_duration_ms", "INTEGER"),
        ("preprocess_duration_ms", "INTEGER"),
        ("prepared_audio_path", "TEXT"),
        ("quality_flags", "TEXT"),
    ]
    added: list[str] = []
    for name, definition in missing_columns:
        if name not in columns:
            conn.execute(f"ALTER TABLE transcription_jobs ADD COLUMN {name} {definition}")
            added.append(name)
    return added


@contextmanager
def get_connection():
    conn = sqlite3.connect(DATABASE_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute(f"PRAGMA busy_timeout = {SQLITE_BUSY_TIMEOUT_MS};")
    conn.execute("PRAGMA foreign_keys = ON;")
    try:
        yield conn
        conn.commit()
    except sqlite3.Error as exc:
        conn.rollback()
        log_event(
            logger,
            event="db_query_failed",
            level=logging.ERROR,
            component="db.connection",
            error_type=type(exc).__name__,
            error_message=str(exc),
        )
        raise
    finally:
        conn.close()


def requeue_unfinished_jobs() -> int:
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT id, status_message
            FROM transcription_jobs
            WHERE status = ?
            ORDER BY created_at ASC
            """,
            (JobStatus.PROCESSING,),
        ).fetchall()
        if not rows:
            return 0

        for row in rows:
            existing_message = (row["status_message"] or "").strip()
            message = "Перезапуск сервиса, задача возвращена в очередь."
            restored_message = (
                f"{existing_message}. {message}" if existing_message else message
            )
            conn.execute(
                """
                UPDATE transcription_jobs
                SET status = ?, stage = ?, progress = ?, status_message = ?,
                    started_at = NULL, finished_at = NULL,
                    duration_sec = NULL, processing_duration_ms = NULL,
                    transcribe_duration_ms = NULL, preprocess_duration_ms = NULL,
                    prepared_audio_path = NULL, quality_flags = NULL,
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
        restored_count = len(rows)
    log_event(
        logger,
        event="db_requeue_unfinished_jobs",
        component="db.queue",
        restored_count=restored_count,
    )
    return restored_count


def claim_next_job() -> dict[str, str | None] | None:
    with get_connection() as conn:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute(
            """
            SELECT id, request_id
            FROM transcription_jobs
            WHERE status = ?
            ORDER BY created_at ASC
            LIMIT 1
            """,
            (JobStatus.QUEUED,),
        ).fetchone()
        if row is None:
            return None

        job_id = str(row["id"])
        updated = conn.execute(
            """
            UPDATE transcription_jobs
            SET status = ?, stage = ?, progress = ?, status_message = ?
            WHERE id = ? AND status = ?
            """,
            (
                JobStatus.PROCESSING,
                JobStage.PREPARING,
                1.0,
                "Задача назначена воркеру",
                job_id,
                JobStatus.QUEUED,
            ),
        )
        if updated.rowcount != 1:
            log_event(
                logger,
                event="db_claim_conflict",
                level=logging.DEBUG,
                component="db.queue",
                job_id=job_id,
            )
            return None
        claimed = {"job_id": job_id, "request_id": row["request_id"]}
    log_event(
        logger,
        event="db_job_claimed",
        level=logging.DEBUG,
        component="db.queue",
        job_id=job_id,
        request_id=claimed["request_id"],
    )
    return claimed


def get_queue_stats() -> dict[str, int]:
    with get_connection() as conn:
        queued = conn.execute(
            "SELECT COUNT(*) as count FROM transcription_jobs WHERE status = ?",
            (JobStatus.QUEUED,),
        ).fetchone()
        processing = conn.execute(
            "SELECT COUNT(*) as count FROM transcription_jobs WHERE status = ?",
            (JobStatus.PROCESSING,),
        ).fetchone()
    return {
        "queued": int(queued["count"]) if queued else 0,
        "processing": int(processing["count"]) if processing else 0,
    }
