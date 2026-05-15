from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from app.models import Job, JobStatus, utc_now


class JobRepository:
    def __init__(self, database_path: Path):
        self.database_path = database_path
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self.init_db()

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.database_path, timeout=30, isolation_level=None)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA busy_timeout=30000")
        conn.execute("PRAGMA foreign_keys=ON")
        try:
            yield conn
        finally:
            conn.close()

    def init_db(self) -> None:
        with self.connect() as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS jobs (
                    id TEXT PRIMARY KEY,
                    original_filename TEXT NOT NULL,
                    upload_path TEXT NOT NULL,
                    status TEXT NOT NULL,
                    progress REAL NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    started_at TEXT,
                    finished_at TEXT,
                    params_json TEXT NOT NULL DEFAULT '{}',
                    text TEXT,
                    segments_json TEXT NOT NULL DEFAULT '[]',
                    error TEXT,
                    prepared_path TEXT,
                    transcript_dir TEXT,
                    worker_id TEXT,
                    heartbeat_at TEXT,
                    duration_seconds REAL
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_jobs_status_created ON jobs(status, created_at)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_jobs_worker ON jobs(worker_id, heartbeat_at)")

    def create_job(
        self,
        job_id: str,
        original_filename: str,
        upload_path: str,
        params: dict[str, Any],
    ) -> Job:
        now = utc_now()
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO jobs (
                    id, original_filename, upload_path, status, progress, created_at, params_json
                ) VALUES (?, ?, ?, 'pending', 0, ?, ?)
                """,
                (job_id, original_filename, upload_path, now, json.dumps(params, ensure_ascii=False)),
            )
        job = self.get_job(job_id)
        if job is None:
            raise RuntimeError("Созданную задачу не удалось загрузить из БД")
        return job

    def get_job(self, job_id: str) -> Job | None:
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        return self._row_to_job(row) if row else None

    def list_jobs(self, limit: int = 100, offset: int = 0) -> list[Job]:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM jobs ORDER BY created_at DESC LIMIT ? OFFSET ?",
                (limit, offset),
            ).fetchall()
        return [self._row_to_job(row) for row in rows]

    def recover_running_jobs(
        self,
        worker_id: str | None = None,
        stale_minutes: int | None = None,
    ) -> int:
        with self.connect() as conn:
            if worker_id is not None:
                # On restart: reset only this worker's own jobs (safe for multi-worker).
                cursor = conn.execute(
                    """
                    UPDATE jobs
                    SET status = 'pending',
                        worker_id = NULL,
                        heartbeat_at = NULL,
                        error = NULL
                    WHERE status = 'running'
                      AND worker_id = ?
                    """,
                    (worker_id,),
                )
            elif stale_minutes is None:
                # Reset ALL running jobs regardless of age (single-worker only!).
                cursor = conn.execute(
                    """
                    UPDATE jobs
                    SET status = 'pending',
                        worker_id = NULL,
                        heartbeat_at = NULL,
                        error = NULL
                    WHERE status = 'running'
                    """
                )
            else:
                cursor = conn.execute(
                    """
                    UPDATE jobs
                    SET status = 'pending',
                        worker_id = NULL,
                        heartbeat_at = NULL,
                        error = NULL
                    WHERE status = 'running'
                      AND (
                        heartbeat_at IS NULL
                        OR datetime(heartbeat_at) < datetime('now', ?)
                      )
                    """,
                    (f"-{stale_minutes} minutes",),
                )
        return cursor.rowcount

    def claim_next_job(self, worker_id: str) -> Job | None:
        now = utc_now()
        with self.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                """
                SELECT * FROM jobs
                WHERE status = 'pending'
                ORDER BY created_at ASC
                LIMIT 1
                """
            ).fetchone()
            if row is None:
                conn.execute("COMMIT")
                return None
            conn.execute(
                """
                UPDATE jobs
                SET status = 'running',
                    started_at = COALESCE(started_at, ?),
                    worker_id = ?,
                    heartbeat_at = ?,
                    progress = CASE WHEN progress < 0.01 THEN 0.01 ELSE progress END,
                    error = NULL
                WHERE id = ?
                """,
                (now, worker_id, now, row["id"]),
            )
            conn.execute("COMMIT")
        return self.get_job(row["id"])

    def update_progress(
        self,
        job_id: str,
        progress: float,
        *,
        status: JobStatus | None = None,
        prepared_path: str | None = None,
        transcript_dir: str | None = None,
        duration_seconds: float | None = None,
    ) -> None:
        assignments = ["progress = ?", "heartbeat_at = ?"]
        values: list[Any] = [max(0.0, min(progress, 1.0)), utc_now()]
        if status is not None:
            assignments.append("status = ?")
            values.append(status)
        if prepared_path is not None:
            assignments.append("prepared_path = ?")
            values.append(prepared_path)
        if transcript_dir is not None:
            assignments.append("transcript_dir = ?")
            values.append(transcript_dir)
        if duration_seconds is not None:
            assignments.append("duration_seconds = ?")
            values.append(duration_seconds)
        values.append(job_id)
        with self.connect() as conn:
            conn.execute(f"UPDATE jobs SET {', '.join(assignments)} WHERE id = ?", values)

    def complete_job(
        self,
        job_id: str,
        text: str,
        segments: list[dict[str, Any]],
        transcript_dir: str,
        prepared_path: str | None,
        duration_seconds: float | None,
    ) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                UPDATE jobs
                SET status = 'completed',
                    progress = 1,
                    finished_at = ?,
                    text = ?,
                    segments_json = ?,
                    error = NULL,
                    transcript_dir = ?,
                    prepared_path = COALESCE(?, prepared_path),
                    duration_seconds = COALESCE(?, duration_seconds),
                    worker_id = NULL,
                    heartbeat_at = NULL
                WHERE id = ?
                """,
                (
                    utc_now(),
                    text,
                    json.dumps(segments, ensure_ascii=False),
                    transcript_dir,
                    prepared_path,
                    duration_seconds,
                    job_id,
                ),
            )

    def fail_job(self, job_id: str, error: str) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                UPDATE jobs
                SET status = 'failed',
                    finished_at = ?,
                    error = ?,
                    worker_id = NULL,
                    heartbeat_at = NULL
                WHERE id = ?
                """,
                (utc_now(), error, job_id),
            )

    def _row_to_job(self, row: sqlite3.Row) -> Job:
        return Job(
            id=row["id"],
            original_filename=row["original_filename"],
            upload_path=row["upload_path"],
            status=row["status"],
            progress=row["progress"],
            created_at=row["created_at"],
            started_at=row["started_at"],
            finished_at=row["finished_at"],
            params=json.loads(row["params_json"] or "{}"),
            text=row["text"],
            segments=json.loads(row["segments_json"] or "[]"),
            error=row["error"],
            prepared_path=row["prepared_path"],
            transcript_dir=row["transcript_dir"],
            worker_id=row["worker_id"],
            heartbeat_at=row["heartbeat_at"],
            duration_seconds=row["duration_seconds"],
        )
