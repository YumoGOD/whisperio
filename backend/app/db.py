import os
import sqlite3
from contextlib import contextmanager
from pathlib import Path

DATABASE_PATH = Path(os.getenv("DATABASE_PATH", "data/app.db"))


def ensure_data_dirs() -> None:
    DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)


def init_db() -> None:
    ensure_data_dirs()
    with get_connection() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS transcription_jobs (
                id TEXT PRIMARY KEY,
                original_filename TEXT NOT NULL,
                stored_path TEXT NOT NULL,
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
            """
        )
        _ensure_job_columns(conn)


def _ensure_job_columns(conn: sqlite3.Connection) -> None:
    columns = {
        row["name"]
        for row in conn.execute("PRAGMA table_info(transcription_jobs)").fetchall()
    }
    missing_columns = [
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
    for name, definition in missing_columns:
        if name not in columns:
            conn.execute(f"ALTER TABLE transcription_jobs ADD COLUMN {name} {definition}")


@contextmanager
def get_connection():
    conn = sqlite3.connect(DATABASE_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()
