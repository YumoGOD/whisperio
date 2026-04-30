from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from app import db
from app.main import delete_job, get_stats
from app.models import JobStatus


def _request():
    return SimpleNamespace(state=SimpleNamespace(request_id="test-request"))


def _insert_job(
    job_id: str,
    *,
    status: str,
    stored_path: Path,
    prepared_path: Path | None = None,
    created_at: str | None = None,
    finished_at: str | None = None,
    duration_sec: float | None = None,
    processing_duration_ms: int | None = None,
    preprocess_duration_ms: int | None = None,
    transcribe_duration_ms: int | None = None,
) -> None:
    created = created_at or datetime.now(timezone.utc).isoformat()
    with db.get_connection() as conn:
        conn.execute(
            """
            INSERT INTO transcription_jobs (
                id, original_filename, stored_path, status, created_at, finished_at,
                duration_sec, processing_duration_ms, preprocess_duration_ms,
                transcribe_duration_ms, prepared_audio_path
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                job_id,
                f"{job_id}.wav",
                str(stored_path),
                status,
                created,
                finished_at,
                duration_sec,
                processing_duration_ms,
                preprocess_duration_ms,
                transcribe_duration_ms,
                str(prepared_path) if prepared_path else None,
            ),
        )


@pytest.fixture()
def temp_db(tmp_path, monkeypatch):
    database_path = tmp_path / "test.db"
    monkeypatch.setattr(db, "DATABASE_PATH", database_path)
    db.init_db()
    return tmp_path


def test_delete_processing_marks_request(temp_db):
    source_file = temp_db / "processing.wav"
    source_file.write_bytes(b"audio")
    _insert_job("job-processing", status=JobStatus.PROCESSING, stored_path=source_file)

    payload = delete_job("job-processing", _request())

    assert payload["status"] == "pending_delete"
    with db.get_connection() as conn:
        row = conn.execute(
            "SELECT delete_requested FROM transcription_jobs WHERE id = ?",
            ("job-processing",),
        ).fetchone()
    assert row is not None
    assert row["delete_requested"] == 1


def test_delete_done_removes_job_and_files(temp_db):
    source_file = temp_db / "done.wav"
    prepared_file = temp_db / "done.prepared.wav"
    source_file.write_bytes(b"audio")
    prepared_file.write_bytes(b"prepared")
    _insert_job(
        "job-done",
        status=JobStatus.DONE,
        stored_path=source_file,
        prepared_path=prepared_file,
        finished_at=datetime.now(timezone.utc).isoformat(),
    )

    payload = delete_job("job-done", _request())

    assert payload["status"] == "deleted"
    assert not source_file.exists()
    assert not prepared_file.exists()
    with db.get_connection() as conn:
        row = conn.execute("SELECT id FROM transcription_jobs WHERE id = ?", ("job-done",)).fetchone()
    assert row is None


def test_get_stats_returns_normalized_values(temp_db):
    now = datetime.now(timezone.utc)
    common_from = now - timedelta(days=1)
    common_to = now + timedelta(days=1)
    src_a = temp_db / "a.wav"
    src_b = temp_db / "b.wav"
    src_a.write_bytes(b"a")
    src_b.write_bytes(b"b")
    _insert_job(
        "job-a",
        status=JobStatus.DONE,
        stored_path=src_a,
        finished_at=now.isoformat(),
        duration_sec=1800.0,
        processing_duration_ms=900000,
        preprocess_duration_ms=180000,
        transcribe_duration_ms=540000,
    )
    _insert_job(
        "job-b",
        status=JobStatus.DONE,
        stored_path=src_b,
        finished_at=now.isoformat(),
        duration_sec=3600.0,
        processing_duration_ms=1200000,
        preprocess_duration_ms=300000,
        transcribe_duration_ms=840000,
    )

    payload = get_stats(_request(), from_=common_from, to=common_to)

    assert payload.completed_jobs == 2
    assert payload.fastest.job_id == "job-b"
    assert payload.slowest.job_id == "job-a"
    assert payload.average.total_ms == 1500000.0
    assert payload.average.preprocess_ms == 330000.0
    assert payload.average.transcribe_ms == 960000.0


def test_get_stats_rejects_invalid_range():
    now = datetime.now(timezone.utc)
    with pytest.raises(HTTPException) as exc:
        get_stats(_request(), from_=now, to=now - timedelta(days=1))
    assert exc.value.status_code == 400
