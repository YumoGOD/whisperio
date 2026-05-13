from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Literal

JobStatus = Literal["pending", "running", "completed", "failed"]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(slots=True)
class Job:
    id: str
    original_filename: str
    upload_path: str
    status: JobStatus
    progress: float
    created_at: str
    started_at: str | None
    finished_at: str | None
    params: dict[str, Any]
    text: str | None
    segments: list[dict[str, Any]]
    error: str | None
    prepared_path: str | None
    transcript_dir: str | None
    worker_id: str | None
    heartbeat_at: str | None
    duration_seconds: float | None
