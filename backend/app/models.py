"""ORM-модели: Job и WorkerHeartbeat.

Схема намеренно плоская: одна таблица задач, плюс одна строка с heartbeat'ом
воркера (id всегда = 1). См. 02_ARCHITECTURE.md.
"""

from datetime import datetime, timezone
from uuid import uuid4

from sqlalchemy import Boolean, DateTime, Float, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.db import Base


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _new_uuid() -> str:
    return str(uuid4())


# Возможные значения Job.status.
JOB_STATUSES = ("queued", "preprocessing", "transcribing", "done", "failed")


class Job(Base):
    __tablename__ = "jobs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_new_uuid)
    filename: Mapped[str] = mapped_column(String(512))
    file_size: Mapped[int] = mapped_column(Integer)

    duration_sec: Mapped[float | None] = mapped_column(Float, nullable=True, default=None)
    language: Mapped[str | None] = mapped_column(String(16), nullable=True, default=None)
    detected_language: Mapped[str | None] = mapped_column(String(16), nullable=True, default=None)

    model: Mapped[str] = mapped_column(String(64))
    status: Mapped[str] = mapped_column(String(32), default="queued", index=True)
    progress: Mapped[float] = mapped_column(Float, default=0.0)
    error: Mapped[str | None] = mapped_column(Text, nullable=True, default=None)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, index=True
    )
    started_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True, default=None
    )
    finished_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True, default=None
    )

    input_path: Mapped[str] = mapped_column(String(1024))
    result_dir: Mapped[str | None] = mapped_column(String(1024), nullable=True, default=None)
    transcript_text: Mapped[str | None] = mapped_column(Text, nullable=True, default=None)


class WorkerHeartbeat(Base):
    __tablename__ = "worker_heartbeat"

    # Всегда одна строка с id=1 — воркер один.
    id: Mapped[int] = mapped_column(Integer, primary_key=True, default=1)
    last_seen: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow
    )
    gpu_available: Mapped[bool] = mapped_column(Boolean, default=False)
