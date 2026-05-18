"""Pydantic-схемы для HTTP API (request/response).

Совместимы со SQLAlchemy ORM через `from_attributes=True`.
"""

from datetime import datetime

from pydantic import BaseModel, ConfigDict


class JobCreateResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    filename: str
    status: str
    created_at: datetime


class JobSummary(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    filename: str
    status: str
    duration_sec: float | None = None
    model: str
    detected_language: str | None = None
    progress: float = 0.0
    created_at: datetime
    finished_at: datetime | None = None


class JobsList(BaseModel):
    jobs: list[JobSummary]


class Segment(BaseModel):
    start: float
    end: float
    text: str


class JobDetail(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    filename: str
    status: str
    progress: float
    duration_sec: float | None = None
    model: str
    language: str | None = None
    detected_language: str | None = None
    transcript_text: str | None = None
    segments: list[Segment] | None = None
    error: str | None = None
    created_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None


class HealthResponse(BaseModel):
    status: str
    worker_alive: bool
    gpu_available: bool
    queue_size: int
