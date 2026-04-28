from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from app.models import JobStatus


class SegmentOut(BaseModel):
    idx: int
    start_sec: float
    end_sec: float
    text: str


class JobBase(BaseModel):
    id: str
    original_filename: str
    status: str
    stage: str | None = None
    progress: float | None = None
    status_message: str | None = None
    error: str | None = None
    error_code: str | None = None
    created_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None
    duration_sec: float | None = None
    processing_duration_ms: int | None = None
    transcribe_duration_ms: int | None = None
    preprocess_duration_ms: int | None = None
    prepared_audio_path: str | None = None
    quality_flags: str | None = None


class JobDetail(JobBase):
    segments: list[SegmentOut] = Field(default_factory=list)


class JobListItem(JobBase):
    pass


class JobCreateResponse(BaseModel):
    job_id: str
    status: JobStatus


class ErrorResponse(BaseModel):
    code: str
    message: str
    details: list[Any] | None = None
