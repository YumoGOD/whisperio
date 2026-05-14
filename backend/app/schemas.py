from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from app.models import JobStatus


class SegmentOut(BaseModel):
    idx: int
    start_sec: float
    end_sec: float
    text: str
    label: str | None = None
    confidence: float | None = None
    quality_flags: str | None = None


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
    original_audio_url: str | None = None
    prepared_audio_url: str | None = None
    delete_requested: bool = False
    quality_flags: str | None = None


class JobDetail(JobBase):
    segments: list[SegmentOut] = Field(default_factory=list)


class JobListItem(JobBase):
    pass


class JobCreateResponse(BaseModel):
    job_id: str
    status: JobStatus


class JobDeleteResponse(BaseModel):
    job_id: str
    status: str
    message: str


class SpeedBreakdown(BaseModel):
    total_ms: float | None = None
    preprocess_ms: float | None = None
    transcribe_ms: float | None = None


class SpeedExtreme(SpeedBreakdown):
    job_id: str | None = None
    original_filename: str | None = None


class JobStatsResponse(BaseModel):
    range_from: datetime | None = None
    range_to: datetime | None = None
    completed_jobs: int = 0
    average: SpeedBreakdown
    fastest: SpeedExtreme
    slowest: SpeedExtreme


class ErrorResponse(BaseModel):
    code: str
    message: str
    details: list[Any] | None = None
