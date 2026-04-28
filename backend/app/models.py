from enum import StrEnum


class JobStatus(StrEnum):
    QUEUED = "queued"
    PROCESSING = "processing"
    DONE = "done"
    FAILED = "failed"


class JobStage(StrEnum):
    QUEUED = "queued"
    PREPARING = "preparing"
    PREPROCESSING = "preprocessing"
    TRANSCRIBING = "transcribing"
    SAVING_SEGMENTS = "saving_segments"
    COMPLETED = "completed"
    FAILED = "failed"
