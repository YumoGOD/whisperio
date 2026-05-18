"""Application settings loaded from .env via Pydantic Settings v2.

Импортируется как:
    from app.config import settings

Никаких os.getenv в бизнес-логике — только через этот модуль.
"""

from pathlib import Path
from typing import Any

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
ENV_FILE: Path = PROJECT_ROOT / ".env"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=ENV_FILE,
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ===== Backend / API =====
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    STORAGE_DIR: Path = PROJECT_ROOT / "storage"
    DB_URL: str = "sqlite:///./storage/db.sqlite"
    MAX_UPLOAD_SIZE_MB: int = 2048
    ALLOWED_EXTENSIONS: list[str] = [
        "mp3", "wav", "m4a", "ogg", "flac",
        "mp4", "mov", "mkv", "webm", "avi",
    ]
    LOG_LEVEL: str = "INFO"
    CORS_ORIGINS: list[str] = ["http://localhost:5173"]

    # ===== Whisper =====
    WHISPER_MODEL: str = "large-v3"
    WHISPER_DEVICE: str = "cuda"
    WHISPER_COMPUTE_TYPE: str = "float16"
    WHISPER_BATCH_SIZE: int = 8
    WHISPER_MODELS_DIR: Path = PROJECT_ROOT / "storage" / "models"
    DEFAULT_LANGUAGE: str | None = None
    BEAM_SIZE: int = 5

    # ===== VAD =====
    VAD_ENABLED: bool = True
    VAD_MIN_SILENCE_MS: int = 500
    VAD_SPEECH_PAD_MS: int = 200
    VAD_MAX_SEGMENT_SEC: int = 30

    # ===== Worker =====
    WORKER_POLL_INTERVAL_SEC: int = 2
    WORKER_HEARTBEAT_SEC: int = 5
    STALE_JOB_TIMEOUT_MIN: int = 60

    # ===== Misc =====
    MAX_JOBS_IN_LIST: int = 100

    @field_validator("CORS_ORIGINS", "ALLOWED_EXTENSIONS", mode="before")
    @classmethod
    def _split_csv(cls, v: Any) -> Any:
        # Pydantic v2 не парсит CSV в list[str] автоматически — делаем вручную.
        if isinstance(v, str):
            return [item.strip() for item in v.split(",") if item.strip()]
        return v

    @field_validator("DEFAULT_LANGUAGE", mode="before")
    @classmethod
    def _empty_to_none(cls, v: Any) -> Any:
        # В .env пустая строка означает «автоопределение» (None).
        if isinstance(v, str) and v.strip() == "":
            return None
        return v

    @field_validator("STORAGE_DIR", "WHISPER_MODELS_DIR", mode="after")
    @classmethod
    def _resolve_relative_path(cls, v: Path) -> Path:
        # Относительные пути из .env резолвим относительно корня проекта,
        # чтобы api (cwd=backend/) и worker смотрели в одну и ту же папку.
        if not v.is_absolute():
            return (PROJECT_ROOT / v).resolve()
        return v


settings = Settings()
