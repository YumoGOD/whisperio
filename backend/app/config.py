from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "Local Transcriber"
    app_env: str = "local"
    app_host: str = "0.0.0.0"
    app_port: int = 8000

    data_dir: Path = Path("./data")
    upload_dir: Path = Path("./data/uploads")
    work_dir: Path = Path("./data/work")
    transcript_dir: Path = Path("./data/transcripts")
    log_dir: Path = Path("./data/logs")
    glossary_dir: Path = Path("./data/glossary")
    database_path: Path = Path("./data/transcriber.db")

    max_upload_mb: int = 5120
    allowed_extensions: str = "*"

    default_profile: Literal["accuracy_first"] = "accuracy_first"
    whisper_model: str = "large-v3"
    whisper_device: str = "cuda"
    whisper_compute_type: str = "float16"
    whisper_num_workers: int = 1
    whisper_download_root: Path | None = Path("./data/models")
    whisper_language: str | None = "ru"
    whisper_task: Literal["transcribe", "translate"] = "transcribe"

    # Перекрывают decode-параметры профиля (если не None). Температуры: список через запятую, напр. 0.0,0.2,0.4
    whisper_beam_size: int | None = Field(default=None, ge=1, le=32)
    whisper_best_of: int | None = Field(default=None, ge=1, le=32)
    whisper_patience: float | None = Field(default=None, ge=0.0, le=5.0)
    whisper_length_penalty: float | None = Field(default=None, ge=0.0, le=2.0)
    whisper_repetition_penalty: float | None = Field(default=None, ge=1.0, le=2.0)
    whisper_no_repeat_ngram_size: int | None = Field(default=None, ge=0, le=16)
    whisper_prompt_reset_on_temperature: float | None = Field(default=None, ge=0.0, le=1.0)
    whisper_temperature: str | None = None
    whisper_compression_ratio_threshold: float | None = Field(default=None, ge=0.5, le=10.0)
    whisper_log_prob_threshold: float | None = Field(default=None, ge=-10.0, le=1.0)
    whisper_no_speech_threshold: float | None = Field(default=None, ge=0.0, le=1.0)
    whisper_condition_on_previous_text: bool | None = None
    whisper_word_timestamps: bool | None = None

    worker_concurrency: int = 1
    worker_poll_seconds: float = 5.0
    worker_stale_running_minutes: int = 30
    worker_id: str | None = None

    chunk_seconds: int = 1800
    chunk_overlap_seconds: int = 15
    enable_loudnorm: bool = True
    target_sample_rate: int = 16000

    vad_filter: bool | None = None
    vad_min_silence_ms: int | None = None
    vad_speech_pad_ms: int | None = None
    vad_threshold: float | None = None

    glossary_path: Path = Path("./data/glossary/global.yml")
    glossary_prompt_max_chars: int = 700
    glossary_context_max_chars: int = 1200
    glossary_hotwords_max: int = 80
    glossary_enable_hotwords: bool = False
    glossary_enable_hard_normalization: bool = True
    glossary_hard_min_segment_words: int = 3
    glossary_repetition_compression_threshold: float = 4.0

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024

    @property
    def allowed_extension_set(self) -> set[str]:
        return {
            item.strip().lower().lstrip(".")
            for item in self.allowed_extensions.split(",")
            if item.strip()
        }

    def ensure_directories(self) -> None:
        for path in [
            self.data_dir,
            self.upload_dir,
            self.work_dir,
            self.transcript_dir,
            self.log_dir,
            self.glossary_dir,
        ]:
            path.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.ensure_directories()
    return settings
