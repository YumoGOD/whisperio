"""Файловое хранилище: фиксированные пути и инициализация директорий.

Структура:
    <STORAGE_DIR>/
        uploads/<job_id>/input.<ext>      — оригинальный загруженный файл
        results/<job_id>/transcript.txt   — итоги
        results/<job_id>/transcript.srt
        results/<job_id>/transcript.json
        models/                           — кэш весов faster-whisper
        db.sqlite                         — БД
"""

import shutil
from pathlib import Path

from app.config import settings


def uploads_dir() -> Path:
    return settings.STORAGE_DIR / "uploads"


def results_dir() -> Path:
    return settings.STORAGE_DIR / "results"


def models_dir() -> Path:
    return settings.WHISPER_MODELS_DIR


def get_input_path(job_id: str) -> Path:
    """Директория задачи в uploads/. Конкретный файл — input.<ext> внутри неё
    (расширение знает только вызывающий код, поэтому возвращаем директорию)."""
    return uploads_dir() / job_id


def get_result_dir(job_id: str) -> Path:
    return results_dir() / job_id


def ensure_dirs() -> None:
    """Создать все корневые директории хранилища. Идемпотентно."""
    for d in (settings.STORAGE_DIR, uploads_dir(), results_dir(), models_dir()):
        d.mkdir(parents=True, exist_ok=True)


def cleanup_job(job_id: str) -> None:
    """Удалить все файлы задачи (uploads + results). Идемпотентно."""
    for d in (get_input_path(job_id), get_result_dir(job_id)):
        if d.exists():
            shutil.rmtree(d, ignore_errors=True)
