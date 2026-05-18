from __future__ import annotations

import logging
import os
import signal
import socket
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from app.config import get_settings
from app.db import JobRepository
from app.logging_config import configure_logging
from app.models import Job
from app.transcription.pipeline import TranscriptionPipeline

settings = get_settings()
configure_logging(settings)
logger = logging.getLogger(__name__)

stop_requested = False


def request_stop(signum, _frame) -> None:
    global stop_requested
    logger.info("Получен сигнал %s, остановка после завершения активных задач", signum)
    stop_requested = True


def worker_id() -> str:
    return settings.worker_id or f"{socket.gethostname()}-{os.getpid()}"


def process_job(job: Job, repo: JobRepository, worker_name: str, pipeline: TranscriptionPipeline) -> None:
    logger.info("Запуск задачи %s (%s)", job.id, job.original_filename)

    def progress(pct: float, stage: str) -> None:
        repo.update_progress(job.id, pct, status="running")
        logger.info("Задача %s: прогресс %.1f%%, этап: %s", job.id, pct * 100, stage)

    try:
        result = pipeline.run(
            job_id=job.id,
            input_path=Path(job.upload_path),
            original_filename=job.original_filename,
            params=job.params,
            progress_callback=progress,
        )
        repo.complete_job(
            job_id=job.id,
            text=result["text"],
            segments=result["segments"],
            transcript_dir=result["transcript_dir"],
            prepared_path=result["prepared_path"],
            duration_seconds=result["duration_seconds"],
        )
        logger.info("Задача %s завершена worker-ом %s", job.id, worker_name)
    except Exception as exc:
        logger.error("Задача %s завершилась ошибкой: %s\n%s", job.id, exc, traceback.format_exc())
        repo.fail_job(job.id, str(exc))


def _run_worker_thread(name: str, repo: JobRepository, pipeline: TranscriptionPipeline) -> None:
    thread_label = threading.current_thread().name
    logger.info("Worker-поток %s запущен", thread_label)
    while not stop_requested:
        job = repo.claim_next_job(name)
        if job is None:
            time.sleep(settings.worker_poll_seconds)
            continue
        process_job(job, repo, name, pipeline)
    logger.info("Worker-поток %s остановлен", thread_label)


def main() -> None:
    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    repo = JobRepository(settings.database_path)
    name = worker_id()
    recovered = repo.recover_running_jobs(worker_id=name)
    if recovered:
        logger.info("Восстановлено незавершённых задач после перезапуска: %s", recovered)

    concurrency = settings.worker_concurrency
    logger.info(
        "Worker %s запущен, concurrency=%d, интервал опроса=%.1f сек.",
        name, concurrency, settings.worker_poll_seconds,
    )

    if concurrency > 1 and settings.whisper_device == "cuda":
        logger.warning(
            "worker_concurrency=%d при device=cuda: каждый поток загружает отдельную копию модели в VRAM. "
            "Убедитесь, что видеопамяти достаточно.",
            concurrency,
        )

    if concurrency == 1:
        pipeline = TranscriptionPipeline(settings)
        while not stop_requested:
            job = repo.claim_next_job(name)
            if job is None:
                time.sleep(settings.worker_poll_seconds)
                continue
            process_job(job, repo, name, pipeline)
    else:
        # Каждый поток получает собственный экземпляр pipeline (и модели).
        pipelines = [TranscriptionPipeline(settings) for _ in range(concurrency)]
        with ThreadPoolExecutor(max_workers=concurrency, thread_name_prefix="worker") as executor:
            futures = [
                executor.submit(_run_worker_thread, name, repo, pipelines[i])
                for i in range(concurrency)
            ]
            for future in futures:
                try:
                    future.result()
                except Exception as exc:
                    logger.error("Worker-поток завершился с ошибкой: %s\n%s", exc, traceback.format_exc())

    logger.info("Worker %s остановлен", name)


if __name__ == "__main__":
    main()
