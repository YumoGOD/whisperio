from __future__ import annotations

import logging
import os
import queue
import signal
import socket
import time
import traceback
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
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


def process_job(job: Job, repo: JobRepository, worker_name: str, pipeline_pool: "queue.Queue[TranscriptionPipeline]") -> None:
    logger.info("Запуск задачи %s (%s)", job.id, job.original_filename)
    pipeline = pipeline_pool.get()

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
    finally:
        pipeline_pool.put(pipeline)


def main() -> None:
    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    repo = JobRepository(settings.database_path)
    name = worker_id()
    recovered = repo.recover_running_jobs(worker_id=name)
    if recovered:
        logger.info("Восстановлено незавершенных задач после перезапуска: %s", recovered)

    concurrency = max(1, settings.worker_concurrency)
    logger.info(
        "Worker %s запущен: параллельных задач=%d, интервал опроса=%s сек.",
        name,
        concurrency,
        settings.worker_poll_seconds,
    )

    # Каждый слот получает собственный TranscriptionPipeline (и свою копию WhisperModel),
    # что устраняет race condition CTranslate2 при concurrency > 1.
    # Модели загружаются лениво — при первой задаче в каждом слоте.
    pipeline_pool: queue.Queue[TranscriptionPipeline] = queue.Queue()
    for _ in range(concurrency):
        pipeline_pool.put(TranscriptionPipeline(settings))

    if concurrency > 1:
        logger.info(
            "Параллельный режим: %d независимых pipeline. "
            "Убедитесь, что GPU имеет достаточно VRAM для %d экземпляров модели одновременно.",
            concurrency,
            concurrency,
        )

    active: set[Future] = set()
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        while not stop_requested:
            active = {future for future in active if not future.done()}
            while len(active) < concurrency:
                job = repo.claim_next_job(name)
                if job is None:
                    break
                active.add(executor.submit(process_job, job, repo, name, pipeline_pool))

            if active:
                done, active = wait(active, timeout=settings.worker_poll_seconds, return_when=FIRST_COMPLETED)
                for future in done:
                    try:
                        future.result()
                    except Exception:
                        # process_job handles all exceptions internally (logs + fail_job).
                        # Re-raising here would kill the worker loop.
                        pass
            else:
                time.sleep(settings.worker_poll_seconds)

    if active:
        logger.info("Ожидание завершения активных задач перед остановкой: %s", len(active))
        wait(active)
    logger.info("Worker %s остановлен", name)


if __name__ == "__main__":
    main()
