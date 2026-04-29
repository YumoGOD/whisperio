import asyncio
import logging
import os
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import Any

from app.db import claim_next_job, get_queue_stats
from app.logging_utils import bind_log_context, configure_logging, log_event, reset_log_context
from app.queue_worker import TranscriptionWorker

logger = logging.getLogger("whisperio.runtime")

_PROCESS_WORKER: TranscriptionWorker | None = None


def _configure_process_env() -> None:
    omp_threads = os.getenv("WHISPER_PROCESS_OMP_THREADS")
    mkl_threads = os.getenv("WHISPER_PROCESS_MKL_THREADS")
    if omp_threads:
        os.environ["OMP_NUM_THREADS"] = omp_threads
    if mkl_threads:
        os.environ["MKL_NUM_THREADS"] = mkl_threads


def _get_process_worker() -> TranscriptionWorker:
    global _PROCESS_WORKER
    if _PROCESS_WORKER is None:
        configure_logging()
        _configure_process_env()
        _PROCESS_WORKER = TranscriptionWorker()
        token = bind_log_context(worker_pid=os.getpid())
        try:
            log_event(
                logger,
                event="worker_ready",
                component="runtime.worker_process",
                worker_pid=os.getpid(),
            )
        finally:
            reset_log_context(token)
    return _PROCESS_WORKER


def process_job(job_id: str, request_id: str | None) -> str:
    token = bind_log_context(worker_pid=os.getpid(), request_id=request_id, job_id=job_id)
    worker = _get_process_worker()
    try:
        worker.process_claimed_job(job_id)
        return job_id
    finally:
        reset_log_context(token)


@dataclass(slots=True)
class RuntimeConfig:
    process_count: int
    poll_interval_ms: int
    snapshot_interval_sec: int


class TranscriptionRuntime:
    def __init__(self) -> None:
        process_count = max(1, int(os.getenv("WHISPER_WORKER_PROCESSES", "5")))
        poll_interval_ms = max(100, int(os.getenv("WHISPER_DISPATCH_POLL_MS", "300")))
        snapshot_interval_sec = max(1, int(os.getenv("LOG_SNAPSHOT_INTERVAL_SEC", "10")))
        self.config = RuntimeConfig(
            process_count=process_count,
            poll_interval_ms=poll_interval_ms,
            snapshot_interval_sec=snapshot_interval_sec,
        )
        self._shutdown = asyncio.Event()
        self._loop_task: asyncio.Task[None] | None = None
        self._executor: ProcessPoolExecutor | None = None
        self._in_flight: dict[str, tuple[asyncio.Future[Any], str | None]] = {}
        self._last_snapshot_ts = time.monotonic()
        self._last_queue_backlog = 0
        self._claimed_since_snapshot = 0
        self._completed_since_snapshot = 0
        self._failed_since_snapshot = 0

    async def start(self) -> None:
        if self._loop_task is not None:
            return
        self._shutdown.clear()
        self._executor = ProcessPoolExecutor(max_workers=self.config.process_count)
        self._loop_task = asyncio.create_task(self._run(), name="transcription-runtime")
        log_event(
            logger,
            event="runtime_started",
            component="runtime.dispatcher",
            process_count=self.config.process_count,
            poll_interval_ms=self.config.poll_interval_ms,
            snapshot_interval_sec=self.config.snapshot_interval_sec,
        )

    async def stop(self) -> None:
        self._shutdown.set()
        if self._loop_task is not None:
            await self._loop_task
            self._loop_task = None
        if self._executor is not None:
            self._executor.shutdown(wait=True, cancel_futures=True)
            self._executor = None
        log_event(logger, event="runtime_stopped", component="runtime.dispatcher")

    async def _run(self) -> None:
        if self._executor is None:
            return
        loop = asyncio.get_running_loop()
        while not self._shutdown.is_set():
            tick_started = time.monotonic()
            self._collect_done_futures()
            free_slots = self.config.process_count - len(self._in_flight)
            for _ in range(max(0, free_slots)):
                claimed = claim_next_job()
                if not claimed:
                    break
                job_id = str(claimed["job_id"])
                request_id = claimed.get("request_id")
                log_event(
                    logger,
                    event="job_claimed",
                    component="runtime.dispatcher",
                    job_id=job_id,
                    request_id=request_id,
                )
                future = loop.run_in_executor(self._executor, process_job, job_id, request_id)
                self._in_flight[job_id] = (future, request_id)
                self._claimed_since_snapshot += 1
                log_event(
                    logger,
                    event="job_dispatched",
                    component="runtime.dispatcher",
                    job_id=job_id,
                    request_id=request_id,
                    in_flight=len(self._in_flight),
                    free_slots=max(0, self.config.process_count - len(self._in_flight)),
                )
            await asyncio.sleep(self.config.poll_interval_ms / 1000.0)
            loop_lag_ms = max(
                0,
                int(
                    (time.monotonic() - tick_started - self.config.poll_interval_ms / 1000.0)
                    * 1000
                ),
            )
            self._emit_runtime_snapshot(loop_lag_ms=loop_lag_ms)

        # Drain already submitted jobs on shutdown.
        while self._in_flight:
            self._collect_done_futures()
            await asyncio.sleep(0.1)

    def _collect_done_futures(self) -> None:
        done_ids: list[str] = []
        for job_id, (future, request_id) in self._in_flight.items():
            if not future.done():
                continue
            done_ids.append(job_id)
            try:
                future.result()
                self._completed_since_snapshot += 1
                log_event(
                    logger,
                    event="job_process_done",
                    component="runtime.dispatcher",
                    job_id=job_id,
                    request_id=request_id,
                )
            except Exception as exc:  # noqa: BLE001
                self._failed_since_snapshot += 1
                log_event(
                    logger,
                    event="job_process_crash",
                    level=logging.ERROR,
                    component="runtime.dispatcher",
                    job_id=job_id,
                    request_id=request_id,
                    error_message=str(exc),
                )
        for job_id in done_ids:
            self._in_flight.pop(job_id, None)

    def _emit_runtime_snapshot(self, *, loop_lag_ms: int) -> None:
        now = time.monotonic()
        elapsed = now - self._last_snapshot_ts
        if elapsed < self.config.snapshot_interval_sec:
            return
        stats = get_queue_stats()
        queue_backlog = int(stats["queued"])
        processing_jobs = int(stats["processing"])
        snapshot_minutes = max(elapsed / 60.0, 1e-6)
        claimed_per_min = round(self._claimed_since_snapshot / snapshot_minutes, 3)
        completed_per_min = round(self._completed_since_snapshot / snapshot_minutes, 3)
        failed_per_min = round(self._failed_since_snapshot / snapshot_minutes, 3)
        backlog_growth = queue_backlog - self._last_queue_backlog
        log_event(
            logger,
            event="runtime_snapshot",
            component="runtime.dispatcher",
            process_count=self.config.process_count,
            in_flight=len(self._in_flight),
            free_slots=max(0, self.config.process_count - len(self._in_flight)),
            queue_backlog=queue_backlog,
            processing_jobs=processing_jobs,
            dispatch_poll_ms=self.config.poll_interval_ms,
            loop_lag_ms=loop_lag_ms,
            backlog_growth=backlog_growth,
            claimed_per_min=claimed_per_min,
            completed_per_min=completed_per_min,
            failed_per_min=failed_per_min,
        )
        if backlog_growth > 0 and queue_backlog > self.config.process_count:
            log_event(
                logger,
                event="sla_drift_detected",
                level=logging.WARNING,
                component="runtime.dispatcher",
                reason_code="queue_backlog_growth",
                queue_backlog=queue_backlog,
                backlog_growth=backlog_growth,
                in_flight=len(self._in_flight),
            )
        self._last_queue_backlog = queue_backlog
        self._last_snapshot_ts = now
        self._claimed_since_snapshot = 0
        self._completed_since_snapshot = 0
        self._failed_since_snapshot = 0
