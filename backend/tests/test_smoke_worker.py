"""End-to-end smoke-test: API + worker в одном процессе.

Через TestClient заливаем speech.wav из Задачи 4, прогоняем одну итерацию
воркера синхронно, проверяем что задача дошла до status=done и API возвращает
правильные данные. На CPU прогон занимает около минуты (faster-whisper large-v3).

Запуск:
    cd backend
    python -m tests.test_smoke_worker
"""

from __future__ import annotations

import sys
from pathlib import Path

from fastapi.testclient import TestClient

# Свежий старт: удаляем старую БД (после Задачи 5 в схеме нет колонки
# worker_heartbeat.gpu_available — пересоздаём).
from app.db import DB_URL  # noqa: E402

_PREFIX = "sqlite:///"
if DB_URL.startswith(_PREFIX):
    _db_file = Path(DB_URL[len(_PREFIX):])
    for suffix in ("", "-wal", "-shm"):
        f = Path(str(_db_file) + suffix)
        if f.exists():
            f.unlink()

from app.config import settings  # noqa: E402
from app.db import SessionLocal  # noqa: E402
from app.main import app  # noqa: E402
from app.worker import (  # noqa: E402
    _gpu_available,
    heartbeat,
    pick_queued_job,
    process_one_job,
    recover_stale_jobs,
)

SPEECH = settings.STORAGE_DIR / "_smoke_pipeline" / "speech.wav"


def _expect(label: str, cond: bool, detail: str = "") -> None:
    if not cond:
        print(f"  FAIL {label}: {detail}", file=sys.stderr)
        sys.exit(1)
    print(f"  ok   {label}")


def main() -> int:
    if not SPEECH.exists():
        print(f"ERR: {SPEECH} not found; run task 4 smoke test first", file=sys.stderr)
        return 2

    with TestClient(app) as client:
        # cold health: воркера ещё нет
        r = client.get("/api/health")
        _expect("GET /api/health (cold)", r.status_code == 200, r.text)
        body = r.json()
        _expect("health worker_alive=False", body["worker_alive"] is False, str(body))
        _expect("health gpu_available=False", body["gpu_available"] is False, str(body))
        _expect("health queue_size=0", body["queue_size"] == 0, str(body))

        # подаём задачу через POST
        with SPEECH.open("rb") as f:
            r = client.post(
                "/api/jobs",
                files={"file": ("speech.wav", f, "audio/wav")},
                data={"language": "en", "model": settings.WHISPER_MODEL},
            )
        _expect("POST /api/jobs", r.status_code == 201, r.text)
        job_id = r.json()["id"]
        print(f"  created job: {job_id}")

        # один прогон воркера
        gpu = _gpu_available()
        heartbeat(gpu)
        recover_stale_jobs()  # ничего не должно восстановить

        with SessionLocal() as db:
            picked = pick_queued_job(db)
            _expect("worker picked queued job", picked is not None and picked.id == job_id,
                    f"picked={picked}")
            print(f"  worker processing job {picked.id} ...")
            process_one_job(db, picked)

        # В реальном цикле воркер пишет heartbeat каждые WORKER_HEARTBEAT_SEC сек.
        # Здесь process_one_job отработал >30s — симулируем след. итерацию.
        heartbeat(gpu)

        # проверяем результат через API
        r = client.get(f"/api/jobs/{job_id}")
        _expect("GET /api/jobs/{id}", r.status_code == 200, r.text)
        body = r.json()
        print(f"  status={body['status']}, progress={body['progress']}, "
              f"detected={body['detected_language']!r}")
        _expect("job.status == done", body["status"] == "done", str(body))
        _expect("job.progress == 1.0", body["progress"] == 1.0, str(body))
        _expect("job.error is None", body["error"] is None, str(body))
        _expect("detected_language is set",
                isinstance(body["detected_language"], str) and len(body["detected_language"]) > 0,
                str(body))
        _expect("transcript_text non-empty",
                isinstance(body["transcript_text"], str) and len(body["transcript_text"]) > 0,
                str(body)[:200])
        _expect("segments non-empty",
                isinstance(body["segments"], list) and len(body["segments"]) > 0,
                str(body)[:200])
        print(f"  transcript preview: {body['transcript_text'][:120]!r}")
        print(f"  segments[0]: start={body['segments'][0]['start']}, "
              f"end={body['segments'][0]['end']}, text={body['segments'][0]['text']!r}")

        # скачивание трёх форматов
        for fmt, mime_prefix in (("txt", "text/plain"), ("srt", "application/x-subrip"),
                                 ("json", "application/json")):
            r = client.get(f"/api/jobs/{job_id}/download", params={"format": fmt})
            _expect(f"download .{fmt} = 200", r.status_code == 200, r.text[:200])
            _expect(f"download .{fmt} non-empty", len(r.content) > 0, "")
            ct = r.headers.get("content-type", "")
            _expect(f"download .{fmt} content-type", ct.startswith(mime_prefix),
                    f"got={ct}")

        # health после обработки
        r = client.get("/api/health")
        body = r.json()
        print(f"  health after: {body}")
        _expect("health worker_alive=True", body["worker_alive"] is True, str(body))
        _expect("health gpu_available matches", body["gpu_available"] == gpu, str(body))
        _expect("health queue_size=0", body["queue_size"] == 0, str(body))

        # list должен показать одну задачу done
        r = client.get("/api/jobs")
        jobs = r.json()["jobs"]
        _expect("list has 1 job", len(jobs) == 1, str(jobs))
        _expect("listed job status=done", jobs[0]["status"] == "done", str(jobs))

    print("\nsmoke worker ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
