"""Smoke-test всех HTTP API эндпоинтов.

Запуск:
    cd backend
    python -m tests.test_smoke_api

Воркер не нужен — задачи останутся в status=queued.
"""

import io
import json
import sys
from pathlib import Path

from fastapi.testclient import TestClient

# Свежий старт: убрать БД от прошлых прогонов, чтобы статистика была чистой.
from app.db import DB_URL  # noqa: E402

_PREFIX = "sqlite:///"
if DB_URL.startswith(_PREFIX):
    _db_file = Path(DB_URL[len(_PREFIX):])
    if _db_file.exists():
        _db_file.unlink()
    for suffix in ("-wal", "-shm"):
        side = _db_file.with_name(_db_file.name + suffix)
        if side.exists():
            side.unlink()

from app.main import app  # noqa: E402


def _ok(label: str, resp, expected_status: int) -> None:
    body = ""
    try:
        body = json.dumps(resp.json(), ensure_ascii=False, indent=2)
    except Exception:
        body = resp.text
    assert resp.status_code == expected_status, (
        f"{label}: expected {expected_status}, got {resp.status_code}\n{body}"
    )
    print(f"  ok  {label:40s} -> {resp.status_code}")


def main() -> int:
    with TestClient(app) as client:
        # 1) health (на старте: worker_alive=False, queue_size=0)
        r = client.get("/api/health")
        _ok("GET /api/health (cold)", r, 200)
        h = r.json()
        assert h["status"] == "ok"
        assert h["worker_alive"] is False
        assert h["gpu_available"] is False
        assert h["queue_size"] == 0

        # 2) list empty
        r = client.get("/api/jobs")
        _ok("GET /api/jobs (empty)", r, 200)
        assert r.json()["jobs"] == []

        # 3) POST с плохим расширением -> 400
        r = client.post(
            "/api/jobs",
            files={"file": ("bad.exe", b"\x00\x01\x02", "application/octet-stream")},
        )
        _ok("POST /api/jobs (bad ext)", r, 400)

        # 4) POST с неизвестной моделью -> 400
        r = client.post(
            "/api/jobs",
            files={"file": ("hello.mp3", b"fake-bytes", "audio/mpeg")},
            data={"model": "bogus"},
        )
        _ok("POST /api/jobs (bad model)", r, 400)

        # 5) POST ok
        payload = b"fake-audio-bytes" * 100
        r = client.post(
            "/api/jobs",
            files={"file": ("meeting.mp3", io.BytesIO(payload), "audio/mpeg")},
            data={"language": "ru", "model": "large-v3"},
        )
        _ok("POST /api/jobs (ok)", r, 201)
        body = r.json()
        assert body["status"] == "queued"
        assert body["filename"] == "meeting.mp3"
        job_id = body["id"]

        # 6) list now has 1
        r = client.get("/api/jobs")
        _ok("GET /api/jobs (1)", r, 200)
        jobs = r.json()["jobs"]
        assert len(jobs) == 1
        assert jobs[0]["id"] == job_id
        assert jobs[0]["model"] == "large-v3"

        # 7) filter by status=done (ничего)
        r = client.get("/api/jobs", params={"status": "done"})
        _ok("GET /api/jobs?status=done", r, 200)
        assert r.json()["jobs"] == []

        # 8) filter by status=unknown -> 400
        r = client.get("/api/jobs", params={"status": "bogus"})
        _ok("GET /api/jobs?status=bogus", r, 400)

        # 9) GET by id
        r = client.get(f"/api/jobs/{job_id}")
        _ok("GET /api/jobs/{id}", r, 200)
        detail = r.json()
        assert detail["status"] == "queued"
        assert detail["segments"] is None
        assert detail["transcript_text"] is None
        assert detail["language"] == "ru"

        # 10) GET by missing id -> 404
        r = client.get("/api/jobs/00000000-0000-0000-0000-000000000000")
        _ok("GET /api/jobs/{missing}", r, 404)

        # 11) download до завершения -> 404 (job не done)
        r = client.get(f"/api/jobs/{job_id}/download", params={"format": "txt"})
        _ok("GET /api/jobs/{id}/download (not done)", r, 404)

        # 12) download c неверным форматом -> 400
        r = client.get(f"/api/jobs/{job_id}/download", params={"format": "doc"})
        _ok("GET /api/jobs/{id}/download (bad format)", r, 400)

        # 13) health после создания (queue_size=1)
        r = client.get("/api/health")
        _ok("GET /api/health (with queued)", r, 200)
        assert r.json()["queue_size"] == 1

        # 14) DELETE
        r = client.delete(f"/api/jobs/{job_id}")
        _ok("DELETE /api/jobs/{id}", r, 204)

        # 15) DELETE дважды -> 404
        r = client.delete(f"/api/jobs/{job_id}")
        _ok("DELETE /api/jobs/{id} (twice)", r, 404)

        # 16) после удаления — список пуст, очередь пуста
        r = client.get("/api/jobs")
        _ok("GET /api/jobs (after delete)", r, 200)
        assert r.json()["jobs"] == []
        r = client.get("/api/health")
        assert r.json()["queue_size"] == 0

    print("\nall checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
