# Архитектура

## Компоненты

```
┌─────────────────┐         ┌──────────────────┐
│  React (Vite)   │ ──────▶ │  FastAPI (api)   │
│  localhost:5173 │  HTTP   │  localhost:8000  │
└─────────────────┘         └────────┬─────────┘
                                     │
                              SQLite │ + ./storage/
                                     │
                            ┌────────▼─────────┐
                            │  Worker (python) │
                            │  + faster-whisper │
                            │  + GPU (L4)       │
                            └──────────────────┘
```

- **API** — принимает файлы, кладёт в `./storage/uploads/<job_id>/`, создаёт запись Job в SQLite со статусом `queued`. Отдаёт статус и результат.
- **Worker** — отдельный процесс (`python -m app.worker`). В цикле раз в 2 секунды забирает из БД `SELECT * WHERE status='queued' LIMIT 1`, помечает `processing`, обрабатывает, обновляет статус и результат.
- **SQLite** — единая БД `./storage/db.sqlite`. WAL-режим для конкуррентных чтений API/worker.
- **Файлы** — `./storage/uploads/<job_id>/input.ext` (оригинал), `./storage/results/<job_id>/{transcript.txt, transcript.srt, transcript.json}`.

## Структура репозитория

```
whisper-service/
├── .env.example
├── .gitignore
├── README.md
├── docker-compose.yml        # api + worker (опц.)
├── backend/
│   ├── pyproject.toml         # или requirements.txt
│   ├── Dockerfile
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py            # FastAPI app, эндпоинты
│   │   ├── worker.py          # python -m app.worker
│   │   ├── config.py          # Pydantic Settings, читает .env
│   │   ├── db.py              # SQLAlchemy engine, SessionLocal, Base.metadata.create_all
│   │   ├── models.py          # Job (ORM)
│   │   ├── schemas.py         # Pydantic-схемы для API
│   │   ├── pipeline/
│   │   │   ├── __init__.py
│   │   │   ├── audio.py       # ffmpeg-обёртка
│   │   │   ├── vad.py         # Silero VAD
│   │   │   ├── transcribe.py  # faster-whisper + батчинг
│   │   │   └── formats.py     # txt / srt / json генераторы
│   │   └── storage.py         # пути, создание директорий
│   └── tests/
│       └── test_smoke.py
└── frontend/
    ├── package.json
    ├── vite.config.ts
    ├── tailwind.config.ts
    ├── index.html
    └── src/
        ├── main.tsx
        ├── App.tsx
        ├── api.ts             # типизированный клиент к FastAPI
        ├── components/
        │   ├── UploadZone.tsx
        │   ├── JobCard.tsx
        │   ├── JobList.tsx
        │   └── TranscriptView.tsx
        └── pages/
            ├── Home.tsx
            └── History.tsx
```

## Модель данных (одна таблица)

```python
class Job:
    id: str (UUID, PK)
    filename: str               # оригинальное имя
    file_size: int              # байты
    duration_sec: float | None  # после ffprobe
    language: str | None        # запрошенный язык или None=auto
    detected_language: str | None
    model: str                  # large-v3 / large-v3-turbo / medium
    status: str                 # queued | preprocessing | transcribing | done | failed
    progress: float             # 0..1, обновляется воркером
    error: str | None
    created_at: datetime
    started_at: datetime | None
    finished_at: datetime | None
    input_path: str
    result_dir: str | None
    transcript_text: str | None  # для быстрого превью без чтения файла
```

## Жизненный цикл задачи

```
POST /jobs  →  status=queued        (API)
             ↓
Worker picks up
             ↓
            status=preprocessing    (ffmpeg + VAD)
             ↓
            status=transcribing     (faster-whisper, обновляет progress)
             ↓
            status=done / failed
```

## Принципы кода

- Все настройки идут через `app.config.settings` (instance of `Settings`)
- Никаких `os.getenv` в бизнес-логике
- Все пути — через `pathlib.Path`, базовый путь из `settings.STORAGE_DIR`
- Логи через стандартный `logging`, формат настраивается через .env
- Никаких глобальных моделей в API-процессе — модель грузится **только в воркере** один раз при старте
