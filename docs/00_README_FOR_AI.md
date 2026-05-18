# Whisper Transcription Service — Контекст для AI

Ты строишь production-сервис для транскрибации аудио на сервере с NVIDIA L4 24GB.
Целевая нагрузка: до 40k минут аудио в сутки.

## Жёсткие правила

1. **Никакой авторизации, JWT, OAuth, API-ключей пользователей** — внутренний сервис.
2. **Никакого Redis, Memcached, кеширования** — БД SQLite + файловая система.
3. **Никаких микросервисов** — один FastAPI-процесс + один worker-процесс.
4. **Все настройки — через `.env`** (Pydantic Settings). Никаких хардкодов URL/путей/моделей.
5. **Простота важнее «красоты кода»** — плоская структура, минимум абстракций.
6. **Не выдумывай API faster-whisper / WhisperX** — если не уверен в сигнатуре метода, остановись и спроси.

## Стек (зафиксирован, не меняй)

- **Backend:** Python 3.11, FastAPI, Uvicorn, SQLAlchemy 2.x (sync), SQLite, Pydantic Settings v2
- **ML:** faster-whisper (CTranslate2), silero-vad, ffmpeg-python
- **Worker:** отдельный Python-процесс, опрашивает БД (без Celery/RQ — простая polling-очередь)
- **Frontend:** React 18 + Vite + TypeScript + Tailwind CSS + shadcn/ui (минимальный набор компонентов)
- **GPU:** CUDA 12.x, faster-whisper с `device="cuda"`, `compute_type="float16"`

## Что НЕ делать

- Не предлагать Docker Compose с 5 сервисами — максимум 2 контейнера (api+worker) + nginx опционально.
- Не добавлять Alembic — для SQLite достаточно `Base.metadata.create_all()` при старте.
- Не подключать SocketIO — для статуса задач достаточно polling раз в 2 секунды с фронта.
- Не использовать async SQLAlchemy — sync проще и достаточно.
