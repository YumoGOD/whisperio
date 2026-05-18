# Пошаговый план реализации

Выполняй задачи **строго по порядку**. После каждой — остановись, сообщи что сделано, дождись подтверждения от пользователя перед следующей.

## Задача 1: Скелет проекта + .env

- Создай структуру папок согласно `02_ARCHITECTURE.md`
- Создай `.env.example` со всеми ключами (см. ниже список)
- Создай `.gitignore` (Python, Node, .env, ./storage/, *.sqlite)
- Создай `backend/pyproject.toml` (или `requirements.txt`) со всеми зависимостями
- Создай `backend/app/config.py` с Pydantic Settings v2, читающим .env

**Ключи .env (минимум):**
```
# Backend
HOST=0.0.0.0
PORT=8000
STORAGE_DIR=./storage
DB_URL=sqlite:///./storage/db.sqlite
MAX_UPLOAD_SIZE_MB=2048
ALLOWED_EXTENSIONS=mp3,wav,m4a,ogg,flac,mp4,mov,mkv,webm,avi
LOG_LEVEL=INFO
CORS_ORIGINS=http://localhost:5173

# Whisper
WHISPER_MODEL=large-v3
WHISPER_DEVICE=cuda
WHISPER_COMPUTE_TYPE=float16
WHISPER_BATCH_SIZE=8
WHISPER_MODELS_DIR=./storage/models
DEFAULT_LANGUAGE=
BEAM_SIZE=5

# VAD
VAD_ENABLED=true
VAD_MIN_SILENCE_MS=500
VAD_SPEECH_PAD_MS=200
VAD_MAX_SEGMENT_SEC=30

# Worker
WORKER_POLL_INTERVAL_SEC=2
WORKER_HEARTBEAT_SEC=5
STALE_JOB_TIMEOUT_MIN=60

# Jobs
MAX_JOBS_IN_LIST=100
```

**Зависимости backend (зафиксируй версии):**
- fastapi, uvicorn[standard], python-multipart
- sqlalchemy>=2.0, pydantic>=2, pydantic-settings>=2
- faster-whisper>=1.0, ffmpeg-python, silero-vad (через torch.hub или pip)
- torch (с CUDA 12.x — указать индекс PyTorch)

**Зависимости frontend:**
- react, react-dom, react-router-dom
- vite, typescript, @types/react
- tailwindcss, postcss, autoprefixer
- shadcn/ui (минимум: button, card, input, select, progress, table, toast)
- lucide-react (иконки)

⏹ **СТОП. Покажи структуру, дождись подтверждения.**

---

## Задача 2: БД, модели, базовая инфраструктура

- `backend/app/db.py`: SQLAlchemy engine с `connect_args={"check_same_thread": False}`, WAL pragma, SessionLocal
- `backend/app/models.py`: модель `Job` со всеми полями из `02_ARCHITECTURE.md`. Добавь модель `WorkerHeartbeat` (id=1, last_seen)
- `backend/app/storage.py`: функции `ensure_dirs()`, `get_input_path(job_id)`, `get_result_dir(job_id)`
- В `main.py`: на старте вызывай `Base.metadata.create_all` и `ensure_dirs()`

⏹ **СТОП. Запусти `python -c "from app.db import engine; from app.models import Base; Base.metadata.create_all(engine)"` — должно создать sqlite-файл без ошибок.**

---

## Задача 3: API эндпоинты (без ML)

Реализуй все эндпоинты из `03_API_CONTRACT.md`:
- POST /api/jobs (валидация расширения, сохранение файла, запись в БД, ответ)
- GET /api/jobs, GET /api/jobs/{id}
- GET /api/jobs/{id}/download
- DELETE /api/jobs/{id}
- GET /api/health

CORS из настроек. Тайпинг через Pydantic-схемы в `schemas.py`.

⏹ **СТОП. Через curl/HTTPie проверь все эндпоинты на корректность (без воркера статус будет вечно queued — это ок).**

---

## Задача 4: Audio preprocessing

`backend/app/pipeline/audio.py`:
- `probe_duration(path) -> float` — через ffprobe
- `to_wav_16k_mono(input_path, output_path)` — через ffmpeg-python

`backend/app/pipeline/vad.py`:
- Загрузка Silero VAD один раз (lazy singleton)
- `split_into_segments(wav_path) -> list[(start_sec, end_sec)]` с учётом параметров из .env
- Если `VAD_ENABLED=false` — возвращай один сегмент на весь файл

Без интеграции с воркером пока — только модули.

⏹ **СТОП. Напиши маленький smoke-тест: возьми любой mp3, конвертни, прогони VAD, выведи сегменты.**

---

## Задача 5: Whisper transcribe

`backend/app/pipeline/transcribe.py`:
- Класс `Transcriber` с lazy-init faster-whisper модели (по `settings.WHISPER_MODEL`)
- Метод `transcribe_segments(wav_path, segments, language) -> list[Segment]`
- Используй `BatchedInferencePipeline` если доступен в установленной версии faster-whisper; иначе — обычный `model.transcribe` с временной нарезкой
- Возвращай список `{start, end, text}` с абсолютными таймкодами (offset от начала исходника)

`backend/app/pipeline/formats.py`:
- `to_txt(segments) -> str`
- `to_srt(segments) -> str`
- `to_json(segments, meta) -> str`

⏹ **СТОП. Прогоните `python -c "..."` с тестовым файлом — выведите первые 3 сегмента.**

---

## Задача 6: Worker

`backend/app/worker.py`:
- Скрипт `python -m app.worker`
- На старте: лог GPU info, загрузка модели, проверка CUDA, восстановление зависших задач (status=processing + старше STALE_JOB_TIMEOUT_MIN → queued)
- Бесконечный цикл:
  - Обнови heartbeat
  - Возьми одну задачу `queued` (атомарно: `UPDATE ... WHERE status='queued' ORDER BY created_at LIMIT 1 RETURNING ...`)
  - Если есть — обработай:
    1. status=preprocessing, ffmpeg → wav, ffprobe duration
    2. VAD сегментация
    3. status=transcribing, прогоняй батчами, обновляй progress (0..1) после каждого батча
    4. Сохрани файлы результата (txt/srt/json) в `./storage/results/<id>/`
    5. Сохрани segments + transcript_text в БД, status=done
  - При любом исключении: status=failed, error=str(e), continue
  - Если задач нет — sleep WORKER_POLL_INTERVAL_SEC

⏹ **СТОП. Запусти api и worker параллельно, загрузи через curl небольшой файл, проверь весь flow до status=done.**

---

## Задача 7: Frontend — каркас

- `npm create vite@latest frontend -- --template react-ts`
- Установи Tailwind + shadcn/ui
- Настрой `vite.config.ts` с proxy `/api` → `http://localhost:8000`
- Роутинг: `/` (Home — загрузка), `/history` (History), `/jobs/:id` (Detail)
- Общий layout с шапкой: лого/название, переключение тем (light/dark), индикатор статуса воркера (зелёная/красная точка из /api/health)
- `src/api.ts` — типизированные функции для всех эндпоинтов

⏹ **СТОП. `npm run dev` — должен открыться пустой layout с роутами.**

---

## Задача 8: Frontend — Home + Upload

- `UploadZone.tsx`: drag & drop + кнопка выбора, поддержка множественной загрузки
- Селекторы: модель, язык, чекбокс диаризации (disabled)
- При сабмите: для каждого файла POST /api/jobs, добавляй карточку в локальный список
- `JobCard.tsx`: имя, статус (цветной badge), прогресс-бар если processing/transcribing, время
- Polling: для незавершённых задач — GET /api/jobs/{id} каждые 2 сек
- Кнопка «Открыть» → переход на /jobs/:id

⏹ **СТОП. Проверь полный сценарий загрузки.**

---

## Задача 9: Frontend — Detail + History

- `TranscriptView.tsx`: блок с сегментами (таймкод слева моноширинно, текст справа), клик по таймкоду — выделяет сегмент
- Кнопки: «Копировать», «Скачать TXT/SRT/JSON»
- Метаданные: модель, длительность, язык, время обработки
- History: таблица из shadcn/ui с пагинацией (или просто 100 последних), фильтр по статусу, кнопка удаления с подтверждением

⏹ **СТОП. Финальный smoke-test — пользовательский сценарий от начала до конца.**

---

## Задача 10: Полировка и деплой

- README.md с инструкцией: `cp .env.example .env`, `pip install`, `npm install`, как запускать api + worker + frontend
- Опционально: `docker-compose.yml` с двумя сервисами (api+worker) с пробросом GPU (`runtime: nvidia`)
- Опционально: systemd-юниты как комментарии в README
- Проверка, что при `WHISPER_DEVICE=cpu` всё тоже работает (медленнее, но работает) — для dev без GPU

⏹ **ГОТОВО.**
