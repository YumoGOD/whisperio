# whisperio2

Локальный транскрибатор длинных и зашумлённых аудиозаписей с временными метками. Стек: FastAPI + SQLite + ffmpeg + faster-whisper. Без Redis, Celery, PostgreSQL и облаков.

**Требуется NVIDIA GPU** — работает только на CUDA, CPU-режима нет.

---

## Prerequisites

| Требование | Подробности |
|---|---|
| NVIDIA GPU | VRAM ≥ 10 GB для `large-v3`; 6 GB для `medium` |
| NVIDIA Driver | Актуальный (≥ 525 рекомендовано) |
| NVIDIA Container Toolkit | [Инструкция](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) |
| Docker + Compose | Docker Desktop (Windows/Mac) или Docker Engine (Linux) |
| Windows | Docker Desktop с WSL2 и GPU-поддержкой в дистрибутиве |

---

## Quick Start

```bash
cp .env.example .env
docker compose up --build
```

Откройте `http://localhost:8000`, загрузите файл и наблюдайте за очередью.

**Первый запуск** скачивает модель в `./data/models` — это медленно. Для offline-окружений скачайте модель заранее:

```bash
docker compose run --rm worker python scripts/download_model.py large-v3 \
  --output-dir /app/data/models/faster-whisper-large-v3
```

Затем в `.env`:

```env
WHISPER_MODEL=/app/data/models/faster-whisper-large-v3
```

Проверить, что воркер видит GPU:

```bash
docker compose run --rm worker nvidia-smi
```

---

## Архитектура

Два сервиса в Compose:

- **api** — FastAPI, порт 8000, GPU не нужен
- **worker** — обрабатывает очередь заданий, требует `gpus: all`

Оба монтируют `./data` как `/app/data` — общее хранилище для загрузок, артефактов и БД.

---

## Web UI

`http://localhost:8000` — полный интерфейс управления:

- Загрузка файла с контекстом и глоссарием
- Список заданий с прогрессом в реальном времени
- Аудиоплеер с кликабельными временными метками сегментов
- Поиск по тексту транскрипта
- Скачивание в любом формате

---

## REST API

| Метод | Путь | Описание |
|---|---|---|
| `POST` | `/api/jobs` | Создать задание (загрузка файла) |
| `GET` | `/api/jobs` | Список заданий с пагинацией |
| `GET` | `/api/jobs/{id}` | Статус и результат задания |
| `GET` | `/api/jobs/{id}/audio` | Стриминг исходного аудио |
| `GET` | `/api/jobs/{id}/download/{fmt}` | Скачать транскрипт |
| `GET` | `/api/profiles` | Список профилей транскрипции |
| `GET` | `/api/health` | Проверка работоспособности |

Форматы скачивания `{fmt}`: `txt`, `json`, `srt`, `vtt`, `docx`.

**Примеры:**

```bash
# Создать задание
curl -F "file=@lecture.mp3" -F "profile=accuracy_first" http://localhost:8000/api/jobs

# Проверить статус
curl http://localhost:8000/api/jobs/<job_id>

# Скачать SRT
curl -OJ http://localhost:8000/api/jobs/<job_id>/download/srt

# Скачать DOCX
curl -OJ http://localhost:8000/api/jobs/<job_id>/download/docx
```

---

## Configuration

Все настройки через `.env` (скопируйте из `.env.example`).

### App

| Переменная | По умолчанию | Описание |
|---|---|---|
| `APP_NAME` | `Локальный транскрибатор` | Название приложения |
| `APP_ENV` | `local` | Окружение |
| `APP_HOST` | `0.0.0.0` | Адрес FastAPI |
| `APP_PORT` | `8000` | Порт FastAPI |
| `MAX_UPLOAD_MB` | `5120` | Максимальный размер файла (МБ) |
| `ALLOWED_EXTENSIONS` | `*` | Допустимые расширения; `*` = любые (ffmpeg — реальный фильтр) |

### Whisper

| Переменная | По умолчанию | Описание |
|---|---|---|
| `WHISPER_MODEL` | `large-v3` | Имя модели или путь к локальной CTranslate2-директории |
| `WHISPER_DEVICE` | `cuda` | Устройство — только `cuda` |
| `WHISPER_COMPUTE_TYPE` | `float16` | `float16` — стандарт; `int8_float16` — меньше VRAM |
| `WHISPER_NUM_WORKERS` | `1` | Потоки загрузки данных внутри WhisperModel |
| `WHISPER_DOWNLOAD_ROOT` | `/app/data/models` | Кэш моделей |
| `WHISPER_LANGUAGE` | `ru` | ISO-код языка; пустая строка = автодетект |
| `WHISPER_TASK` | `transcribe` | `transcribe` или `translate` (в EN) |
| `DEFAULT_PROFILE` | `accuracy_first` | Профиль по умолчанию |

`WHISPER_LANGUAGE=ru` рекомендуется для длинных русских записей — предотвращает случайное переключение языка в чанках.

### Worker

| Переменная | По умолчанию | Описание |
|---|---|---|
| `WORKER_CONCURRENCY` | `1` | Параллельных заданий; на одной GPU обычно `1` |
| `WORKER_POLL_SECONDS` | `5` | Интервал опроса очереди (с) |
| `WORKER_STALE_RUNNING_MINUTES` | `30` | Таймаут зависших заданий до возврата в `pending` |
| `WORKER_ID` | _(пусто)_ | ID воркера; auto-генерируется если не задан |

### Audio

| Переменная | По умолчанию | Описание |
|---|---|---|
| `CHUNK_SECONDS` | `1800` | Длина чанка (с); 1800 = 30 мин |
| `CHUNK_OVERLAP_SECONDS` | `15` | Перекрытие чанков для исключения потерь на границах |
| `ENABLE_LOUDNORM` | `true` | Нормализация громкости через ffmpeg loudnorm |
| `TARGET_SAMPLE_RATE` | `16000` | Частота дискретизации (Гц) |

### VAD (Voice Activity Detection)

Silero VAD встроен в faster-whisper. По умолчанию **выключен** — на записях с постоянным фоновым шумом VAD может срезать тихую речь. Используйте `ENABLE_LOUDNORM=true` вместо VAD для плохого аудио.

| Переменная | По умолчанию | Описание |
|---|---|---|
| `VAD_FILTER` | `false` | Включить VAD-фильтрацию |
| `VAD_THRESHOLD` | `0.35` | Порог чувствительности (ниже = чувствительнее; Silero default = 0.5) |
| `VAD_MIN_SILENCE_MS` | `1200` | Минимальная пауза для разреза (мс) |
| `VAD_SPEECH_PAD_MS` | `600` | Паддинг вокруг речевых сегментов (мс) |

Рекомендуемые настройки при `VAD_FILTER=true` для тихой речи: `VAD_THRESHOLD=0.25`, `VAD_SPEECH_PAD_MS=800`.

### Glossary

| Переменная | По умолчанию | Описание |
|---|---|---|
| `GLOSSARY_PATH` | `/app/data/glossary/global.yml` | Путь к глобальному глоссарию |
| `GLOSSARY_PROMPT_MAX_CHARS` | `700` | Максимальный размер промпта для Whisper (символов) |
| `GLOSSARY_CONTEXT_MAX_CHARS` | `1200` | Максимальный размер контекстного блока (символов) |
| `GLOSSARY_HOTWORDS_MAX` | `80` | Максимальное число hotwords |
| `GLOSSARY_ENABLE_HOTWORDS` | `false` | Агрессивный boost терминов (может вызывать повторения) |
| `GLOSSARY_ENABLE_HARD_NORMALIZATION` | `true` | Применять regex-замены после распознавания |
| `GLOSSARY_REPETITION_COMPRESSION_THRESHOLD` | `4.0` | Порог для детекции галлюцинаций глоссария |

---

## Glossary System

Глоссарий улучшает распознавание корпоративных терминов, имён и аббревиатур.

### Режимы терминов

- **soft** — термин попадает в промпт Whisper, текст не модифицируется.
- **hard** — термин в промпте + текст нормализуется regex-заменами после распознавания.

### Глобальный глоссарий (`./data/glossary/global.yml`)

```yaml
terms:
  - canonical: "Bauer"
    category: "brand"
    mode: "hard"
    spoken_forms:
      - "Бауэр"
      - "Бауер"
    description: "Bauer — название компании, произносится как Бауэр."
    replacements:
      - from: "\\b[Бб]ауэр\\b"
        to: "Bauer"
      - from: "\\b[Бб]ауер\\b"
        to: "Bauer"
```

Поддерживаемые категории: `brand`, `product`, `person`, `abbreviation`, `department`.

### Контекст задания

При создании задания можно передать дополнительные поля:

| Поле | Описание |
|---|---|
| `audio_type` | Тип записи: `лекция`, `планёрка`, `презентация` |
| `audio_context` | Краткое описание содержимого |
| `expected_content` | Ожидаемые темы и структура |
| `dynamic_terms` | Термины только для этого задания (см. формат ниже) |

### Динамические термины

```text
Bauer Juicer Pro | hard | Бауэр джусер про, джусер про, соковыжималка Бауэр
Bauer Lounge Massage | hard | Бауэр лаундж массаж, лаундж мессаж, массажная накидка
Дмитрий Канищев | soft | Канищев, Дмитрий Канищев
```

Формат: `Каноническая форма | режим | spoken_form1, spoken_form2, ...`

### Диагностика

`./data/transcripts/<job_id>/diagnostics.json` сохраняет: промпт, подобранные термины, счётчики замен, сброшенные сегменты-галлюцинации.

---

## Profiles

| Параметр | `accuracy_first` | Описание |
|---|---|---|
| `beam_size` | 5 | Размер луча поиска |
| `best_of` | 5 | Кандидатов на выбор |
| `vad_filter` | false | VAD выключен — сохраняет тихую речь |
| `condition_on_previous_text` | false | Предотвращает каскадные ошибки на длинных записях |
| `no_speech_threshold` | 0.60 | Порог детекции тишины |
| `temperature` | [0.0, 0.2, 0.4, 0.6] | Несколько температур для выборки |

`accuracy_first` — профиль по умолчанию, оптимизирован для шумных русскоязычных лекций (приоритет — полнота, а не скорость).

VAD можно переопределить глобально через `VAD_*` переменные в `.env` независимо от профиля.

---

## Benchmark

```bash
docker compose run --rm worker python scripts/benchmark.py /app/data/uploads/example.mp3 --profile accuracy_first
```

Выводит: длительность аудио, время обработки и RTF (real-time factor). RTF < 1.0 означает быстрее реального времени.

---

## File Artifacts

| Путь | Содержимое |
|---|---|
| `./data/uploads/<job_id>_<filename>` | Исходный загруженный файл |
| `./data/work/<job_id>/prepared_16k_mono.wav` | Подготовленный WAV (16kHz, mono) |
| `./data/work/<job_id>/chunks/` | Нарезанные чанки |
| `./data/transcripts/<job_id>/<job_id>.txt` | Полный текст |
| `./data/transcripts/<job_id>/<job_id>.json` | Сегменты с временными метками и уверенностью |
| `./data/transcripts/<job_id>/<job_id>.srt` | SubRip субтитры |
| `./data/transcripts/<job_id>/<job_id>.vtt` | WebVTT субтитры |
| `./data/transcripts/<job_id>/<job_id>.docx` | Word-документ с временными метками |
| `./data/transcripts/<job_id>/diagnostics.json` | Диагностика глоссария и пайплайна |
| `./data/logs/app.log` | Лог приложения |

Артефакты намеренно сохраняются — они позволяют выяснить, где именно была потеряна речь.

---

## Tips

**Выбор GPU** (при нескольких картах):
```env
CUDA_VISIBLE_DEVICES=1
```

**Снизить VRAM** (если не хватает для `float16`):
```env
WHISPER_COMPUTE_TYPE=int8_float16
```

**Hotwords** — включать только после тестирования на коротком фрагменте; на зашумлённых записях могут вызывать повторения терминов в начале чанков:
```env
GLOSSARY_ENABLE_HOTWORDS=true
```

**SQLite** работает в WAL-режиме с короткими write-транзакциями — достаточно для локального API + воркер. При необходимости масштабирования можно добавить несколько воркер-сервисов (осторожно с конкурентным доступом к GPU).
