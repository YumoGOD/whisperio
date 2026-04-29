# WhisperIO: настройка через docker-compose

Проект использует `faster-whisper` на CPU и полностью настраивается через `docker-compose.yml`.

## Текущий baseline (из compose)

- Модель: `WHISPER_MODEL_SIZE=medium`
- Язык: `WHISPER_LANGUAGE=ru`, режим `WHISPER_TASK=transcribe`
- Приоритет скорости: `WHISPER_BEAM_SIZE=1`, `WHISPER_BEST_OF=1`, `WHISPER_TEMPERATURE=0.0`
- VAD и спец-теги включены (`WHISPER_VAD_FILTER=true`, `WHISPER_ENABLE_TAGS=true`)

## Где менять качество и скорость

Все параметры вынесены в `docker-compose.yml`:

- **Модель/железо:** `WHISPER_MODEL_SIZE`, `WHISPER_COMPUTE_TYPE`, `WHISPER_CPU_THREADS`
- **Параллелизм:**
  - `WHISPER_WORKERS` — число async-воркеров очереди (pipeline задач)
  - `WHISPER_MODEL_WORKERS` — `num_workers` CTranslate2 внутри `WhisperModel`
- **Качество распознавания:** `WHISPER_BEAM_SIZE`, `WHISPER_BEST_OF`, `WHISPER_CONDITION_ON_PREVIOUS_TEXT`, `WHISPER_TEMPERATURE`
- **Сегментация/VAD:** `WHISPER_VAD_*`, `WHISPER_NO_SPEECH_THRESHOLD`, `WHISPER_LOG_PROB_THRESHOLD`, `WHISPER_COMPRESSION_RATIO_THRESHOLD`
- **Метки тишины/музыки/неразборчивости:** `WHISPER_ENABLE_TAGS` (общий вкл/выкл), `WHISPER_TAG_*` (пороги при включенных тегах)

## Как влияют ключевые параметры

- `WHISPER_BEAM_SIZE` и `WHISPER_BEST_OF`:
  - выше -> лучше точность, медленнее обработка
  - ниже -> быстрее, больше ошибок
- `WHISPER_COMPUTE_TYPE`:
  - `int8` -> быстрее и меньше RAM, чем float
- `WHISPER_WORKERS`:
  - повышает параллелизм этапов очереди (чтение/подготовка/сохранение)
  - не гарантирует ускорение одного `transcribe` вызова
- `WHISPER_MODEL_WORKERS` и `WHISPER_CPU_THREADS`:
  - управляют параллелизмом CTranslate2 и загрузкой CPU на инференсе
  - слишком высокие значения могут дать конкуренцию потоков

## Запуск

```bash
docker compose up --build
```

Backend API: `http://localhost:8000`  
Frontend: `http://localhost:3000`

## Быстрая подстройка от текущего baseline

- Нужно быстрее: оставьте `1/1`, уменьшите `WHISPER_MODEL_SIZE` (например, до `small`) и/или снизьте `WHISPER_CPU_THREADS`.
- Нужно точнее: повышайте `WHISPER_BEAM_SIZE` и `WHISPER_BEST_OF` (например, `3/3` -> `5/5`) и при необходимости переключайтесь на более крупную модель.
- Слишком много `<Неразборчиво>`: ослабьте пороги `WHISPER_TAG_UNINTELLIGIBLE_*`.
- Слишком мало `<Музыка>`: немного увеличьте `WHISPER_TAG_MUSIC_MAX_ZCR` и `WHISPER_TAG_MUSIC_MAX_ENERGY_VARIATION`.

## Совместимость версии faster-whisper

Параметр `WHISPER_CHUNK_LENGTH_S` применяется только если версия `faster-whisper` поддерживает аргумент `chunk_length` в `transcribe`. Проверка выполняется в рантайме через introspection сигнатуры метода.
