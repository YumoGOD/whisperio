# WhisperIO: настройка через docker-compose

Проект использует `faster-whisper` с моделью `large-v3` на CPU и управляется через `docker-compose.yml`.

## Что настроено

- 2 backend-воркера (`WHISPER_WORKERS=2`)
- Русский язык (`WHISPER_LANGUAGE=ru`)
- Приоритет точности (`beam=5`, `best_of=5`, `condition_on_previous_text=true`)
- Спец-теги в сегментах: `<Тишина>`, `<Музыка>`, `<Неразборчиво>`

## Где менять качество и скорость

Все параметры вынесены в `docker-compose.yml` и подписаны комментариями:

- **Модель/железо:** `WHISPER_MODEL_SIZE`, `WHISPER_COMPUTE_TYPE`, `WHISPER_CPU_THREADS`, `WHISPER_WORKERS`
- **Качество распознавания:** `WHISPER_BEAM_SIZE`, `WHISPER_BEST_OF`, `WHISPER_CONDITION_ON_PREVIOUS_TEXT`
- **Сегментация/VAD:** `WHISPER_VAD_*`, `WHISPER_NO_SPEECH_THRESHOLD`, `WHISPER_LOG_PROB_THRESHOLD`
- **Метки тишины/музыки/неразборчивости:** `WHISPER_TAG_*`

## Как влияют ключевые параметры

- `WHISPER_BEAM_SIZE` и `WHISPER_BEST_OF`:
  - выше -> лучше точность, медленнее обработка
  - ниже -> быстрее, больше ошибок
- `WHISPER_COMPUTE_TYPE`:
  - `int8` -> быстрее и меньше RAM, чем float
- `WHISPER_WORKERS`:
  - больше воркеров -> выше throughput очереди
  - один файл быстрее не станет, но несколько файлов обрабатываются параллельно
- `WHISPER_CPU_THREADS`:
  - выше -> может ускорить один инференс до упора CPU
  - слишком высоко -> конкуренция потоков и нестабильная производительность

## Запуск

```bash
docker compose up --build
```

Backend API: `http://localhost:8000`  
Frontend: `http://localhost:3000`

## Быстрая практическая подстройка

- Нужно быстрее: снизьте `WHISPER_BEAM_SIZE` и `WHISPER_BEST_OF` до `4` или `3`.
- Нужно точнее: верните `5/5`, при необходимости уменьшите `WHISPER_WORKERS` до `1`.
- Слишком много `<Неразборчиво>`: ослабьте пороги `WHISPER_TAG_UNINTELLIGIBLE_*`.
- Слишком мало `<Музыка>`: немного увеличьте `WHISPER_TAG_MUSIC_MAX_ZCR` и `WHISPER_TAG_MUSIC_MAX_ENERGY_VARIATION`.
