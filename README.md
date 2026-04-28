# WhisperIO: accuracy-first профиль для русского аудио

Этот проект использует `faster-whisper` и настроен под распознавание русской речи на `large-v3` с CPU.

## Рекомендованный профиль для вашего сервера

Конфигурация в `docker-compose.yml` уже выставлена под:
- `WHISPER_MODEL_SIZE=large-v3`
- `WHISPER_DEVICE=cpu`
- `WHISPER_COMPUTE_TYPE=int8`
- `WHISPER_WORKERS=2`
- `WHISPER_CPU_THREADS=4`
- `OMP_NUM_THREADS=8`
- `MKL_NUM_THREADS=8`

Это дает 2 параллельных backend-воркера без агрессивного oversubscription на 8 ядрах.

## Метки неречевых участков

В итоговые сегменты добавляются специальные теги:
- `<Тишина>`: для пауз длительностью от `WHISPER_TAG_SILENCE_MIN_SEC` (по умолчанию `5.0` сек).
- `<Музыка>`: для длинных неречевых интервалов, похожих на музыку по аудио-метрикам.
- `<Неразборчиво>`: когда модель видит речь, но сегмент имеет плохие метрики разборчивости.

Теги добавляются отдельными сегментами и не заменяют нормальные речевые фрагменты.

## Параметры точности

### Декодирование Whisper
- `WHISPER_BEAM_SIZE` (по умолчанию `5`)
- `WHISPER_BEST_OF` (по умолчанию `5`)
- `WHISPER_CONDITION_ON_PREVIOUS_TEXT` (по умолчанию `true`)
- `WHISPER_NO_SPEECH_THRESHOLD` (по умолчанию `0.4`)
- `WHISPER_LOG_PROB_THRESHOLD` (по умолчанию `-1.0`)
- `WHISPER_COMPRESSION_RATIO_THRESHOLD` (по умолчанию `2.4`)

### VAD
- `WHISPER_VAD_FILTER` (`true`)
- `WHISPER_VAD_THRESHOLD` (`0.5`)
- `WHISPER_VAD_MIN_SILENCE_DURATION_MS` (`500`)
- `WHISPER_VAD_SPEECH_PAD_MS` (`400`)

### Пороги тегов
- `WHISPER_TAG_SILENCE_MIN_SEC` (`5.0`)
- `WHISPER_TAG_SILENCE_DBFS` (`-38.0`)
- `WHISPER_TAG_MUSIC_MIN_SEC` (`3.0`)
- `WHISPER_TAG_MUSIC_MAX_ZCR` (`0.08`)
- `WHISPER_TAG_MUSIC_MAX_ENERGY_VARIATION` (`0.35`)
- `WHISPER_TAG_UNINTELLIGIBLE_MAX_AVG_LOGPROB` (`-1.15`)
- `WHISPER_TAG_UNINTELLIGIBLE_MIN_NO_SPEECH_PROB` (`0.55`)
- `WHISPER_TAG_UNINTELLIGIBLE_MAX_COMPRESSION_RATIO` (`2.4`)

## Запуск

```bash
docker compose up --build
```

API backend: `http://localhost:8000`  
Frontend: `http://localhost:3000`

## Практика тонкой подстройки

- Если слишком много `<Неразборчиво>`, сначала ослабьте:
  - `WHISPER_TAG_UNINTELLIGIBLE_MIN_NO_SPEECH_PROB` (ниже),
  - `WHISPER_TAG_UNINTELLIGIBLE_MAX_AVG_LOGPROB` (ниже по модулю, например `-1.25`).
- Если `<Музыка>` срабатывает редко:
  - увеличьте `WHISPER_TAG_MUSIC_MAX_ZCR`,
  - увеличьте `WHISPER_TAG_MUSIC_MAX_ENERGY_VARIATION`.
- Если в системе высокий CPU wait и очередь растет:
  - уменьшите `WHISPER_BEAM_SIZE` / `WHISPER_BEST_OF` до `4` или `3`,
  - оставьте `WHISPER_WORKERS=2`, но уменьшите `WHISPER_CPU_THREADS` до `3`.
