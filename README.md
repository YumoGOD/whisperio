# WhisperIO

Приложение для одной задачи: **надежная транскрибация длинных (1-4 часа) аудио/видео файлов**, включая шумные и тихие записи.

## Что изменено в quality-first рефакторинге

- Убрана избыточная конфигурация уровня `WHISPER_*` из пользовательского потока.
- Вынесен модульный pipeline:
  - `backend/app/transcription/settings.py`
  - `backend/app/transcription/preprocess.py`
  - `backend/app/transcription/segmenter.py`
  - `backend/app/transcription/inference.py`
  - `backend/app/transcription/quality.py`
  - `backend/app/transcription/pipeline.py`
- Воркеры теперь используют `backend/app/transcription_worker.py` как тонкий оркестратор.
- Full-file fallback заменен на **segment-level rescue** (повторная транскрибация только проблемных участков).

## Pipeline обработки

1. `ffprobe` определяет длительность.
2. `ffmpeg` делает mono/16kHz PCM и применяет профильную предобработку:
   - `highpass`
   - `lowpass`
   - `afftdn` (noise reduction)
   - `dynaudnorm` (динамическое выравнивание громкости)
3. VAD Cut & Merge сегментирует речь в окна около 30 секунд.
4. Первичный проход транскрибации (`condition_on_previous_text=false`).
5. Quality-анализ на уровне сегментов.
6. Segment-level rescue только для низкокачественных сегментов.
7. Сохранение финальных сегментов и метрик задачи.

## Минимальная конфигурация

### Качество/поведение

- `TRANSCRIBE_PROFILE` — профиль обработки (`robust_long_noisy` по умолчанию, альтернативно `balanced`)
- `TRANSCRIBE_LANGUAGE` — язык (`ru` или `auto`)

### Модель/железо

- `WHISPER_MODEL_SIZE` (по умолчанию `large-v3`)
- `WHISPER_DEVICE` (`cpu`/`cuda`)
- `WHISPER_COMPUTE_TYPE` (например `int8`, `float16`)
- `WHISPER_CPU_THREADS`
- `WHISPER_WORKER_PROCESSES`

### Операционные

- `MAX_UPLOAD_SIZE_MB`
- `DATABASE_PATH`
- `UPLOAD_DIR`
- `LOG_LEVEL`

Остальные параметры теперь внутренние и задаются profile-presets.

## Профили

- `robust_long_noisy` — для плохого качества речи, шума и длинных файлов.
- `balanced` — более быстрый и менее агрессивный по rescue.

## Запуск

```bash
docker compose up --build
```

- Backend API: `http://localhost:8000`
- Frontend: `http://localhost:3000`

## Benchmark

Скрипт нагрузочного прогона:

```bash
python backend/scripts/benchmark_matrix.py --files /path/a.wav /path/b.wav --parallel 2 --repeat 1 --profile-name robust_long_noisy --out benchmark-results.json
```

Отчет сохраняет `processing_rtf`, `effective_speed_x`, статусы и `quality_flags`.

## API (кратко)

- `POST /api/jobs` — загрузить файл
- `GET /api/jobs` — список задач
- `GET /api/jobs/{job_id}` — детали + сегменты
- `DELETE /api/jobs/{job_id}` — удалить задачу
- `GET /api/jobs/{job_id}/audio?variant=original|prepared` — скачать аудио
