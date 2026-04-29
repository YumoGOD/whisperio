# WhisperIO: CPU-only runtime под high-throughput

Проект использует `faster-whisper` + `CTranslate2` в режиме CPU-only и рассчитан на стабильную параллельную транскрибацию.

## Что изменилось в архитектуре

- Очередь задач стала DB-backed (`SQLite`) вместо in-memory очереди.
- Диспетчер запускает несколько worker-процессов (`WHISPER_WORKER_PROCESSES`).
- Каждый worker-процесс держит свой экземпляр `WhisperModel` (реальный параллельный inference).
- При старте сервиса незавершенные `processing` задачи возвращаются в `queued`.

Это убирает глобальный bottleneck "одна модель + lock на весь сервис".

## Рекомендованный профиль для сервера 64 cores / 128 GB RAM

Базовый профиль в `docker-compose.yml`:

- `WHISPER_MODEL_SIZE=large-v3`
- `WHISPER_COMPUTE_TYPE=int8`
- `WHISPER_WORKER_PROCESSES=5`
- `WHISPER_CPU_THREADS=10`
- `OMP_NUM_THREADS=10`
- `MKL_NUM_THREADS=10`
- `WHISPER_BEAM_SIZE=2`
- `WHISPER_BEST_OF=2`

Идея профиля: accuracy-first внутри SLA (`1h -> 10-15 min`) при 4-5 параллельных файлах.

## Ключевые переменные

### Параллелизм

- `WHISPER_WORKER_PROCESSES` — количество независимых worker-процессов.
- `WHISPER_DISPATCH_POLL_MS` — частота claim задач из БД.
- `WHISPER_CPU_THREADS` — потоки CTranslate2 на процесс.
- `WHISPER_MODEL_WORKERS` — внутренние worker'ы CTranslate2 (для CPU обычно `1`).

### Бюджет потоков (важно для стабильности)

- `OMP_NUM_THREADS`
- `MKL_NUM_THREADS`
- `WHISPER_PROCESS_OMP_THREADS`
- `WHISPER_PROCESS_MKL_THREADS`

Рекомендуется держать их согласованными с `WHISPER_CPU_THREADS`, чтобы избежать oversubscription.

### Декодер и качество

- `WHISPER_BEAM_SIZE`, `WHISPER_BEST_OF` — рост качества ценой времени.
- `WHISPER_TEMPERATURE` — для детерминизма обычно `0.0`.
- `WHISPER_CONDITION_ON_PREVIOUS_TEXT` — связность текста, небольшой overhead.
- `WHISPER_ENABLE_QUALITY_FALLBACK` — повторный проход только при аномалиях.

### VAD и сегментация

- `WHISPER_VAD_FILTER`, `WHISPER_VAD_*`
- `WHISPER_NO_SPEECH_THRESHOLD`
- `WHISPER_LOG_PROB_THRESHOLD`
- `WHISPER_COMPRESSION_RATIO_THRESHOLD`

## Компромиссные профили

- **Accuracy-first (текущий по умолчанию):**
  - `large-v3`, `int8`, `beam=2`, `best_of=2`.
- **Balanced:**
  - `large-v3`, `int8`, `beam=1`, `best_of=1`.
- **SLA-peak (в пиковую нагрузку):**
  - `medium`, `int8`, `beam=1`, `best_of=1`.

## Runbook: калибровка под SLA

1. Запустите систему:
   ```bash
   docker compose up --build
   ```
2. Подготовьте набор из 5 файлов длительностью 45-60 минут.
3. Выполните 3-4 прогона с разными профилями:
   - `beam/best_of`: `2/2`, `3/3`, `1/1`
   - `worker_processes x cpu_threads`: `5x10`, `4x12`, `6x8`
4. Фиксируйте по каждому прогону:
   - `p95 processing_duration_ms`
   - `transcribe_duration_ms`
   - долю fallback-проходов из `quality_flags`
   - прокси качества на контрольном наборе
5. Зафиксируйте профиль, который укладывается в SLA и дает лучшее качество.

Для автоматизации прогона можно использовать утилиту:

```bash
python backend/scripts/benchmark_matrix.py --files /path/a.wav /path/b.wav /path/c.wav /path/d.wav /path/e.wav --parallel 5 --out benchmark-results.json
```

## Метрики приемки

- `p95 <= 15 min` на `1 hour` аудио.
- Стабильная обработка 4-5 файлов параллельно.
- Нет падений worker-процессов и ошибок конкурентной записи в БД.

## Запуск

```bash
docker compose up --build
```

Backend API: `http://localhost:8000`  
Frontend: `http://localhost:3000`

## Примечание про версию faster-whisper

`WHISPER_CHUNK_LENGTH_S` применяется только если текущая версия `faster-whisper` поддерживает аргумент `chunk_length`.

## Логирование и диагностика нагрузки

Сервис пишет структурированные JSON-логи в stdout. По умолчанию включены:

- lifecycle API-запроса (`api_request_started`, `api_request_finished`);
- lifecycle job (`job_created`, `job_claimed`, `job_dispatched`, `job_completed`, `job_failed`);
- stage telemetry (`stage_started`, `stage_finished`, `stage_failed`);
- runtime snapshots (`runtime_snapshot`);
- worker resource snapshots (`worker_resource_snapshot`, `resource_pressure_warning`);
- SLA diagnostics (`sla_drift_detected`).

### Обязательные поля в логах

- `ts`, `level`, `event`, `service`, `component`, `schema_version`, `logger`;
- correlation-поля: `request_id`, `job_id`, `worker_pid`;
- performance-поля (по контексту): `duration_ms`, `preprocess_duration_ms`, `transcribe_duration_ms`, `rtf`, `effective_speed_x`, `queue_backlog`.

### Переменные окружения для логов

- `LOG_FORMAT` (`json` или `text`);
- `LOG_LEVEL` (`INFO`, `DEBUG`, ...);
- `LOG_SERVICE_NAME`;
- `LOG_SNAPSHOT_INTERVAL_SEC`;
- `LOG_RESOURCE_RSS_WARN_MB`;
- `WHISPER_SLA_RTF_THRESHOLD`.

### Быстрый анализ через jq

Ошибки pipeline:

```bash
docker compose logs backend | jq -c 'select(.event=="job_failed" or .event=="stage_failed")'
```

SLA drift и причины:

```bash
docker compose logs backend | jq -c 'select(.event=="sla_drift_detected") | {ts,job_id,reason_code,rtf,queue_backlog}'
```

Runtime load snapshots:

```bash
docker compose logs backend | jq -c 'select(.event=="runtime_snapshot") | {ts,in_flight,queue_backlog,claimed_per_min,completed_per_min,failed_per_min,loop_lag_ms}'
```

Высокое потребление памяти воркерами:

```bash
docker compose logs backend | jq -c 'select(.event=="resource_pressure_warning") | {ts,worker_pid,rss_bytes,rss_warn_mb,job_id}'
```
