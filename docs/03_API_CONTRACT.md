# API Contract

Base URL: `http://localhost:8000`
Все ответы — JSON. Без авторизации.

## POST /api/jobs
Создать задачу транскрипции.

**Request:** `multipart/form-data`
- `file`: бинарный файл (обязательно)
- `language`: string, опционально (`"ru"`, `"en"`, `null` = auto)
- `model`: string, опционально (`"large-v3"` | `"large-v3-turbo"` | `"medium"`), по умолчанию из `.env`
- `diarize`: bool, опционально (в v1 игнорируется)

**Response 201:**
```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "filename": "meeting.mp3",
  "status": "queued",
  "created_at": "2026-05-18T10:30:00Z"
}
```

**Response 400:** неподдерживаемый формат / размер превышен.

## GET /api/jobs
Список задач (последние N, N из `.env`).

**Query:**
- `limit`: int, default 100
- `status`: фильтр опционально

**Response 200:**
```json
{
  "jobs": [
    {
      "id": "...", "filename": "...", "status": "done",
      "duration_sec": 1834.5, "model": "large-v3",
      "detected_language": "ru", "progress": 1.0,
      "created_at": "...", "finished_at": "..."
    }
  ]
}
```

## GET /api/jobs/{id}
Детали одной задачи.

**Response 200:**
```json
{
  "id": "...",
  "filename": "meeting.mp3",
  "status": "done",
  "progress": 1.0,
  "duration_sec": 1834.5,
  "model": "large-v3",
  "language": null,
  "detected_language": "ru",
  "transcript_text": "Полный текст транскрипта...",
  "segments": [
    {"start": 0.0, "end": 4.32, "text": "Добрый день, коллеги."},
    {"start": 4.32, "end": 9.10, "text": "Сегодня обсудим..."}
  ],
  "error": null,
  "created_at": "...",
  "started_at": "...",
  "finished_at": "..."
}
```

Если `status != "done"` — `segments` и `transcript_text` могут быть `null`.

## GET /api/jobs/{id}/download?format=txt|srt|json
Скачать результат.

**Response 200:** файл (с `Content-Disposition: attachment`).
**Response 404:** если задача не done или не существует.

## DELETE /api/jobs/{id}
Удалить задачу и связанные файлы.

**Response 204.**

## GET /api/health
**Response 200:**
```json
{
  "status": "ok",
  "worker_alive": true,
  "gpu_available": true,
  "queue_size": 3
}
```

`worker_alive` определяется по наличию записи в таблице `worker_heartbeat` не старше 30 секунд (воркер пишет туда раз в 5 сек).
