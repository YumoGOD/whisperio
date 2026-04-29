import contextvars
import json
import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path
from datetime import datetime, timezone
from typing import Any

_LOG_CONTEXT: contextvars.ContextVar[dict[str, Any]] = contextvars.ContextVar(
    "whisperio_log_context",
    default={},
)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _sanitize_field(key: str, value: Any) -> Any:
    lowered = key.lower()
    if value is None:
        return None
    if isinstance(value, (int, float, bool)):
        return value
    if isinstance(value, list):
        return [_sanitize_field(key, item) for item in value]
    if isinstance(value, dict):
        return {k: _sanitize_field(k, v) for k, v in value.items()}
    text = str(value)
    if "error" in lowered:
        return text[:400]
    if lowered in {"prepared_audio_path", "stored_path", "path_on_disk"}:
        return Path(text).name
    if lowered in {"authorization", "token", "password", "secret", "api_key"}:
        return "***"
    return text


def sanitize_event_fields(fields: dict[str, Any]) -> dict[str, Any]:
    sanitized: dict[str, Any] = {}
    for key, value in fields.items():
        clean = _sanitize_field(key, value)
        if clean is not None:
            sanitized[key] = clean
    return sanitized


def bind_log_context(**fields: Any) -> contextvars.Token[dict[str, Any]]:
    current = dict(_LOG_CONTEXT.get({}))
    for key, value in fields.items():
        if value is not None:
            current[key] = value
    return _LOG_CONTEXT.set(current)


def reset_log_context(token: contextvars.Token[dict[str, Any]]) -> None:
    _LOG_CONTEXT.reset(token)


def get_log_context() -> dict[str, Any]:
    return dict(_LOG_CONTEXT.get({}))


class JsonLogFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": _utc_now_iso(),
            "level": record.levelname.lower(),
            "event": getattr(record, "event", record.getMessage()),
            "service": os.getenv("LOG_SERVICE_NAME", "whisperio-backend"),
            "component": getattr(record, "component", record.name),
            "schema_version": "1",
            "logger": record.name,
        }
        payload.update(get_log_context())
        event_data = getattr(record, "event_data", None)
        if isinstance(event_data, dict):
            for key, value in sanitize_event_fields(event_data).items():
                if value is not None:
                    payload[key] = value
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def configure_logging() -> None:
    root = logging.getLogger()
    level = os.getenv("LOG_LEVEL", "INFO").upper()
    root.setLevel(level)
    for handler in list(root.handlers):
        root.removeHandler(handler)

    use_json = os.getenv("LOG_FORMAT", "json").strip().lower() == "json"
    formatter: logging.Formatter
    if use_json:
        formatter = JsonLogFormatter()
    else:
        formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    root.addHandler(stream_handler)

    file_enabled = _env_bool("LOG_FILE_ENABLED", True)
    if file_enabled:
        file_path = Path(os.getenv("LOG_FILE_PATH", "data/logs/backend.jsonl"))
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            file_path,
            maxBytes=max(1024, _env_int("LOG_FILE_MAX_BYTES", 50 * 1024 * 1024)),
            backupCount=max(1, _env_int("LOG_FILE_BACKUP_COUNT", 5)),
            encoding="utf-8",
        )
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)


def log_event(
    logger: logging.Logger,
    *,
    event: str,
    level: int = logging.INFO,
    component: str | None = None,
    **fields: Any,
) -> None:
    clean_fields = sanitize_event_fields(fields)
    logger.log(
        level,
        event,
        extra={
            "event": event,
            "component": component or logger.name,
            "event_data": clean_fields,
        },
    )
