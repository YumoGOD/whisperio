"""SQLAlchemy engine, session factory, declarative Base.

Используется и API-процессом, и worker'ом. SQLite в режиме WAL — чтобы
один процесс мог писать, пока другой читает, без блокировок.
"""

from pathlib import Path

from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from app.config import PROJECT_ROOT, settings

_SQLITE_PREFIX = "sqlite:///"


def _resolve_sqlite_url(url: str) -> str:
    """Превращает относительный sqlite-URL в абсолютный (относительно корня проекта).
    Нужно, чтобы api (cwd=backend/) и worker открывали одну и ту же БД.
    """
    if not url.startswith(_SQLITE_PREFIX):
        return url
    path_part = url[len(_SQLITE_PREFIX):]
    p = Path(path_part)
    if not p.is_absolute():
        p = (PROJECT_ROOT / p).resolve()
    return f"{_SQLITE_PREFIX}{p.as_posix()}"


DB_URL: str = _resolve_sqlite_url(settings.DB_URL)
_IS_SQLITE: bool = DB_URL.startswith(_SQLITE_PREFIX)

if _IS_SQLITE:
    # SQLite не создаёт родительские директории сам — обеспечим их сразу,
    # чтобы create_all() не падал при первом запуске.
    _sqlite_file = Path(DB_URL[len(_SQLITE_PREFIX):])
    _sqlite_file.parent.mkdir(parents=True, exist_ok=True)

engine: Engine = create_engine(
    DB_URL,
    connect_args={"check_same_thread": False} if _IS_SQLITE else {},
    future=True,
)


@event.listens_for(engine, "connect")
def _set_sqlite_pragmas(dbapi_connection, connection_record):  # type: ignore[no-untyped-def]
    if not _IS_SQLITE:
        return
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.execute("PRAGMA synchronous=NORMAL")
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.close()


SessionLocal = sessionmaker(
    bind=engine,
    autoflush=False,
    autocommit=False,
    expire_on_commit=False,
    future=True,
)


class Base(DeclarativeBase):
    pass
