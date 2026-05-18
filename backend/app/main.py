from __future__ import annotations

import html
import json
import mimetypes
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, PlainTextResponse

from app.config import get_settings
from app.db import JobRepository
from app.logging_config import configure_logging
from app.models import Job
from app.transcription.exports import write_docx_export
from app.transcription.profiles import PROFILES

settings = get_settings()
configure_logging(settings)
repo = JobRepository(settings.database_path)

app = FastAPI(title=settings.app_name)


def format_duration(seconds: float | None) -> str:
    if seconds is None:
        return "-"
    total = int(seconds)
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:02}:{minutes:02}:{secs:02}"
    return f"{minutes:02}:{secs:02}"


def format_timestamp(seconds: float | None) -> str:
    if seconds is None:
        return "00:00.000"
    milliseconds = int(round(seconds * 1000))
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1000)
    if hours:
        return f"{hours:02}:{minutes:02}:{secs:02}.{millis:03}"
    return f"{minutes:02}:{secs:02}.{millis:03}"


def status_class(status: str) -> str:
    return {
        "pending": "status-pending",
        "running": "status-running",
        "completed": "status-completed",
        "failed": "status-failed",
    }.get(status, "status-pending")


def status_label(status: str) -> str:
    return {
        "pending": "В очереди",
        "running": "Обрабатывается",
        "completed": "Готово",
        "failed": "Ошибка",
    }.get(status, status)


def format_seconds(seconds: Any) -> str:
    try:
        value = float(seconds)
    except (TypeError, ValueError):
        return "-"
    if value < 60:
        return f"{value:.1f} сек."
    return f"{format_duration(value)} ({value:.1f} сек.)"


def format_ratio(value: Any) -> str:
    try:
        return f"{float(value):.3f}x"
    except (TypeError, ValueError):
        return "-"


_RU_MONTHS = [
    "", "января", "февраля", "марта", "апреля", "мая", "июня",
    "июля", "августа", "сентября", "октября", "ноября", "декабря",
]


def format_ru_datetime(iso_str: str | None) -> str:
    if not iso_str:
        return "-"
    try:
        dt = datetime.fromisoformat(iso_str)
        return f"{dt.day} {_RU_MONTHS[dt.month]} {dt.year}, {dt.hour:02}:{dt.minute:02}"
    except (ValueError, IndexError):
        return iso_str


def job_elapsed_seconds(job: Job) -> float | None:
    if not job.started_at or not job.finished_at:
        return None
    try:
        return (datetime.fromisoformat(job.finished_at) - datetime.fromisoformat(job.started_at)).total_seconds()
    except ValueError:
        return None


def load_diagnostics(job: Job) -> dict[str, Any]:
    if not job.transcript_dir:
        return {}
    path = Path(job.transcript_dir) / "diagnostics.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def render_kv(rows: list[tuple[str, Any]]) -> str:
    items = []
    for label, value in rows:
        safe_value = html.escape("-" if value is None or value == "" else str(value))
        items.append(f"<div><dt>{html.escape(label)}</dt><dd>{safe_value}</dd></div>")
    return f'<dl class="kv">{"".join(items)}</dl>'


def json_for_script(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False).replace("</", "<\\/")


def page_shell(title: str, body: str, *, auto_refresh: bool = False) -> str:
    refresh = '<meta http-equiv="refresh" content="8" />' if auto_refresh else ""
    return f"""
    <!doctype html>
    <html lang="ru">
    <head>
      <meta charset="utf-8" />
      <meta name="viewport" content="width=device-width, initial-scale=1" />
      {refresh}
      <title>{html.escape(title)}</title>
      <style>
        :root {{
          color-scheme: light dark;
          --bg: #0f172a;
          --panel: #111827;
          --panel-2: #1f2937;
          --panel-soft: rgba(15, 23, 42, 0.58);
          --text: #e5e7eb;
          --muted: #9ca3af;
          --border: #374151;
          --brand: #60a5fa;
          --brand-strong: #3b82f6;
          --danger: #f87171;
          --ok: #34d399;
          --warn: #fbbf24;
          --shadow: 0 20px 70px rgba(0, 0, 0, 0.32);
        }}
        * {{ box-sizing: border-box; }}
        html {{ min-width: 320px; }}
        body {{
          margin: 0;
          min-height: 100vh;
          background:
            radial-gradient(circle at top left, rgba(96, 165, 250, 0.18), transparent 34rem),
            radial-gradient(circle at bottom right, rgba(52, 211, 153, 0.10), transparent 30rem),
            linear-gradient(135deg, #0f172a 0%, #111827 55%, #172554 100%);
          color: var(--text);
          font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        }}
        a {{ color: var(--brand); text-decoration: none; }}
        a:hover {{ text-decoration: underline; }}
        a:focus-visible, button:focus-visible, input:focus-visible, select:focus-visible, textarea:focus-visible {{
          outline: 3px solid rgba(96, 165, 250, 0.36);
          outline-offset: 2px;
        }}
        .container {{ width: min(100% - 28px, 1220px); margin: 0 auto; padding: 28px 0 48px; }}
        .topbar {{ display: flex; justify-content: space-between; gap: 16px; align-items: center; flex-wrap: wrap; margin-bottom: 22px; }}
        .topbar > * {{ min-width: 0; }}
        .brand {{ display: flex; flex-direction: column; gap: 5px; min-width: 0; }}
        .brand h1 {{
          margin: 0;
          font-size: clamp(26px, 4vw, 42px);
          line-height: 1.05;
          letter-spacing: -0.04em;
          overflow-wrap: anywhere;
        }}
        .hero {{
          display: grid;
          grid-template-columns: minmax(0, 1fr) max-content;
          gap: 18px;
          align-items: end;
          margin-bottom: 18px;
        }}
        .hero-actions {{ display: flex; gap: 10px; flex-wrap: wrap; justify-content: flex-end; }}
        .muted {{ color: var(--muted); }}
        .job-name {{ font-weight: 700; overflow-wrap: anywhere; }}
        .job-id {{ display: inline-block; max-width: 100%; font-size: 12px; overflow-wrap: anywhere; }}
        .status-line {{ display: flex; align-items: center; gap: 8px; flex-wrap: wrap; }}
        .status-line p {{ margin: 0; }}
        .grid {{ display: grid; grid-template-columns: 390px minmax(0, 1fr); gap: 18px; align-items: start; }}
        .job-grid {{ display: grid; grid-template-columns: minmax(0, 430px) minmax(0, 1fr); gap: 18px; align-items: start; }}
        .grid > *, .job-grid > *, .hero > * {{ min-width: 0; }}
        .card {{
          background: rgba(17, 24, 39, 0.86);
          border: 1px solid rgba(148, 163, 184, 0.2);
          border-radius: 22px;
          box-shadow: var(--shadow);
          padding: 18px;
          backdrop-filter: blur(10px);
          min-width: 0;
          overflow: hidden;
        }}
        .sticky {{ position: sticky; top: 18px; }}
        .card h2, .card h3 {{ margin-top: 0; letter-spacing: -0.02em; }}
        .card h2 {{ margin-bottom: 8px; }}
        .card h3 {{ margin-bottom: 10px; }}
        .subtle-card {{
          border: 1px solid rgba(148, 163, 184, 0.14);
          border-radius: 14px;
          background: var(--panel-soft);
          padding: 12px;
        }}
        label {{ display: block; margin: 12px 0 6px; color: var(--muted); font-size: 14px; }}
        input, select, textarea, button {{
          width: 100%;
          min-width: 0;
          border: 1px solid var(--border);
          border-radius: 12px;
          background: #0b1220;
          color: var(--text);
          font: inherit;
          padding: 11px 12px;
          transition: border-color 140ms ease, box-shadow 140ms ease, transform 140ms ease, background 140ms ease;
        }}
        input:hover, select:hover, textarea:hover {{ border-color: rgba(96, 165, 250, 0.52); }}
        textarea {{ min-height: 92px; resize: vertical; }}
        input[type="file"] {{ border-style: dashed; padding: 18px; background: rgba(11, 18, 32, 0.82); }}
        button {{
          margin-top: 14px;
          background: linear-gradient(135deg, var(--brand), var(--brand-strong));
          border: none;
          color: white;
          cursor: pointer;
          font-weight: 700;
        }}
        button:hover {{ transform: translateY(-1px); }}
        button:disabled {{ cursor: not-allowed; opacity: 0.64; transform: none; }}
        button.secondary, .button-link {{
          display: inline-flex;
          justify-content: center;
          align-items: center;
          width: auto;
          margin-top: 0;
          background: rgba(31, 41, 55, 0.82);
          border: 1px solid var(--border);
          border-radius: 999px;
          color: var(--text);
          padding: 9px 12px;
          font-weight: 700;
          line-height: 1.2;
          text-decoration: none;
          cursor: pointer;
        }}
        button.secondary:hover, .button-link:hover {{ text-decoration: none; border-color: var(--brand); background: rgba(37, 99, 235, 0.18); }}
        table {{ width: 100%; min-width: 720px; border-collapse: collapse; }}
        th, td {{ border-bottom: 1px solid rgba(148, 163, 184, 0.18); padding: 12px 10px; text-align: left; vertical-align: top; }}
        th {{ color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: 0.08em; }}
        td {{ overflow-wrap: anywhere; }}
        .table-wrap {{ overflow-x: auto; -webkit-overflow-scrolling: touch; }}
        .badge {{ display: inline-flex; align-items: center; border-radius: 999px; padding: 4px 9px; font-size: 12px; font-weight: 700; white-space: nowrap; }}
        .status-pending {{ background: rgba(251, 191, 36, 0.15); color: var(--warn); }}
        .status-running {{ background: rgba(96, 165, 250, 0.16); color: var(--brand); }}
        .status-completed {{ background: rgba(52, 211, 153, 0.14); color: var(--ok); }}
        .status-failed {{ background: rgba(248, 113, 113, 0.16); color: var(--danger); }}
        .progress {{ height: 10px; background: #0b1220; border-radius: 999px; overflow: hidden; border: 1px solid var(--border); }}
        .progress span {{ display: block; height: 100%; background: linear-gradient(90deg, var(--brand), var(--ok)); }}
        audio {{ width: 100%; margin: 10px 0 12px; }}
        .player-controls {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px; margin: 8px 0 12px; }}
        .player-controls button {{ width: 100%; margin: 0; padding: 9px; font-size: 14px; }}
        .playback-rate {{ display: grid; grid-template-columns: minmax(0, 1fr) 96px; gap: 10px; align-items: center; }}
        .downloads {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(78px, 1fr)); gap: 8px; }}
        .downloads a, .pill {{
          display: inline-flex;
          justify-content: center;
          align-items: center;
          border: 1px solid var(--border);
          border-radius: 999px;
          padding: 8px 10px;
          background: rgba(31, 41, 55, 0.72);
          color: var(--text);
          min-width: 0;
          overflow-wrap: anywhere;
        }}
        .workspace-header {{
          display: flex;
          justify-content: space-between;
          gap: 12px;
          align-items: flex-start;
          flex-wrap: wrap;
          margin-bottom: 14px;
        }}
        .workspace-header > * {{ min-width: 0; }}
        .toolbar {{ display: flex; gap: 8px; flex-wrap: wrap; align-items: center; margin-bottom: 12px; }}
        .toolbar input {{ flex: 1 1 260px; width: auto; margin: 0; }}
        .toolbar button {{ width: auto; margin: 0; padding: 9px 12px; }}
        .segments {{ display: flex; flex-direction: column; gap: 8px; max-height: 72vh; overflow: auto; padding-right: 4px; scroll-behavior: smooth; }}
        .segment {{
          display: grid;
          grid-template-columns: minmax(92px, 110px) minmax(0, 1fr);
          gap: 12px;
          width: 100%;
          margin: 0;
          text-align: left;
          border: 1px solid rgba(148, 163, 184, 0.16);
          border-radius: 14px;
          background: rgba(15, 23, 42, 0.7);
          padding: 11px;
          cursor: pointer;
          transition: border-color 120ms ease, background 120ms ease, transform 120ms ease;
        }}
        .segment:hover, .segment.active {{ border-color: var(--brand); background: rgba(37, 99, 235, 0.16); }}
        .segment:hover {{ transform: translateY(-1px); }}
        .time {{ color: var(--brand); font-variant-numeric: tabular-nums; font-weight: 800; }}
        .segment-text {{ min-width: 0; line-height: 1.55; overflow-wrap: anywhere; }}
        .search {{ margin-bottom: 12px; }}
        .error {{ color: var(--danger); white-space: pre-wrap; }}
        .empty {{ padding: 24px; text-align: center; color: var(--muted); }}
        .metric-grid {{ display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 10px; margin: 12px 0 16px; }}
        .metric {{
          border: 1px solid rgba(148, 163, 184, 0.16);
          border-radius: 14px;
          background: var(--panel-soft);
          padding: 12px;
          min-width: 0;
        }}
        .metric strong {{ display: block; font-size: clamp(18px, 2vw, 22px); margin-bottom: 2px; overflow-wrap: anywhere; }}
        .sticky .metric-grid {{ grid-template-columns: 1fr; }}
        .kv {{ display: grid; grid-template-columns: 1fr; gap: 8px; margin: 0; }}
        .kv div {{ display: grid; grid-template-columns: minmax(120px, 0.86fr) minmax(0, 1fr); gap: 12px; border-bottom: 1px solid rgba(148, 163, 184, 0.12); padding: 7px 0; }}
        .kv dt {{ color: var(--muted); min-width: 0; }}
        .kv dd {{ margin: 0; min-width: 0; text-align: right; overflow-wrap: anywhere; }}
        .stage-list {{ display: flex; flex-direction: column; gap: 8px; }}
        .stage-row {{
          display: grid;
          grid-template-columns: minmax(0, 1fr) 92px;
          gap: 10px;
          align-items: center;
          border: 1px solid rgba(148, 163, 184, 0.14);
          border-radius: 12px;
          padding: 9px 10px;
          background: rgba(15, 23, 42, 0.48);
        }}
        .stage-row span:last-child {{ color: var(--brand); font-variant-numeric: tabular-nums; text-align: right; }}
        .timeline {{
          display: grid;
          grid-template-columns: repeat(4, minmax(0, 1fr));
          gap: 8px;
          margin: 12px 0;
        }}
        .timeline-step {{
          border: 1px solid rgba(148, 163, 184, 0.14);
          border-radius: 12px;
          padding: 9px;
          background: rgba(15, 23, 42, 0.45);
        }}
        .timeline-step strong {{ display: block; margin-bottom: 4px; }}
        details {{ margin-top: 12px; overflow-x: auto; }}
        details table {{ min-width: 620px; }}
        summary {{ cursor: pointer; color: var(--brand); font-weight: 700; margin-bottom: 10px; }}
        @media (max-width: 900px) {{
          .grid, .job-grid, .hero {{ grid-template-columns: 1fr; }}
          .hero-actions {{ justify-content: flex-start; }}
          .sticky {{ position: static; }}
          .metric-grid {{ grid-template-columns: 1fr; }}
          .timeline {{ grid-template-columns: 1fr 1fr; }}
        }}
        @media (max-width: 640px) {{
          .container {{ width: min(100% - 20px, 1220px); padding-top: 18px; }}
          .card {{ border-radius: 18px; padding: 14px; }}
          .hero-actions, .toolbar, .player-controls {{ display: grid; grid-template-columns: 1fr; width: 100%; }}
          .button-link, .toolbar button {{ width: 100%; }}
          .playback-rate, .timeline, .stage-row {{ grid-template-columns: 1fr; }}
          .stage-row span:last-child, .kv dd {{ text-align: left; }}
          .kv div {{ grid-template-columns: 1fr; gap: 3px; }}
          .segment {{ grid-template-columns: 1fr; gap: 6px; }}
          .segments {{ max-height: none; padding-right: 0; }}
          table {{ min-width: 640px; }}
        }}
        .hidden {{ display: none !important; }}
        @keyframes pulse-running {{
          0%, 100% {{ opacity: 1; }}
          50% {{ opacity: 0.55; }}
        }}
        .status-running {{ animation: pulse-running 1.6s ease-in-out infinite; }}
        .file-drop-zone {{
          border: 2px dashed var(--border);
          border-radius: 12px;
          transition: border-color 140ms ease, background 140ms ease;
        }}
        .file-drop-zone.drag-over {{
          border-color: var(--brand);
          background: rgba(96, 165, 250, 0.08);
        }}
        .file-drop-zone input[type="file"] {{ border: none; width: 100%; }}
        .upload-progress {{
          display: none;
          height: 4px;
          background: #0b1220;
          border-radius: 999px;
          overflow: hidden;
          margin: 8px 0 0;
          border: 1px solid var(--border);
        }}
        .upload-progress.active {{ display: block; }}
        .upload-progress-bar {{
          height: 100%;
          background: linear-gradient(90deg, var(--brand), var(--ok));
          animation: upload-indeterminate 1.4s ease-in-out infinite;
        }}
        @keyframes upload-indeterminate {{
          0%   {{ transform: translateX(-100%) scaleX(0.4); }}
          50%  {{ transform: translateX(60%) scaleX(0.5); }}
          100% {{ transform: translateX(200%) scaleX(0.4); }}
        }}
        .metric-grid-4 {{ grid-template-columns: repeat(4, minmax(0, 1fr)); }}
        @media (max-width: 900px) {{ .metric-grid-4 {{ grid-template-columns: repeat(2, 1fr); }} }}
        @media (max-width: 640px) {{ .metric-grid-4 {{ grid-template-columns: 1fr; }} }}
      </style>
    </head>
    <body>
      <main class="container">{body}</main>
    </body>
    </html>
    """


def job_to_dict(job: Job, include_result: bool = False) -> dict[str, Any]:
    payload = {
        "id": job.id,
        "filename": job.original_filename,
        "status": job.status,
        "progress": job.progress,
        "created_at": job.created_at,
        "started_at": job.started_at,
        "finished_at": job.finished_at,
        "params": job.params,
        "error": job.error,
        "duration_seconds": job.duration_seconds,
        "download": {
            "txt": f"/api/jobs/{job.id}/download/txt",
            "json": f"/api/jobs/{job.id}/download/json",
            "srt": f"/api/jobs/{job.id}/download/srt",
            "vtt": f"/api/jobs/{job.id}/download/vtt",
            "docx": f"/api/jobs/{job.id}/download/docx",
        },
    }
    if include_result:
        payload["text"] = job.text
        payload["segments"] = job.segments
    return payload


def validate_extension(filename: str) -> None:
    extension = Path(filename).suffix.lower().lstrip(".")
    allowed_extensions = settings.allowed_extension_set
    if "*" in allowed_extensions:
        return
    if extension not in allowed_extensions:
        allowed = ", ".join(sorted(allowed_extensions))
        raise HTTPException(status_code=400, detail=f"Формат '.{extension}' не поддерживается. Разрешено: {allowed}")


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    jobs = repo.list_jobs(limit=50)
    total_jobs = len(jobs)
    running_jobs = sum(1 for job in jobs if job.status == "running")
    completed_jobs = sum(1 for job in jobs if job.status == "completed")
    failed_jobs = sum(1 for job in jobs if job.status == "failed")
    rows = []
    for job in jobs:
        safe_name = html.escape(job.original_filename)
        progress_percent = int(job.progress * 100)
        rows.append(
            f"""
            <tr>
              <td><a class="job-name" href="/jobs/{job.id}">{safe_name}</a><br><span class="muted job-id">{job.id}</span></td>
              <td><span class="badge {status_class(job.status)}">{status_label(job.status)}</span></td>
              <td>
                <div class="progress" title="{progress_percent}%"><span style="width: {progress_percent}%"></span></div>
                <span class="muted">{progress_percent}%</span>
              </td>
              <td>{format_duration(job.duration_seconds)}</td>
              <td><span class="muted">{format_ru_datetime(job.created_at)}</span></td>
            </tr>
            """
        )
    rows_html = "".join(rows) or '<tr><td class="empty" colspan="5">Задач пока нет. Загрузите аудиофайл, чтобы начать.</td></tr>'
    profile_options = "\n".join(
        f'<option value="{html.escape(key)}"'
        f'{" selected" if key == settings.default_profile else ""}>'
        f'{html.escape(key)}: {html.escape(val.get("description", ""))}</option>'
        for key, val in PROFILES.items()
    )
    body = f"""
      <div class="hero">
        <div class="brand">
          <h1>{html.escape(settings.app_name)}</h1>
          <span class="muted">Локальная транскрибация длинных шумных лекций с таймингами и прослушиванием.</span>
        </div>
        <div class="hero-actions">
          <a class="button-link" href="/api/jobs">Задачи JSON</a>
          <a class="button-link" href="/api/profiles">Профили JSON</a>
        </div>
      </div>

      <section class="metric-grid metric-grid-4">
        <div class="metric"><strong>{total_jobs}</strong><span class="muted">Всего задач</span></div>
        <div class="metric"><strong>{running_jobs}</strong><span class="muted">Сейчас в работе</span></div>
        <div class="metric"><strong>{completed_jobs}</strong><span class="muted">Готово</span></div>
        <div class="metric"><strong>{failed_jobs}</strong><span class="muted">Ошибок</span></div>
      </section>

      <section class="grid">
        <div class="card sticky">
          <h2>Загрузка аудио</h2>
          <p class="muted">Поддерживаются форматы, которые читает ffmpeg: mp3, wav, m4a, flac, webm, mp4 и другие.</p>
          <form id="upload-form" action="/api/jobs" method="post" enctype="multipart/form-data">
            <label for="file">Файл</label>
            <div class="file-drop-zone" id="file-drop-zone">
              <input id="file" type="file" name="file" required />
            </div>
            <label for="profile">Профиль транскрибации</label>
            <select id="profile" name="profile">
              {profile_options}
            </select>
            <label for="audio_context">Описание аудио</label>
            <textarea id="audio_context" name="audio_context" placeholder="Что за запись, кто говорит, тип мероприятия. Например: рекламная лекция Bauer, ведущий и участники"></textarea>
            <label for="expected_content">Примерное содержание</label>
            <textarea id="expected_content" name="expected_content" placeholder="Темы, блоки, продукты. Например: соковыжималки, массаж, призы, сертификаты"></textarea>
            <label for="dynamic_terms">Дополнительные слова (необязательно)</label>
            <textarea id="dynamic_terms" name="dynamic_terms" placeholder="Одно слово или фраза на строку. Опционально: Слово | вариант1, вариант2"></textarea>
            <button id="upload-button" type="submit">Загрузить и распознать</button>
            <div class="upload-progress" id="upload-progress"><div class="upload-progress-bar"></div></div>
            <p id="upload-status" class="muted"></p>
          </form>
          <div class="timeline">
            <div class="timeline-step"><strong>1</strong><span class="muted">Загрузка</span></div>
            <div class="timeline-step"><strong>2</strong><span class="muted">Подготовка</span></div>
            <div class="timeline-step"><strong>3</strong><span class="muted">Whisper</span></div>
            <div class="timeline-step"><strong>4</strong><span class="muted">Экспорт</span></div>
          </div>
          <p class="muted">После загрузки откроется страница задачи, где можно слушать аудио и проверять текст по таймингам.</p>
        </div>

        <div class="card">
          <div class="workspace-header">
            <div>
              <h2>Задачи</h2>
              <p class="muted">Откройте задачу, чтобы слушать запись, сверять сегменты и скачать результат.</p>
            </div>
          </div>
          <div class="table-wrap">
            <table>
              <thead><tr><th>Файл</th><th>Статус</th><th>Прогресс</th><th>Длительность</th><th>Создана</th></tr></thead>
              <tbody>{rows_html}</tbody>
            </table>
          </div>
        </div>
      </section>

      <script>
        const form = document.getElementById("upload-form");
        const button = document.getElementById("upload-button");
        const status = document.getElementById("upload-status");
        const uploadProgress = document.getElementById("upload-progress");
        form.addEventListener("submit", async (event) => {{
          event.preventDefault();
          button.disabled = true;
          status.textContent = "Загрузка файла...";
          uploadProgress.classList.add("active");
          try {{
            const response = await fetch(form.action, {{ method: "POST", body: new FormData(form) }});
            const payload = await response.json();
            if (!response.ok) throw new Error(payload.detail || "Не удалось загрузить файл");
            window.location.href = `/jobs/${{payload.id}}`;
          }} catch (error) {{
            uploadProgress.classList.remove("active");
            status.textContent = error.message;
            button.disabled = false;
          }}
        }});
        const dropZone = document.getElementById("file-drop-zone");
        dropZone.addEventListener("dragover", (e) => {{
          e.preventDefault();
          dropZone.classList.add("drag-over");
        }});
        dropZone.addEventListener("dragleave", () => dropZone.classList.remove("drag-over"));
        dropZone.addEventListener("drop", (e) => {{
          e.preventDefault();
          dropZone.classList.remove("drag-over");
          const files = e.dataTransfer.files;
          if (files.length > 0) {{
            const fileInput = document.getElementById("file");
            const dt = new DataTransfer();
            dt.items.add(files[0]);
            fileInput.files = dt.files;
          }}
        }});
      </script>
    """
    return page_shell(settings.app_name, body)


@app.get("/jobs/{job_id}", response_class=HTMLResponse)
def job_page(job_id: str) -> str:
    job = repo.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Задача не найдена")
    progress_percent = int(job.progress * 100)
    segments = job.segments or []
    segment_rows = []
    for index, segment in enumerate(segments):
        start = float(segment.get("start") or 0)
        end = float(segment.get("end") or start)
        text = html.escape(segment.get("text") or "")
        segment_rows.append(
            f"""
            <button class="segment" type="button" data-index="{index}" data-start="{start:.3f}" data-end="{end:.3f}">
              <span class="time">{format_timestamp(start)}<br><span class="muted">{format_timestamp(end)}</span></span>
              <span class="segment-text">{text}</span>
            </button>
            """
        )
    segment_html = "".join(segment_rows) or '<div class="empty">Сегменты появятся после завершения транскрибации.</div>'
    downloads_disabled = job.status != "completed" or not job.transcript_dir
    if downloads_disabled:
        downloads_html = '<span class="muted">Скачивание будет доступно после завершения обработки.</span>'
    else:
        downloads_html = f"""
          <div class="downloads">
            <a href="/api/jobs/{job.id}/download/txt">TXT</a>
            <a href="/api/jobs/{job.id}/download/json">JSON</a>
            <a href="/api/jobs/{job.id}/download/srt">SRT</a>
            <a href="/api/jobs/{job.id}/download/vtt">VTT</a>
            <a href="/api/jobs/{job.id}/download/docx">DOCX</a>
          </div>
        """
    error_html = f'<p class="error">{html.escape(job.error)}</p>' if job.error else ""
    diagnostics = load_diagnostics(job)
    profile_name = str(job.params.get("profile") or settings.default_profile)
    profile_settings = diagnostics.get("profile") or PROFILES.get(profile_name, {})
    runtime_settings = diagnostics.get("settings") or {}
    total_elapsed = diagnostics.get("elapsed_seconds")
    total_rtf = diagnostics.get("real_time_factor")
    segments_raw = (diagnostics.get("transcribe") or {}).get("segments_raw")
    if total_elapsed is None:
        total_elapsed = job_elapsed_seconds(job)
    metrics_html = f"""
      <div class="metric-grid">
        <div class="metric"><strong>{format_seconds(total_elapsed)}</strong><span class="muted">Общее время</span></div>
        <div class="metric"><strong>{format_ratio(total_rtf)}</strong><span class="muted">Real-time factor</span></div>
        <div class="metric"><strong>{segments_raw if segments_raw is not None else '-'}</strong><span class="muted">Сырых сегментов</span></div>
      </div>
    """
    whisper_html = render_kv(
        [
            ("Модель", runtime_settings.get("model") or job.params.get("model") or settings.whisper_model),
            ("Устройство", runtime_settings.get("device") or settings.whisper_device),
            ("Тип вычислений", runtime_settings.get("compute_type") or job.params.get("compute_type") or settings.whisper_compute_type),
            ("Язык", runtime_settings.get("language") or job.params.get("language") or settings.whisper_language or "auto"),
            ("Задача Whisper", runtime_settings.get("task") or job.params.get("task") or settings.whisper_task),
            ("Профиль", profile_name),
            ("Beam size", profile_settings.get("beam_size")),
            ("Best of", profile_settings.get("best_of")),
            ("VAD", "включен" if profile_settings.get("vad_filter") else "выключен"),
            ("condition_on_previous_text", profile_settings.get("condition_on_previous_text")),
            ("Batch size", profile_settings.get("batch_size")),
            ("Loudnorm", "включен" if runtime_settings.get("enable_loudnorm", settings.enable_loudnorm) else "выключен"),
            ("Sample rate", f"{runtime_settings.get('target_sample_rate') or settings.target_sample_rate} Hz"),
        ]
    )
    stage_labels = [
        ("probe_seconds", "Анализ длительности"),
        ("prepare_seconds", "Подготовка аудио"),
        ("transcribe_seconds", "Распознавание Whisper"),
        ("postprocess_seconds", "Постобработка"),
        ("export_seconds", "Экспорт файлов"),
    ]
    stage_timings = diagnostics.get("stage_timings") or {}
    stage_rows = []
    for key, label in stage_labels:
        if key in stage_timings:
            stage_rows.append(f"<div class=\"stage-row\"><span>{html.escape(label)}</span><span>{format_seconds(stage_timings.get(key))}</span></div>")
    stage_html = (
        f'<div class="stage-list">{"".join(stage_rows)}</div>'
        if stage_rows
        else '<p class="muted">Подробные времена появятся для новых задач после обновления worker-а.</p>'
    )
    chunk_rows = []
    for chunk in diagnostics.get("chunks") or []:
        chunk_rows.append(
            f"""
            <tr>
              <td>{int(chunk.get("index", 0)) + 1}</td>
              <td>{format_timestamp(chunk.get("start"))} - {format_timestamp(chunk.get("end"))}</td>
              <td>{format_seconds(chunk.get("transcribe_seconds"))}</td>
              <td>{format_ratio(chunk.get("real_time_factor"))}</td>
              <td>{html.escape(str(chunk.get("segments", "-")))}</td>
            </tr>
            """
        )
    chunk_html = ""
    if chunk_rows:
        chunk_html = f"""
          <details>
            <summary>Время по фрагментам</summary>
            <table>
              <thead><tr><th>#</th><th>Тайминг</th><th>Whisper</th><th>RTF</th><th>Сегм.</th></tr></thead>
              <tbody>{"".join(chunk_rows)}</tbody>
            </table>
          </details>
        """
    glossary = diagnostics.get("glossary") or {}
    job_context = diagnostics.get("job_context") or {
        "audio_context": job.params.get("audio_context") or "",
        "expected_content": job.params.get("expected_content") or "",
        "dynamic_terms": job.params.get("dynamic_terms") or "",
    }
    glossary_html = render_kv(
        [
            ("Описание аудио", job_context.get("audio_context")),
            ("Примерное содержание", job_context.get("expected_content")),
            ("Терминов всего", glossary.get("terms_total")),
            ("Hard терминов", glossary.get("hard_terms")),
            ("Soft терминов", glossary.get("soft_terms")),
            ("Динамических терминов", glossary.get("dynamic_terms")),
            ("Hard-замен", (glossary.get("replacement_counts") or {}).get("total")),
        ]
    )
    glossary_prompt = html.escape(str(glossary.get("prompt") or "Prompt появится после обработки новой задачи."))
    glossary_terms = glossary.get("terms") or []
    glossary_terms_html = ""
    if glossary_terms:
        rows = []
        for term in glossary_terms:
            rows.append(
                f"""
                <tr>
                  <td>{html.escape(str(term.get("canonical", "-")))}</td>
                  <td>{html.escape(str(term.get("mode", "-")))}</td>
                  <td>{html.escape(str(term.get("category", "-")))}</td>
                  <td>{html.escape(", ".join(term.get("spoken_forms") or []))}</td>
                </tr>
                """
            )
        glossary_terms_html = f"""
          <details>
            <summary>Использованные термины</summary>
            <table>
              <thead><tr><th>Термин</th><th>Mode</th><th>Категория</th><th>Варианты</th></tr></thead>
              <tbody>{"".join(rows)}</tbody>
            </table>
          </details>
        """
    body = f"""
      <div class="topbar" data-job-id="{job.id}" data-initial-status="{job.status}">
        <div class="brand">
          <a href="/">Назад к задачам</a>
          <h1>{html.escape(job.original_filename)}</h1>
          <span class="muted">{job.id}</span>
        </div>
        <span id="status-badge" class="badge {status_class(job.status)}">{status_label(job.status)}</span>
      </div>

      <section class="metric-grid">
        <div class="metric"><strong id="metric-status">{status_label(job.status)}</strong><span class="muted">Текущий статус</span></div>
        <div class="metric"><strong id="metric-progress">{progress_percent}%</strong><span class="muted">Прогресс обработки</span></div>
        <div class="metric"><strong>{format_duration(job.duration_seconds)}</strong><span class="muted">Длительность аудио</span></div>
      </section>

      <section class="job-grid">
        <aside class="card sticky">
          <h2>Прослушивание аудио</h2>
          <audio id="audio" controls preload="metadata" src="/api/jobs/{job.id}/audio"></audio>
          <div class="player-controls">
            <button class="secondary" type="button" id="skip-back">-10 сек.</button>
            <button class="secondary" type="button" id="play-pause">Пуск / пауза</button>
            <button class="secondary" type="button" id="skip-forward">+10 сек.</button>
          </div>
          <label for="playback-rate">Скорость воспроизведения</label>
          <div class="playback-rate">
            <input id="playback-rate" type="range" min="0.5" max="2" step="0.05" value="1" />
            <span class="pill" id="playback-rate-label">1.00x</span>
          </div>
          <p class="muted">Нажмите на сегмент текста, чтобы перейти к этому месту в аудио. Активный сегмент подсвечивается во время проигрывания.</p>

          <h3>Статус задачи</h3>
          <div class="progress" id="progress-bar" title="{progress_percent}%"><span id="progress-bar-inner" style="width: {progress_percent}%"></span></div>
          <div class="status-line">
            <p id="progress-text">Выполнено: {progress_percent}%</p>
            <span class="pill">{format_duration(job.duration_seconds)}</span>
          </div>
          <p><strong>Профиль:</strong> {html.escape(str(job.params.get("profile", "-")))}</p>
          <p><strong>Создана:</strong> <span class="muted">{format_ru_datetime(job.created_at)}</span></p>
          {error_html}

          <h3>Метрики выполнения</h3>
          {metrics_html}

          <h3>Настройки Whisper</h3>
          {whisper_html}

          <h3>Словарь и контекст</h3>
          {glossary_html}
          <details>
            <summary>Prompt для Whisper</summary>
            <p class="muted">{glossary_prompt}</p>
          </details>
          {glossary_terms_html}

          <h3>Время по этапам</h3>
          {stage_html}
          {chunk_html}

          <h3>Скачать результат</h3>
          {downloads_html}
          <p><a href="/api/jobs/{job.id}">Открыть JSON-статус</a></p>
        </aside>

        <section class="card">
          <div class="workspace-header">
            <div>
              <h2>Текст с таймингами</h2>
              <p class="muted">Проверяйте распознавание по аудио, ищите фразы и переходите между сегментами.</p>
            </div>
            <span class="pill" id="segment-count">{len(segments)} сегм.</span>
          </div>
          <div class="toolbar">
            <input id="segment-search" class="search" type="search" placeholder="Поиск по транскрипту..." />
            <button class="secondary" type="button" id="copy-text">Скопировать текст</button>
            <button class="secondary" type="button" id="toggle-autoscroll">Автопрокрутка: вкл.</button>
          </div>
          <div id="segments" class="segments">{segment_html}</div>
        </section>
      </section>

      <script>
        const segments = {json_for_script(segments)};
        const audio = document.getElementById("audio");
        const buttons = Array.from(document.querySelectorAll(".segment"));
        const search = document.getElementById("segment-search");
        const skipBack = document.getElementById("skip-back");
        const skipForward = document.getElementById("skip-forward");
        const playPause = document.getElementById("play-pause");
        const playbackRate = document.getElementById("playback-rate");
        const playbackRateLabel = document.getElementById("playback-rate-label");
        const copyText = document.getElementById("copy-text");
        const toggleAutoscroll = document.getElementById("toggle-autoscroll");
        let autoscroll = true;

        function setActive(index) {{
          buttons.forEach((button) => {{
            const active = Number(button.dataset.index) === index;
            button.classList.toggle("active", active);
            if (active && autoscroll) {{
              button.scrollIntoView({{ block: "nearest", behavior: "smooth" }});
            }}
          }});
        }}

        buttons.forEach((button) => {{
          button.addEventListener("click", () => {{
            const start = Number(button.dataset.start || 0);
            audio.currentTime = start;
            audio.play();
            setActive(Number(button.dataset.index));
          }});
        }});

        skipBack.addEventListener("click", () => {{
          audio.currentTime = Math.max(0, audio.currentTime - 10);
        }});

        skipForward.addEventListener("click", () => {{
          audio.currentTime = Math.min(audio.duration || audio.currentTime + 10, audio.currentTime + 10);
        }});

        playPause.addEventListener("click", () => {{
          if (audio.paused) audio.play();
          else audio.pause();
        }});

        playbackRate.addEventListener("input", () => {{
          audio.playbackRate = Number(playbackRate.value);
          playbackRateLabel.textContent = `${{audio.playbackRate.toFixed(2)}}x`;
        }});

        copyText.addEventListener("click", async () => {{
          const text = segments.map((segment) => segment.text).join("\\n");
          try {{
            await navigator.clipboard.writeText(text);
            copyText.textContent = "Скопировано";
            setTimeout(() => {{ copyText.textContent = "Скопировать текст"; }}, 1400);
          }} catch (err) {{
            copyText.textContent = "Ошибка: нет доступа к буферу";
            setTimeout(() => {{ copyText.textContent = "Скопировать текст"; }}, 2500);
          }}
        }});

        toggleAutoscroll.addEventListener("click", () => {{
          autoscroll = !autoscroll;
          toggleAutoscroll.textContent = `Автопрокрутка: ${{autoscroll ? "вкл." : "выкл."}}`;
        }});

        audio.addEventListener("timeupdate", () => {{
          const current = audio.currentTime;
          const index = segments.findIndex((segment) => current >= segment.start && current <= segment.end);
          if (index >= 0) setActive(index);
        }});

        search.addEventListener("input", () => {{
          const query = search.value.trim().toLowerCase();
          let visible = 0;
          buttons.forEach((button) => {{
            const text = button.textContent.toLowerCase();
            const show = text.includes(query);
            button.classList.toggle("hidden", !show);
            if (show) visible++;
          }});
          const counter = document.getElementById("segment-count");
          if (counter) {{
            counter.textContent = query
              ? `${{visible}} из ${{buttons.length}} сегм.`
              : `${{buttons.length}} сегм.`;
          }}
        }});

        document.addEventListener("keydown", (e) => {{
          const tag = document.activeElement && document.activeElement.tagName;
          if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT") return;
          if (e.key === " " || e.code === "Space") {{
            e.preventDefault();
            if (audio.paused) audio.play();
            else audio.pause();
          }} else if (e.key === "ArrowLeft") {{
            e.preventDefault();
            audio.currentTime = Math.max(0, audio.currentTime - 10);
          }} else if (e.key === "ArrowRight") {{
            e.preventDefault();
            audio.currentTime = Math.min(audio.duration || audio.currentTime + 10, audio.currentTime + 10);
          }}
        }});

        (function () {{
          const topbar = document.querySelector(".topbar[data-job-id]");
          if (!topbar) return;
          const jobId = topbar.dataset.jobId;
          const initialStatus = topbar.dataset.initialStatus;
          if (initialStatus !== "pending" && initialStatus !== "running") return;

          const statusBadge = document.getElementById("status-badge");
          const metricStatus = document.getElementById("metric-status");
          const metricProgress = document.getElementById("metric-progress");
          const progressBar = document.getElementById("progress-bar");
          const progressBarInner = document.getElementById("progress-bar-inner");
          const progressText = document.getElementById("progress-text");

          const STATUS_CLASS = {{
            pending: "status-pending",
            running: "status-running",
            completed: "status-completed",
            failed: "status-failed",
          }};
          const STATUS_LABEL = {{
            pending: "В очереди",
            running: "Обрабатывается",
            completed: "Готово",
            failed: "Ошибка",
          }};

          function applyStatus(data) {{
            const cls = STATUS_CLASS[data.status] || "status-pending";
            const label = STATUS_LABEL[data.status] || data.status;
            const pct = Math.round((data.progress || 0) * 100);
            statusBadge.className = `badge ${{cls}}`;
            statusBadge.textContent = label;
            metricStatus.textContent = label;
            metricProgress.textContent = `${{pct}}%`;
            progressBarInner.style.width = `${{pct}}%`;
            progressBar.title = `${{pct}}%`;
            progressText.textContent = `Выполнено: ${{pct}}%`;
          }}

          const interval = setInterval(async () => {{
            try {{
              const resp = await fetch(`/api/jobs/${{jobId}}`);
              if (!resp.ok) return;
              const data = await resp.json();
              applyStatus(data);
              if (data.status !== "pending" && data.status !== "running") {{
                clearInterval(interval);
                if (data.status === "completed") window.location.reload();
              }}
            }} catch (_) {{}}
          }}, 5000);
        }})();
      </script>
    """
    return page_shell(f"Задача {job.id}", body)


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/profiles")
def profiles() -> dict[str, Any]:
    return PROFILES


@app.post("/api/jobs")
def create_job(
    file: UploadFile = File(...),
    profile: str = Form(default=settings.default_profile),
    audio_context: str = Form(default=""),
    expected_content: str = Form(default=""),
    dynamic_terms: str = Form(default=""),
):
    if profile not in PROFILES:
        raise HTTPException(status_code=400, detail=f"Неизвестный профиль '{profile}'")
    if not file.filename:
        raise HTTPException(status_code=400, detail="У загруженного файла должно быть имя")
    validate_extension(file.filename)

    job_id = uuid.uuid4().hex
    safe_name = Path(file.filename).name
    upload_path = settings.upload_dir / f"{job_id}_{safe_name}"
    bytes_written = 0
    try:
        with upload_path.open("wb") as output:
            while True:
                chunk = file.file.read(1024 * 1024)
                if not chunk:
                    break
                bytes_written += len(chunk)
                if bytes_written > settings.max_upload_bytes:
                    output.close()
                    upload_path.unlink(missing_ok=True)
                    raise HTTPException(
                        status_code=413,
                        detail=f"Файл слишком большой. Максимальный размер: {settings.max_upload_mb} МБ",
                    )
                output.write(chunk)
    finally:
        file.file.close()

    params = {
        "profile": profile,
        "model": settings.whisper_model,
        "compute_type": settings.whisper_compute_type,
        "language": settings.whisper_language,
        "task": settings.whisper_task,
        "audio_context": audio_context.strip(),
        "expected_content": expected_content.strip(),
        "dynamic_terms": dynamic_terms.strip(),
    }
    job = repo.create_job(job_id, safe_name, str(upload_path), params)
    return JSONResponse(job_to_dict(job), status_code=201)


@app.get("/api/jobs")
def list_jobs(limit: int = Query(default=100, ge=1, le=500), offset: int = Query(default=0, ge=0)):
    return {"jobs": [job_to_dict(job) for job in repo.list_jobs(limit=limit, offset=offset)]}


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    job = repo.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Задача не найдена")
    return job_to_dict(job, include_result=True)


@app.get("/api/jobs/{job_id}/result.txt", response_class=PlainTextResponse)
def get_job_text(job_id: str) -> str:
    job = repo.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Задача не найдена")
    if job.status != "completed":
        raise HTTPException(status_code=409, detail=f"Задача еще не завершена: {status_label(job.status)}")
    return job.text or ""


@app.get("/api/jobs/{job_id}/audio")
def stream_job_audio(job_id: str):
    job = repo.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Задача не найдена")
    path = Path(job.upload_path)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Исходный аудиофайл не найден")
    media_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    return FileResponse(path, media_type=media_type, filename=job.original_filename)


@app.get("/api/jobs/{job_id}/download/{fmt}")
def download_result(job_id: str, fmt: str):
    if fmt not in {"txt", "json", "srt", "vtt", "docx"}:
        raise HTTPException(status_code=400, detail="Формат должен быть одним из: txt, json, srt, vtt, docx")
    job = repo.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Задача не найдена")
    if job.status != "completed" or not job.transcript_dir:
        raise HTTPException(status_code=409, detail=f"Задача еще не завершена: {status_label(job.status)}")
    path = Path(job.transcript_dir) / f"{job.id}.{fmt}"
    if fmt == "docx" and not path.exists():
        write_docx_export(path, job.segments or [])
    if not path.exists():
        raise HTTPException(status_code=404, detail="Файл экспорта не найден")
    media_types = {
        "txt": "text/plain; charset=utf-8",
        "json": "application/json",
        "srt": "application/x-subrip",
        "vtt": "text/vtt",
        "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    }
    return FileResponse(path, media_type=media_types[fmt], filename=f"{job.original_filename}.{fmt}")


@app.exception_handler(Exception)
def generic_exception_handler(_, exc: Exception):
    return JSONResponse(status_code=500, content={"detail": str(exc)})
