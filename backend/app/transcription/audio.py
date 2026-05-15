from __future__ import annotations

import json
import logging
import subprocess
from pathlib import Path

from app.config import Settings

logger = logging.getLogger(__name__)


class AudioProcessingError(RuntimeError):
    pass


def run_command(command: list[str], timeout: int | None = None) -> subprocess.CompletedProcess[str]:
    logger.info("Запуск команды: %s", " ".join(command))
    try:
        completed = subprocess.run(command, text=True, capture_output=True, check=False, timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        raise AudioProcessingError(
            f"Команда зависла (timeout={timeout}с): {' '.join(command)}"
        ) from exc
    if completed.returncode != 0:
        raise AudioProcessingError(completed.stderr.strip() or completed.stdout.strip())
    return completed


def probe_duration_seconds(path: Path) -> float:
    completed = run_command(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "json",
            str(path),
        ],
        timeout=30,
    )
    payload = json.loads(completed.stdout)
    try:
        return float(payload["format"]["duration"])
    except (KeyError, TypeError, ValueError) as exc:
        raise AudioProcessingError(f"Не удалось прочитать длительность аудио: {path}") from exc


def extract_chunk(input_path: Path, output_path: Path, start: float, duration: float, settings: Settings) -> Path:
    if output_path.exists() and output_path.stat().st_size > 0:
        return output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    filters = []
    if settings.enable_loudnorm:
        # Voice filter: removes sub-200 Hz rumble, boosts quiet audio by 1.5x.
        # lowpass=8000 is intentionally omitted — redundant when resampling to 16 kHz (Nyquist = 8 kHz).
        filters.append("highpass=f=200,volume=1.5")

    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-ss",
        f"{start:.3f}",
        "-t",
        f"{duration:.3f}",
        "-i",
        str(input_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(settings.target_sample_rate),
    ]
    if filters:
        cmd += ["-af", ",".join(filters)]
    cmd += ["-c:a", "pcm_s16le", str(output_path)]

    # Timeout: 10× real-time, minimum 2 minutes. Prevents worker thread deadlock on corrupt files.
    ffmpeg_timeout = max(120, int(duration * 10))
    run_command(cmd, timeout=ffmpeg_timeout)
    return output_path
