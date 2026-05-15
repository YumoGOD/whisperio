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


def prepare_audio(input_path: Path, output_path: Path, settings: Settings) -> Path:
    """Convert entire audio file to 16 kHz mono WAV with optional voice filter.

    Voice filter: highpass=f=200 removes sub-200 Hz rumble; loudnorm normalises to EBU R128
    (-16 LUFS integrated, -1.5 dBTP peak) — handles quietly recorded files much better than
    a fixed volume boost. lowpass=8000 omitted — redundant at 16 kHz (Nyquist = 8 kHz).
    """
    if output_path.exists() and output_path.stat().st_size > 0:
        return output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-i",
        str(input_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(settings.target_sample_rate),
    ]
    if settings.enable_loudnorm:
        cmd += ["-af", "highpass=f=200,loudnorm=I=-16:TP=-1.5:LRA=11"]
    cmd += ["-c:a", "pcm_s16le", str(output_path)]

    # Timeout: generous upper bound for very long files (10 hours max).
    run_command(cmd, timeout=3600)
    return output_path
