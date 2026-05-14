from __future__ import annotations

import json
import logging
import subprocess
from pathlib import Path

from app.config import Settings

logger = logging.getLogger(__name__)


class AudioProcessingError(RuntimeError):
    pass


def run_command(command: list[str]) -> subprocess.CompletedProcess[str]:
    logger.info("Запуск команды: %s", " ".join(command))
    completed = subprocess.run(command, text=True, capture_output=True, check=False)
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
        ]
    )
    payload = json.loads(completed.stdout)
    try:
        return float(payload["format"]["duration"])
    except (KeyError, TypeError, ValueError) as exc:
        raise AudioProcessingError(f"Не удалось прочитать длительность аудио: {path}") from exc


def prepare_audio(input_path: Path, output_path: Path, settings: Settings) -> Path:
    if output_path.exists() and output_path.stat().st_size > 0:
        logger.info("Используется уже подготовленный аудиофайл: %s", output_path)
        return output_path

    output_path.parent.mkdir(parents=True, exist_ok=True)
    filters = []
    if settings.enable_loudnorm:
        filters.append("loudnorm=I=-16:TP=-1.5:LRA=11")

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
    if filters:
        cmd += ["-af", ",".join(filters)]
    cmd += ["-c:a", "pcm_s16le", str(output_path)]

    run_command(cmd)
    return output_path


def extract_chunk(input_path: Path, output_path: Path, start: float, duration: float) -> Path:
    if output_path.exists() and output_path.stat().st_size > 0:
        return output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    run_command(
        [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-ss",
            f"{start:.3f}",
            "-t",
            f"{duration:.3f}",
            "-i",
            str(input_path),
            "-c:a",
            "copy",
            str(output_path),
        ]
    )
    return output_path
