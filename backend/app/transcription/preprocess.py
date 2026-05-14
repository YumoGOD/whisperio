import json
import subprocess
from dataclasses import dataclass
from pathlib import Path


class AudioPreprocessError(RuntimeError):
    """Raised when ffprobe/ffmpeg preprocessing fails."""


@dataclass(frozen=True, slots=True)
class AudioProbeResult:
    duration_sec: float


def probe_audio(audio_path: str, *, timeout_sec: int) -> AudioProbeResult:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "json",
        str(audio_path),
    ]
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
            timeout=max(30, timeout_sec),
        )
    except subprocess.TimeoutExpired as exc:
        raise AudioPreprocessError("ffprobe timed out") from exc
    if result.returncode != 0:
        details = (result.stderr or result.stdout or "ffprobe failed").strip()
        raise AudioPreprocessError(f"ffprobe failed: {details}")
    try:
        payload = json.loads(result.stdout or "{}")
        duration_sec = max(0.0, float(payload.get("format", {}).get("duration") or 0.0))
    except (ValueError, TypeError, json.JSONDecodeError) as exc:
        raise AudioPreprocessError("ffprobe returned invalid metadata") from exc
    return AudioProbeResult(duration_sec=duration_sec)


def prepare_audio(
    audio_path: str,
    *,
    sample_rate: int,
    timeout_sec: int,
    filters: tuple[str, ...],
) -> str:
    source_path = Path(audio_path)
    prepared_path = source_path.with_name(f"{source_path.stem}.prepared.wav")
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(source_path),
    ]
    if filters:
        command.extend(["-af", ",".join(filters)])
    command.extend(
        [
            "-ac",
            "1",
            "-ar",
            str(sample_rate),
            "-c:a",
            "pcm_s16le",
            str(prepared_path),
        ]
    )
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
            timeout=max(60, timeout_sec),
        )
    except subprocess.TimeoutExpired as exc:
        raise AudioPreprocessError("ffmpeg preprocessing timed out") from exc
    if result.returncode != 0:
        details = (result.stderr or result.stdout or "ffmpeg failed").strip()
        raise AudioPreprocessError(f"ffmpeg preprocessing failed: {details}")
    if not prepared_path.exists() or prepared_path.stat().st_size <= 0:
        raise AudioPreprocessError("ffmpeg preprocessing produced an empty output file")
    return str(prepared_path)


def cleanup_prepared_audio(prepared_audio_path: str | None, *, keep_prepared_audio: bool) -> None:
    if keep_prepared_audio or not prepared_audio_path:
        return
    path = Path(prepared_audio_path)
    if path.exists() and path.is_file():
        path.unlink(missing_ok=True)

