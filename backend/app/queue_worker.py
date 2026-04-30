import asyncio
import inspect
import json
import logging
import math
import os
import subprocess
import tempfile
import time
import wave
from array import array
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

try:
    import psutil
except ImportError:  # pragma: no cover - optional runtime dependency fallback
    psutil = None

try:
    import resource
except ImportError:  # pragma: no cover - not available on Windows
    resource = None
from faster_whisper import WhisperModel

from app.db import get_connection, get_queue_stats
from app.logging_utils import bind_log_context, log_event, reset_log_context
from app.models import JobStage, JobStatus

CONFIG_LOGGER = logging.getLogger("whisperio.worker.config")


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _warn_config(name: str, value: object, default: object, reason: str) -> None:
    log_event(
        CONFIG_LOGGER,
        event="invalid_whisper_config",
        level=logging.WARNING,
        component="worker.config",
        name=name,
        raw_value=value,
        fallback=default,
        reason=reason,
    )


def getenv_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        _warn_config(name, value, default, "invalid_integer")
        return default


def getenv_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        _warn_config(name, value, default, "invalid_float")
        return default


def getenv_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    _warn_config(name, value, default, "invalid_boolean")
    return default


def normalize_choice(name: str, value: str, allowed: set[str], default: str) -> str:
    normalized = value.strip().lower()
    if normalized in allowed:
        return normalized
    _warn_config(name, value, default, f"unsupported_value:{','.join(sorted(allowed))}")
    return default


def clamp_config_float(
    name: str,
    value: float,
    *,
    min_value: float,
    max_value: float,
    default: float,
) -> float:
    if not math.isfinite(value):
        _warn_config(name, value, default, "non_finite_float")
        return default
    if value < min_value:
        _warn_config(name, value, min_value, f"below_min:{min_value}")
        return min_value
    if value > max_value:
        _warn_config(name, value, max_value, f"above_max:{max_value}")
        return max_value
    return value


def ensure_config_min_int(name: str, value: int, *, min_value: int, default: int) -> int:
    if value >= min_value:
        return value
    fallback = max(min_value, default)
    _warn_config(name, value, fallback, f"below_min:{min_value}")
    return fallback


def ensure_config_min_float(name: str, value: float, *, min_value: float, default: float) -> float:
    if not math.isfinite(value):
        _warn_config(name, value, default, "non_finite_float")
        return default
    if value >= min_value:
        return value
    fallback = max(min_value, default)
    _warn_config(name, value, fallback, f"below_min:{min_value}")
    return fallback


def ensure_finite_float(name: str, value: float, default: float) -> float:
    if math.isfinite(value):
        return value
    _warn_config(name, value, default, "non_finite_float")
    return default


def clamp_progress(value: float | None) -> float | None:
    if value is None:
        return None
    return max(0.0, min(100.0, float(value)))


class AudioPreprocessError(RuntimeError):
    pass


class QualityGuardError(RuntimeError):
    pass


TAG_SILENCE = "<Тишина>"
TAG_MUSIC = "<Музыка>"
TAG_UNINTELLIGIBLE = "<Неразборчиво>"
LABEL_SPEECH = "speech"
LABEL_SILENCE = "silence"
LABEL_MUSIC = "music"
LABEL_UNCLEAR = "unclear"


class TranscriptionWorker:
    def __init__(self) -> None:
        self.logger = logging.getLogger("whisperio.worker")
        self.current_request_id: str | None = None
        self.model_name = os.getenv("WHISPER_MODEL_SIZE", "medium")
        self.model_device = os.getenv("WHISPER_DEVICE", "cpu")
        self.model_compute_type = os.getenv("WHISPER_COMPUTE_TYPE", "int8")
        self.model_cpu_threads = max(1, getenv_int("WHISPER_CPU_THREADS", 12))
        self.model_workers = max(1, getenv_int("WHISPER_MODEL_WORKERS", 4))
        self.model: WhisperModel | None = None
        self._model_lock = asyncio.Lock()
        self._transcribe_lock = asyncio.Lock()
        language_raw = os.getenv("WHISPER_LANGUAGE", "ru")
        self.language = language_raw.strip() or "ru"
        if not language_raw.strip():
            _warn_config("WHISPER_LANGUAGE", language_raw, "ru", "empty_language")
        self.task = normalize_choice(
            "WHISPER_TASK",
            os.getenv("WHISPER_TASK", "transcribe"),
            {"transcribe", "translate"},
            "transcribe",
        )
        self.beam_size = ensure_config_min_int(
            "WHISPER_BEAM_SIZE",
            getenv_int("WHISPER_BEAM_SIZE", 1),
            min_value=1,
            default=1,
        )
        self.best_of = ensure_config_min_int(
            "WHISPER_BEST_OF",
            getenv_int("WHISPER_BEST_OF", self.beam_size),
            min_value=1,
            default=max(1, self.beam_size),
        )
        self.temperature = clamp_config_float(
            "WHISPER_TEMPERATURE",
            getenv_float("WHISPER_TEMPERATURE", 0.0),
            min_value=0.0,
            max_value=1.0,
            default=0.0,
        )
        self.initial_prompt = os.getenv("WHISPER_INITIAL_PROMPT", "").strip() or None
        self.condition_on_previous_text = getenv_bool("WHISPER_CONDITION_ON_PREVIOUS_TEXT", True)
        self.vad_filter = getenv_bool("WHISPER_VAD_FILTER", True)
        self.vad_threshold = clamp_config_float(
            "WHISPER_VAD_THRESHOLD",
            getenv_float("WHISPER_VAD_THRESHOLD", 0.6),
            min_value=0.0,
            max_value=1.0,
            default=0.6,
        )
        self.vad_min_silence_duration_ms = ensure_config_min_int(
            "WHISPER_VAD_MIN_SILENCE_DURATION_MS",
            getenv_int("WHISPER_VAD_MIN_SILENCE_DURATION_MS", 700),
            min_value=0,
            default=700,
        )
        self.vad_speech_pad_ms = ensure_config_min_int(
            "WHISPER_VAD_SPEECH_PAD_MS",
            getenv_int("WHISPER_VAD_SPEECH_PAD_MS", 300),
            min_value=0,
            default=300,
        )
        self.no_speech_threshold = clamp_config_float(
            "WHISPER_NO_SPEECH_THRESHOLD",
            getenv_float("WHISPER_NO_SPEECH_THRESHOLD", 0.45),
            min_value=0.0,
            max_value=1.0,
            default=0.45,
        )
        self.log_prob_threshold = ensure_finite_float(
            "WHISPER_LOG_PROB_THRESHOLD",
            getenv_float("WHISPER_LOG_PROB_THRESHOLD", -1.0),
            -1.0,
        )
        self.compression_ratio_threshold = ensure_config_min_float(
            "WHISPER_COMPRESSION_RATIO_THRESHOLD",
            getenv_float("WHISPER_COMPRESSION_RATIO_THRESHOLD", 2.4),
            min_value=0.0,
            default=2.4,
        )
        self.enable_quality_fallback = getenv_bool("WHISPER_ENABLE_QUALITY_FALLBACK", True)
        self.min_unique_ratio = getenv_float("WHISPER_MIN_UNIQUE_SEGMENT_RATIO", 0.25)
        self.max_top_repeat_ratio = getenv_float("WHISPER_MAX_TOP_REPEAT_RATIO", 0.7)
        self.max_prompt_match_ratio = getenv_float("WHISPER_MAX_PROMPT_MATCH_RATIO", 0.5)
        self.preprocess_sample_rate = getenv_int("WHISPER_PREPROCESS_SAMPLE_RATE", 16000)
        self.preprocess_timeout_sec = getenv_int("WHISPER_PREPROCESS_TIMEOUT_SEC", 1800)
        self.enhance_profile = normalize_choice(
            "WHISPER_ENHANCE_PROFILE",
            os.getenv("WHISPER_ENHANCE_PROFILE", "balanced"),
            {"off", "balanced", "aggressive"},
            "balanced",
        )
        self.long_audio_min_sec = max(0, getenv_int("WHISPER_LONG_AUDIO_MIN_SEC", 3600))
        self.long_audio_chunk_sec = max(60, getenv_int("WHISPER_LONG_AUDIO_CHUNK_SEC", 900))
        self.long_audio_overlap_sec = max(
            0.0,
            min(
                float(self.long_audio_chunk_sec - 5),
                getenv_float("WHISPER_LONG_AUDIO_OVERLAP_SEC", 2.0),
            ),
        )
        self.chunk_length_s = max(1, getenv_int("WHISPER_CHUNK_LENGTH_S", 20))
        self.keep_prepared_audio = getenv_bool("WHISPER_KEEP_PREPARED_AUDIO", True)
        self.enable_tags = getenv_bool("WHISPER_ENABLE_TAGS", True)
        self.tag_silence_min_sec = max(0.0, getenv_float("WHISPER_TAG_SILENCE_MIN_SEC", 3.0))
        self.tag_silence_dbfs = getenv_float("WHISPER_TAG_SILENCE_DBFS", -38.0)
        self.tag_music_min_sec = max(0.0, getenv_float("WHISPER_TAG_MUSIC_MIN_SEC", 3.0))
        self.tag_music_max_zcr = max(0.0, getenv_float("WHISPER_TAG_MUSIC_MAX_ZCR", 0.08))
        self.tag_music_max_energy_variation = max(
            0.0, getenv_float("WHISPER_TAG_MUSIC_MAX_ENERGY_VARIATION", 0.35)
        )
        self.tag_unintelligible_max_avg_logprob = getenv_float(
            "WHISPER_TAG_UNINTELLIGIBLE_MAX_AVG_LOGPROB", -1.25
        )
        self.tag_unintelligible_min_no_speech_prob = getenv_float(
            "WHISPER_TAG_UNINTELLIGIBLE_MIN_NO_SPEECH_PROB", 0.60
        )
        self.tag_unintelligible_max_compression_ratio = getenv_float(
            "WHISPER_TAG_UNINTELLIGIBLE_MAX_COMPRESSION_RATIO", 2.4
        )
        self.sla_rtf_threshold = max(0.01, getenv_float("WHISPER_SLA_RTF_THRESHOLD", 0.25))
        self.resource_snapshot_interval_sec = max(
            1, getenv_int("LOG_SNAPSHOT_INTERVAL_SEC", 10)
        )
        self._last_resource_snapshot_ts = 0.0

    def process_claimed_job(self, job_id: str) -> None:
        asyncio.run(self._process_job(job_id))

    async def _get_model(self) -> WhisperModel:
        if self.model is not None:
            return self.model
        async with self._model_lock:
            if self.model is None:
                self.model = await asyncio.to_thread(
                    WhisperModel,
                    self.model_name,
                    device=self.model_device,
                    compute_type=self.model_compute_type,
                    cpu_threads=self.model_cpu_threads,
                    num_workers=self.model_workers,
                )
        return self.model

    async def _run_transcribe(
        self,
        model: WhisperModel,
        audio_path: str,
        initial_prompt: str | None,
        condition_on_previous_text: bool,
    ):
        # A single shared WhisperModel instance is used by multiple queue workers.
        # Serialize inference calls to avoid undefined behavior from concurrent transcribe.
        async with self._transcribe_lock:
            return await asyncio.to_thread(
                self._transcribe_audio,
                model,
                audio_path,
                initial_prompt,
                condition_on_previous_text,
            )

    def _transcribe_audio(
        self,
        model: WhisperModel,
        audio_path: str,
        initial_prompt: str | None,
        condition_on_previous_text: bool,
    ):
        vad_parameters = None
        if self.vad_filter:
            vad_parameters = {
                "threshold": self.vad_threshold,
                "min_silence_duration_ms": self.vad_min_silence_duration_ms,
                "speech_pad_ms": self.vad_speech_pad_ms,
            }
        transcribe_kwargs = {
            "language": self.language,
            "task": self.task,
            "beam_size": self.beam_size,
            "best_of": self.best_of,
            "temperature": self.temperature,
            "vad_filter": self.vad_filter,
            "vad_parameters": vad_parameters,
            "condition_on_previous_text": condition_on_previous_text,
            "initial_prompt": initial_prompt,
            "no_speech_threshold": self.no_speech_threshold,
            "log_prob_threshold": self.log_prob_threshold,
            "compression_ratio_threshold": self.compression_ratio_threshold,
        }
        if "chunk_length" in inspect.signature(model.transcribe).parameters:
            transcribe_kwargs["chunk_length"] = self.chunk_length_s
        if "best_of" not in inspect.signature(model.transcribe).parameters:
            transcribe_kwargs.pop("best_of", None)
        segments, info = model.transcribe(audio_path, **transcribe_kwargs)
        segment_rows = []
        for segment in segments:
            segment_rows.append(
                {
                    "start_sec": float(segment.start),
                    "end_sec": float(segment.end),
                    "text": segment.text.strip(),
                    "avg_logprob": float(getattr(segment, "avg_logprob", 0.0)),
                    "no_speech_prob": float(getattr(segment, "no_speech_prob", 0.0)),
                    "compression_ratio": float(getattr(segment, "compression_ratio", 0.0)),
                }
            )
        return segment_rows, info

    def _probe_audio(self, audio_path: str) -> dict[str, float]:
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
                timeout=max(30, min(self.preprocess_timeout_sec, 300)),
            )
        except subprocess.TimeoutExpired as exc:
            raise AudioPreprocessError("ffprobe timed out") from exc
        if result.returncode != 0:
            details = (result.stderr or result.stdout or "ffprobe failed").strip()
            raise AudioPreprocessError(f"ffprobe failed: {details}")
        try:
            payload = json.loads(result.stdout or "{}")
            duration_raw = payload.get("format", {}).get("duration")
            duration_sec = max(0.0, float(duration_raw or 0.0))
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            raise AudioPreprocessError("ffprobe returned invalid metadata") from exc
        return {"duration_sec": duration_sec}

    def _build_enhance_filters(self) -> list[str]:
        if self.enhance_profile == "off":
            return []
        if self.enhance_profile == "aggressive":
            return [
                "highpass=f=100",
                "lowpass=f=7000",
                "afftdn=nf=-20",
                "dynaudnorm=f=100:g=23",
            ]
        return [
            "highpass=f=80",
            "lowpass=f=7800",
            "afftdn=nf=-25",
            "dynaudnorm=f=150:g=15",
        ]

    def _prepare_audio(self, audio_path: str) -> str:
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
        enhance_filters = self._build_enhance_filters()
        if enhance_filters:
            command.extend(["-af", ",".join(enhance_filters)])
        command.extend(
            [
                "-ac",
                "1",
                "-ar",
                str(self.preprocess_sample_rate),
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
                timeout=self.preprocess_timeout_sec,
            )
        except subprocess.TimeoutExpired as exc:
            raise AudioPreprocessError("ffmpeg preprocessing timed out") from exc
        if result.returncode != 0:
            details = (result.stderr or result.stdout or "ffmpeg failed").strip()
            raise AudioPreprocessError(f"ffmpeg preprocessing failed: {details}")
        if not prepared_path.exists() or prepared_path.stat().st_size == 0:
            raise AudioPreprocessError("ffmpeg preprocessing produced an empty output file")
        return str(prepared_path)

    def _cleanup_prepared_audio(self, prepared_audio_path: str | None) -> None:
        if self.keep_prepared_audio or not prepared_audio_path:
            return
        path = Path(prepared_audio_path)
        if path.exists():
            path.unlink(missing_ok=True)

    def _is_delete_requested(self, conn, job_id: str) -> bool:
        row = conn.execute(
            "SELECT delete_requested FROM transcription_jobs WHERE id = ?",
            (job_id,),
        ).fetchone()
        if not row:
            return False
        return bool(row["delete_requested"])

    def _finalize_delete(self, job_id: str, prepared_audio_path: str | None) -> None:
        with get_connection() as conn:
            row = conn.execute(
                "SELECT stored_path, prepared_audio_path FROM transcription_jobs WHERE id = ?",
                (job_id,),
            ).fetchone()
            if not row:
                return
            stored_path = row["stored_path"]
            kept_prepared_path = row["prepared_audio_path"]
            conn.execute("DELETE FROM transcription_jobs WHERE id = ?", (job_id,))

        for candidate in (stored_path, kept_prepared_path, prepared_audio_path):
            if candidate:
                path = Path(candidate)
                if path.exists() and path.is_file():
                    path.unlink(missing_ok=True)

        self._log_event(
            event="job_deleted_after_processing_step",
            job_id=job_id,
            stage=JobStage.COMPLETED,
            status=JobStatus.DONE,
            progress=100.0,
        )

    def _analyze_quality(
        self,
        segment_rows: list[dict[str, float | str]],
        prompt: str | None,
    ) -> dict[str, object]:
        if not segment_rows:
            return {
                "segment_count": 0,
                "unique_text_count": 0,
                "unique_ratio": 0.0,
                "top_repeat_ratio": 0.0,
                "prompt_match_ratio": 0.0,
                "top_repeat_text": None,
                "reasons": ["NO_SEGMENTS"],
                "is_anomaly": True,
            }

        texts = [
            str(segment["text"]).strip()
            for segment in segment_rows
            if str(segment.get("text", "")).strip()
        ]
        if not texts:
            return {
                "segment_count": len(segment_rows),
                "unique_text_count": 0,
                "unique_ratio": 0.0,
                "top_repeat_ratio": 0.0,
                "prompt_match_ratio": 0.0,
                "top_repeat_text": None,
                "reasons": ["EMPTY_SEGMENTS"],
                "is_anomaly": True,
            }

        counts = Counter(texts)
        top_text, top_count = counts.most_common(1)[0]
        segment_count = len(texts)
        unique_count = len(counts)
        unique_ratio = unique_count / segment_count
        top_repeat_ratio = top_count / segment_count
        prompt_match_ratio = 0.0
        normalized_prompt = prompt.strip() if prompt else None
        if normalized_prompt:
            prompt_hits = sum(1 for text in texts if text == normalized_prompt)
            prompt_match_ratio = prompt_hits / segment_count

        reasons: list[str] = []
        if unique_ratio < self.min_unique_ratio:
            reasons.append("LOW_UNIQUE_SEGMENT_RATIO")
        if top_repeat_ratio > self.max_top_repeat_ratio:
            reasons.append("HIGH_TOP_REPEAT_RATIO")
        if normalized_prompt and prompt_match_ratio > self.max_prompt_match_ratio:
            reasons.append("HIGH_PROMPT_MATCH_RATIO")

        return {
            "segment_count": segment_count,
            "unique_text_count": unique_count,
            "unique_ratio": round(unique_ratio, 4),
            "top_repeat_ratio": round(top_repeat_ratio, 4),
            "prompt_match_ratio": round(prompt_match_ratio, 4),
            "top_repeat_text": top_text,
            "reasons": reasons,
            "is_anomaly": bool(reasons),
        }

    def _read_pcm_interval(
        self,
        wav_file: wave.Wave_read,
        sample_rate: int,
        start_sec: float,
        end_sec: float,
    ) -> array:
        start_frame = max(0, int(start_sec * sample_rate))
        end_frame = max(start_frame, int(end_sec * sample_rate))
        frame_count = end_frame - start_frame
        if frame_count <= 0:
            return array("h")
        wav_file.setpos(start_frame)
        frames = wav_file.readframes(frame_count)
        samples = array("h")
        samples.frombytes(frames)
        return samples

    def _audio_interval_metrics(self, interval_samples: array) -> dict[str, float]:
        sample_count = len(interval_samples)
        if sample_count == 0:
            return {"rms_dbfs": -120.0, "zcr": 0.0, "energy_variation": 0.0}

        squares_sum = 0.0
        zero_crossings = 0
        prev = interval_samples[0]
        frame_size = 800  # ~50ms at 16kHz.
        frame_rms_values: list[float] = []
        frame_squares_sum = 0.0
        frame_samples = 0
        for idx, current in enumerate(interval_samples):
            value = float(current)
            squares_sum += value * value
            frame_squares_sum += value * value
            frame_samples += 1
            if idx > 0 and ((prev >= 0 > current) or (prev < 0 <= current)):
                zero_crossings += 1
            prev = current
            if frame_samples == frame_size:
                frame_rms_values.append(math.sqrt(frame_squares_sum / frame_samples) / 32768.0)
                frame_squares_sum = 0.0
                frame_samples = 0
        if frame_samples > 0:
            frame_rms_values.append(math.sqrt(frame_squares_sum / frame_samples) / 32768.0)

        rms = math.sqrt(squares_sum / sample_count) / 32768.0
        rms_dbfs = 20.0 * math.log10(max(rms, 1e-8))
        zcr = zero_crossings / max(1, sample_count - 1)
        energy_mean = sum(frame_rms_values) / max(1, len(frame_rms_values))
        if energy_mean <= 1e-8 or len(frame_rms_values) < 2:
            energy_variation = 0.0
        else:
            variance = sum((item - energy_mean) ** 2 for item in frame_rms_values) / (
                len(frame_rms_values) - 1
            )
            energy_variation = math.sqrt(max(variance, 0.0)) / energy_mean

        return {
            "rms_dbfs": rms_dbfs,
            "zcr": zcr,
            "energy_variation": energy_variation,
        }

    def _is_unintelligible(self, segment: dict[str, float | str]) -> bool:
        avg_logprob = float(segment.get("avg_logprob", 0.0))
        no_speech_prob = float(segment.get("no_speech_prob", 0.0))
        compression_ratio = float(segment.get("compression_ratio", 0.0))
        text = str(segment.get("text", "")).strip()
        if not text:
            return False
        return (
            avg_logprob <= self.tag_unintelligible_max_avg_logprob
            and no_speech_prob >= self.tag_unintelligible_min_no_speech_prob
            and compression_ratio >= self.tag_unintelligible_max_compression_ratio
        )

    def _classify_gap(
        self,
        wav_file: wave.Wave_read,
        sample_rate: int,
        gap_start_sec: float,
        gap_end_sec: float,
    ) -> tuple[str, dict[str, float]]:
        gap_duration = max(0.0, gap_end_sec - gap_start_sec)
        interval = self._read_pcm_interval(wav_file, sample_rate, gap_start_sec, gap_end_sec)
        metrics = self._audio_interval_metrics(interval)
        if gap_duration <= 0.0:
            return LABEL_SILENCE, metrics
        if metrics["rms_dbfs"] <= self.tag_silence_dbfs or gap_duration < self.tag_silence_min_sec:
            return LABEL_SILENCE, metrics
        if (
            gap_duration >= self.tag_music_min_sec
            and metrics["zcr"] <= self.tag_music_max_zcr
            and metrics["energy_variation"] <= self.tag_music_max_energy_variation
        ):
            return LABEL_MUSIC, metrics
        return LABEL_UNCLEAR, metrics

    def _asr_segment_to_timeline_item(
        self,
        start_sec: float,
        end_sec: float,
        segment: dict[str, float | str],
    ) -> dict[str, object]:
        text = str(segment.get("text", "")).strip()
        avg_logprob = float(segment.get("avg_logprob", 0.0))
        label = LABEL_UNCLEAR if self._is_unintelligible(segment) else LABEL_SPEECH
        if label == LABEL_UNCLEAR:
            text = TAG_UNINTELLIGIBLE
        confidence = max(0.0, min(1.0, math.exp(min(0.0, avg_logprob))))
        return {
            "start_sec": start_sec,
            "end_sec": end_sec,
            "label": label,
            "text": text,
            "confidence": round(confidence, 4),
            "quality_flags": {
                "avg_logprob": round(avg_logprob, 4),
                "no_speech_prob": round(float(segment.get("no_speech_prob", 0.0)), 4),
                "compression_ratio": round(float(segment.get("compression_ratio", 0.0)), 4),
            },
        }

    def _make_non_speech_segment(
        self,
        start_sec: float,
        end_sec: float,
        label: str,
        metrics: dict[str, float],
    ) -> dict[str, object]:
        label_to_text = {
            LABEL_SILENCE: TAG_SILENCE,
            LABEL_MUSIC: TAG_MUSIC,
            LABEL_UNCLEAR: TAG_UNINTELLIGIBLE,
        }
        return {
            "start_sec": start_sec,
            "end_sec": end_sec,
            "label": label,
            "text": label_to_text.get(label, TAG_UNINTELLIGIBLE),
            "confidence": 1.0,
            "quality_flags": {
                "rms_dbfs": round(metrics["rms_dbfs"], 4),
                "zcr": round(metrics["zcr"], 6),
                "energy_variation": round(metrics["energy_variation"], 6),
            },
        }

    def _build_chunk_plan(self, audio_duration_sec: float) -> list[tuple[float, float]]:
        duration = max(0.0, float(audio_duration_sec))
        if (
            duration <= 0.0
            or self.long_audio_chunk_sec <= 0
            or duration < float(self.long_audio_min_sec)
        ):
            return [(0.0, duration)]
        overlap = min(self.long_audio_overlap_sec, max(0.0, float(self.long_audio_chunk_sec - 5)))
        windows: list[tuple[float, float]] = []
        cursor = 0.0
        while cursor < duration:
            end_sec = min(duration, cursor + float(self.long_audio_chunk_sec))
            windows.append((cursor, end_sec))
            if end_sec >= duration:
                break
            cursor = max(0.0, end_sec - overlap)
        return windows

    def _extract_chunk(
        self,
        prepared_audio_path: str,
        chunk_path: str,
        start_sec: float,
        end_sec: float,
    ) -> None:
        duration = max(0.0, end_sec - start_sec)
        command = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-ss",
            str(start_sec),
            "-t",
            str(duration),
            "-i",
            prepared_audio_path,
            "-ac",
            "1",
            "-ar",
            str(self.preprocess_sample_rate),
            "-c:a",
            "pcm_s16le",
            chunk_path,
        ]
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
            timeout=max(30, min(self.preprocess_timeout_sec, 600)),
        )
        if result.returncode != 0:
            details = (result.stderr or result.stdout or "ffmpeg chunk failed").strip()
            raise AudioPreprocessError(f"ffmpeg chunk extraction failed: {details}")

    async def _transcribe_with_chunking(
        self,
        model: WhisperModel,
        prepared_audio_path: str,
        prompt: str | None,
        condition_on_previous_text: bool,
        audio_duration_sec: float,
    ) -> tuple[list[dict[str, float | str]], dict[str, object]]:
        chunk_plan = self._build_chunk_plan(audio_duration_sec)
        if len(chunk_plan) == 1:
            rows, info = await self._run_transcribe(
                model,
                prepared_audio_path,
                prompt,
                condition_on_previous_text,
            )
            return rows, {"duration": float(getattr(info, "duration", audio_duration_sec) or audio_duration_sec)}
        merged_rows: list[dict[str, float | str]] = []
        with tempfile.TemporaryDirectory(prefix="whisperio-chunks-") as tmp_dir:
            for chunk_idx, (start_sec, end_sec) in enumerate(chunk_plan):
                chunk_path = str(Path(tmp_dir) / f"chunk-{chunk_idx:04d}.wav")
                await asyncio.to_thread(
                    self._extract_chunk,
                    prepared_audio_path,
                    chunk_path,
                    start_sec,
                    end_sec,
                )
                chunk_rows, _ = await self._run_transcribe(
                    model,
                    chunk_path,
                    prompt,
                    condition_on_previous_text,
                )
                for row in chunk_rows:
                    merged_rows.append(
                        {
                            **row,
                            "start_sec": max(0.0, float(row["start_sec"]) + start_sec),
                            "end_sec": max(0.0, float(row["end_sec"]) + start_sec),
                        }
                    )
        merged_rows.sort(key=lambda item: (float(item["start_sec"]), float(item["end_sec"])))
        deduped: list[dict[str, float | str]] = []
        for row in merged_rows:
            start_sec = max(0.0, float(row["start_sec"]))
            end_sec = max(start_sec, float(row["end_sec"]))
            if not deduped:
                deduped.append({**row, "start_sec": start_sec, "end_sec": end_sec})
                continue
            last_end = float(deduped[-1]["end_sec"])
            if end_sec <= last_end:
                continue
            if start_sec < last_end:
                start_sec = last_end
            deduped.append({**row, "start_sec": start_sec, "end_sec": end_sec})
        return deduped, {"duration": audio_duration_sec}

    def _decorate_segments(
        self,
        raw_segments: list[dict[str, float | str]],
        audio_path: str,
        audio_duration_sec: float,
    ) -> tuple[list[dict[str, object]], dict[str, object]]:
        if not self.enable_tags:
            decorated: list[dict[str, object]] = []
            for segment in raw_segments:
                start_sec = max(0.0, float(segment["start_sec"]))
                end_sec = max(start_sec, float(segment["end_sec"]))
                text = str(segment.get("text", "")).strip()
                if text:
                    decorated.append(self._asr_segment_to_timeline_item(start_sec, end_sec, segment))
            return decorated, {
                "inserted_labels": {
                    LABEL_SILENCE: 0,
                    LABEL_MUSIC: 0,
                    LABEL_UNCLEAR: 0,
                    LABEL_SPEECH: sum(1 for segment in decorated if segment["label"] == LABEL_SPEECH),
                },
                "gap_metrics": [],
            }

        with wave.open(audio_path, "rb") as wav_file:
            sample_rate = wav_file.getframerate()
            sample_width = wav_file.getsampwidth()
            if sample_width != 2:
                raise AudioPreprocessError("prepared audio must be 16-bit PCM")
            decorated: list[dict[str, object]] = []
            inserted_counts = {
                LABEL_SILENCE: 0,
                LABEL_MUSIC: 0,
                LABEL_UNCLEAR: 0,
                LABEL_SPEECH: 0,
            }
            gap_metrics: list[dict[str, float | str]] = []
            previous_end = 0.0
            sorted_segments = sorted(
                raw_segments,
                key=lambda item: (
                    max(0.0, float(item["start_sec"])),
                    max(0.0, float(item["end_sec"])),
                ),
            )
            for segment in sorted_segments:
                start_sec = max(0.0, float(segment["start_sec"]))
                end_sec = max(start_sec, float(segment["end_sec"]))
                if start_sec > previous_end:
                    gap_label, metrics = self._classify_gap(
                        wav_file,
                        sample_rate,
                        previous_end,
                        start_sec,
                    )
                    decorated.append(
                        self._make_non_speech_segment(previous_end, start_sec, gap_label, metrics)
                    )
                    inserted_counts[gap_label] += 1
                    gap_metrics.append(
                        {
                            "start_sec": round(previous_end, 3),
                            "end_sec": round(start_sec, 3),
                            "label": gap_label,
                            "rms_dbfs": round(metrics["rms_dbfs"], 4),
                            "zcr": round(metrics["zcr"], 6),
                            "energy_variation": round(metrics["energy_variation"], 6),
                        }
                    )
                if end_sec <= previous_end:
                    continue
                if start_sec < previous_end:
                    start_sec = previous_end
                text = str(segment.get("text", "")).strip()
                if not text:
                    previous_end = end_sec
                    continue
                timeline_item = self._asr_segment_to_timeline_item(start_sec, end_sec, segment)
                decorated.append(timeline_item)
                inserted_counts[str(timeline_item["label"])] += 1
                previous_end = end_sec

            tail_end = max(previous_end, float(audio_duration_sec or 0.0))
            if tail_end > previous_end:
                tail_label, metrics = self._classify_gap(
                    wav_file,
                    sample_rate,
                    previous_end,
                    tail_end,
                )
                decorated.append(
                    self._make_non_speech_segment(previous_end, tail_end, tail_label, metrics)
                )
                inserted_counts[tail_label] += 1
                gap_metrics.append(
                    {
                        "start_sec": round(previous_end, 3),
                        "end_sec": round(tail_end, 3),
                        "label": tail_label,
                        "rms_dbfs": round(metrics["rms_dbfs"], 4),
                        "zcr": round(metrics["zcr"], 6),
                        "energy_variation": round(metrics["energy_variation"], 6),
                    }
                )
        return decorated, {"inserted_labels": inserted_counts, "gap_metrics": gap_metrics}

    async def _process_job(self, job_id: str) -> None:
        started_wall_time = time.perf_counter()
        prepared_audio_path: str | None = None
        source_duration_sec: float | None = None
        decode_duration_ms: int | None = None
        preprocess_duration_ms: int | None = None
        enhance_duration_ms: int | None = None
        decorate_duration_ms: int | None = None
        segment_duration_ms: int | None = None
        db_write_duration_ms: int | None = None
        quality_flags_json: str | None = None
        request_id: str | None = None
        audio_size_bytes: int | None = None
        log_context_token = bind_log_context(worker_pid=os.getpid(), job_id=job_id)
        with get_connection() as conn:
            job = conn.execute(
                "SELECT stored_path, request_id, delete_requested FROM transcription_jobs WHERE id = ?",
                (job_id,),
            ).fetchone()
            if not job:
                self._log_event(
                    event="job_not_found",
                    job_id=job_id,
                    stage=JobStage.FAILED,
                    error_code="JOB_NOT_FOUND",
                )
                reset_log_context(log_context_token)
                return
            request_id = job["request_id"]
            reset_log_context(log_context_token)
            log_context_token = bind_log_context(
                worker_pid=os.getpid(),
                job_id=job_id,
                request_id=request_id,
            )

            self._update_job(
                conn=conn,
                job_id=job_id,
                status=JobStatus.PROCESSING,
                stage=JobStage.PREPARING,
                progress=5.0,
                status_message="Подготовка к распознаванию",
                started_at=utc_now_iso(),
                clear_error=True,
            )

        audio_path = job["stored_path"]
        if bool(job["delete_requested"]):
            self._finalize_delete(job_id, prepared_audio_path)
            reset_log_context(log_context_token)
            return
        if Path(audio_path).exists():
            audio_size_bytes = Path(audio_path).stat().st_size
        self._log_event(
            event="job_started",
            job_id=job_id,
            stage=JobStage.PREPARING,
            status=JobStatus.PROCESSING,
            progress=5.0,
            extra_fields={
                "request_id": request_id,
                "audio_size_bytes": audio_size_bytes,
                "decode_profile": "accuracy_first",
                "beam_size": self.beam_size,
                "best_of": self.best_of,
                "chunk_length_s": self.chunk_length_s,
                "vad_filter": self.vad_filter,
            },
        )
        self._log_resource_snapshot(job_id=job_id, stage=JobStage.PREPARING, force=True)
        try:
            decode_started = time.perf_counter()
            source_probe = await asyncio.to_thread(self._probe_audio, audio_path)
            decode_duration_ms = int((time.perf_counter() - decode_started) * 1000)
            source_duration_sec = float(source_probe.get("duration_sec", 0.0))
            with get_connection() as conn:
                self._update_job(
                    conn=conn,
                    job_id=job_id,
                    status=JobStatus.PROCESSING,
                    stage=JobStage.PREPROCESSING,
                    progress=10.0,
                    status_message="Подготовка аудио (mono/16kHz PCM)",
                    decode_duration_ms=decode_duration_ms,
                )
            preprocess_started = time.perf_counter()
            self._log_event(
                event="stage_started",
                job_id=job_id,
                stage=JobStage.PREPROCESSING,
                status=JobStatus.PROCESSING,
                progress=10.0,
                extra_fields={"stage_name": "preprocess"},
            )
            prepared_audio_path = await asyncio.to_thread(self._prepare_audio, audio_path)
            preprocess_duration_ms = int((time.perf_counter() - preprocess_started) * 1000)
            enhance_duration_ms = preprocess_duration_ms
            with get_connection() as conn:
                self._update_job(
                    conn=conn,
                    job_id=job_id,
                    status=JobStatus.PROCESSING,
                    stage=JobStage.PREPROCESSING,
                    progress=14.0,
                    status_message="Аудио подготовлено",
                    preprocess_duration_ms=preprocess_duration_ms,
                    enhance_duration_ms=enhance_duration_ms,
                    decode_duration_ms=decode_duration_ms,
                    prepared_audio_path=prepared_audio_path if self.keep_prepared_audio else None,
                )
                if self._is_delete_requested(conn, job_id):
                    self._finalize_delete(job_id, prepared_audio_path)
                    return
            self._log_event(
                event="stage_finished",
                job_id=job_id,
                stage=JobStage.PREPROCESSING,
                status=JobStatus.PROCESSING,
                progress=14.0,
                duration_ms=preprocess_duration_ms,
                extra_fields={
                    "stage_name": "preprocess",
                    "enhance_profile": self.enhance_profile,
                    "decode_duration_ms": decode_duration_ms,
                    "prepared_audio_file": (
                        Path(prepared_audio_path).name if prepared_audio_path else None
                    ),
                },
            )

            with get_connection() as conn:
                self._update_job(
                    conn=conn,
                    job_id=job_id,
                    status=JobStatus.PROCESSING,
                    stage=JobStage.TRANSCRIBING,
                    progress=20.0,
                    status_message="Выполняется распознавание речи",
                )
            model = await self._get_model()
            transcribe_started = time.perf_counter()
            self._log_event(
                event="stage_started",
                job_id=job_id,
                stage=JobStage.TRANSCRIBING,
                status=JobStatus.PROCESSING,
                progress=20.0,
                extra_fields={"stage_name": "transcribe"},
            )
            primary_rows, info = await self._transcribe_with_chunking(
                model,
                prepared_audio_path or audio_path,
                self.initial_prompt,
                self.condition_on_previous_text,
                source_duration_sec or 0.0,
            )
            quality_primary = self._analyze_quality(primary_rows, self.initial_prompt)
            fallback_used = False
            raw_segment_rows = primary_rows
            quality_payload: dict[str, object] = {"primary": quality_primary, "fallback_used": False}

            if bool(quality_primary["is_anomaly"]):
                self._log_event(
                    event="quality_anomaly_detected",
                    job_id=job_id,
                    stage=JobStage.TRANSCRIBING,
                    status=JobStatus.PROCESSING,
                    progress=70.0,
                    extra_fields={
                        "quality_reasons": ",".join(quality_primary["reasons"]),
                        "unique_ratio": quality_primary["unique_ratio"],
                        "top_repeat_ratio": quality_primary["top_repeat_ratio"],
                        "prompt_match_ratio": quality_primary["prompt_match_ratio"],
                    },
                )
                if not self.enable_quality_fallback:
                    raise QualityGuardError(
                        f"quality guard failed: {','.join(quality_primary['reasons'])}"
                    )
                fallback_used = True
                fallback_rows, info = await self._transcribe_with_chunking(
                    model,
                    prepared_audio_path or audio_path,
                    None,
                    False,
                    source_duration_sec or 0.0,
                )
                quality_fallback = self._analyze_quality(fallback_rows, None)
                quality_payload["fallback_used"] = True
                quality_payload["fallback"] = quality_fallback
                if bool(quality_fallback["is_anomaly"]):
                    raise QualityGuardError(
                        f"quality guard fallback failed: {','.join(quality_fallback['reasons'])}"
                    )
                raw_segment_rows = fallback_rows
                self._log_event(
                    event="quality_fallback_succeeded",
                    job_id=job_id,
                    stage=JobStage.TRANSCRIBING,
                    status=JobStatus.PROCESSING,
                    progress=75.0,
                    extra_fields={
                        "unique_ratio": quality_fallback["unique_ratio"],
                        "top_repeat_ratio": quality_fallback["top_repeat_ratio"],
                        "prompt_match_ratio": quality_fallback["prompt_match_ratio"],
                    },
                )

            transcribe_duration_ms = int((time.perf_counter() - transcribe_started) * 1000)
            decorate_started = time.perf_counter()
            decorated_rows, tagging_stats = await asyncio.to_thread(
                self._decorate_segments,
                raw_segment_rows,
                prepared_audio_path or audio_path,
                float(info.get("duration", source_duration_sec or 0.0)),
            )
            decorate_duration_ms = int((time.perf_counter() - decorate_started) * 1000)
            segment_duration_ms = decorate_duration_ms
            quality_payload["tagging"] = tagging_stats
            quality_flags_json = json.dumps(quality_payload, ensure_ascii=False)
            segment_rows = [
                (
                    job_id,
                    idx,
                    float(segment["start_sec"]),
                    float(segment["end_sec"]),
                    str(segment["text"]),
                    str(segment["label"]),
                    float(segment["confidence"]),
                    json.dumps(segment["quality_flags"], ensure_ascii=False),
                )
                for idx, segment in enumerate(decorated_rows)
            ]
            with get_connection() as conn:
                self._update_job(
                    conn=conn,
                    job_id=job_id,
                    status=JobStatus.PROCESSING,
                    stage=JobStage.SAVING_SEGMENTS,
                    progress=85.0,
                    status_message=f"Сохранение сегментов: {len(segment_rows)}",
                    transcribe_duration_ms=transcribe_duration_ms,
                    preprocess_duration_ms=preprocess_duration_ms,
                    decode_duration_ms=decode_duration_ms,
                    enhance_duration_ms=enhance_duration_ms,
                    segment_duration_ms=segment_duration_ms,
                    prepared_audio_path=prepared_audio_path if self.keep_prepared_audio else None,
                    quality_flags=quality_flags_json,
                )
                if self._is_delete_requested(conn, job_id):
                    self._finalize_delete(job_id, prepared_audio_path)
                    return
            self._log_event(
                event="stage_finished",
                job_id=job_id,
                stage=JobStage.TRANSCRIBING,
                status=JobStatus.PROCESSING,
                progress=80.0,
                duration_ms=transcribe_duration_ms,
                segment_count=len(segment_rows),
                audio_duration_sec=float(info.get("duration", source_duration_sec or 0.0)),
                extra_fields={
                    "stage_name": "transcribe",
                    "fallback_used": fallback_used,
                    "preprocess_duration_ms": preprocess_duration_ms,
                    "decorate_duration_ms": decorate_duration_ms,
                    "tag_silence_count": tagging_stats["inserted_labels"][LABEL_SILENCE],
                    "tag_music_count": tagging_stats["inserted_labels"][LABEL_MUSIC],
                    "tag_unclear_count": tagging_stats["inserted_labels"][LABEL_UNCLEAR],
                    "speech_count": tagging_stats["inserted_labels"][LABEL_SPEECH],
                },
            )
            persist_started = time.perf_counter()

            with get_connection() as conn:
                conn.execute(
                    "DELETE FROM transcription_segments WHERE job_id = ?",
                    (job_id,),
                )
                if segment_rows:
                    conn.executemany(
                        """
                        INSERT INTO transcription_segments
                        (job_id, idx, start_sec, end_sec, text, label, confidence, quality_flags)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        segment_rows,
                    )
                wall_duration_ms = int((time.perf_counter() - started_wall_time) * 1000)
                self._update_job(
                    conn=conn,
                    job_id=job_id,
                    status=JobStatus.DONE,
                    stage=JobStage.COMPLETED,
                    progress=100.0,
                    status_message="Транскрибация завершена",
                    finished_at=utc_now_iso(),
                    duration_sec=float(info.get("duration", source_duration_sec or 0.0)),
                    processing_duration_ms=wall_duration_ms,
                    transcribe_duration_ms=transcribe_duration_ms,
                    preprocess_duration_ms=preprocess_duration_ms,
                    decode_duration_ms=decode_duration_ms,
                    enhance_duration_ms=enhance_duration_ms,
                    segment_duration_ms=segment_duration_ms,
                    decorate_duration_ms=decorate_duration_ms,
                    prepared_audio_path=prepared_audio_path if self.keep_prepared_audio else None,
                    quality_flags=quality_flags_json,
                    clear_error=True,
                )
            persist_duration_ms = int((time.perf_counter() - persist_started) * 1000)
            db_write_duration_ms = persist_duration_ms
            with get_connection() as conn:
                self._update_job(
                    conn=conn,
                    job_id=job_id,
                    persist_duration_ms=persist_duration_ms,
                )
            rtf = None
            effective_speed_x = None
            audio_duration_sec = float(info.get("duration", source_duration_sec or 0.0))
            if audio_duration_sec > 0:
                processing_sec = wall_duration_ms / 1000.0
                rtf = round(processing_sec / audio_duration_sec, 6)
                effective_speed_x = round(audio_duration_sec / max(processing_sec, 1e-6), 6)
            self._log_event(
                event="job_completed",
                job_id=job_id,
                stage=JobStage.COMPLETED,
                status=JobStatus.DONE,
                progress=100.0,
                duration_ms=wall_duration_ms,
                persist_duration_ms=persist_duration_ms,
                segment_count=len(segment_rows),
                extra_fields={
                    "preprocess_duration_ms": preprocess_duration_ms,
                    "decode_duration_ms": decode_duration_ms,
                    "enhance_duration_ms": enhance_duration_ms,
                    "transcribe_duration_ms": transcribe_duration_ms,
                    "decorate_duration_ms": decorate_duration_ms,
                    "segment_duration_ms": segment_duration_ms,
                    "db_write_duration_ms": db_write_duration_ms,
                    "total_duration_ms": wall_duration_ms,
                    "audio_duration_sec": audio_duration_sec,
                    "rtf": rtf,
                    "effective_speed_x": effective_speed_x,
                },
            )
            self._emit_sla_event_if_needed(
                rtf=rtf,
                job_id=job_id,
                stage=JobStage.COMPLETED,
                fallback_used=fallback_used,
            )
        except Exception as exc:  # noqa: BLE001
            wall_duration_ms = int((time.perf_counter() - started_wall_time) * 1000)
            self._log_event(
                event="stage_failed",
                job_id=job_id,
                stage=JobStage.FAILED,
                status=JobStatus.FAILED,
                progress=100.0,
                error_code=self._infer_error_code(exc),
                error_message=self._safe_error_message(exc),
                extra_fields={"failed_stage": "pipeline"},
            )
            with get_connection() as conn:
                status_message = "Ошибка обработки"
                if isinstance(exc, AudioPreprocessError):
                    status_message = "Ошибка подготовки аудио"
                elif isinstance(exc, QualityGuardError):
                    status_message = "Результат не прошел проверку качества"
                self._update_job(
                    conn=conn,
                    job_id=job_id,
                    status=JobStatus.FAILED,
                    stage=JobStage.FAILED,
                    progress=100.0,
                    status_message=status_message,
                    finished_at=utc_now_iso(),
                    error=str(exc),
                    error_code=self._infer_error_code(exc),
                    processing_duration_ms=wall_duration_ms,
                    preprocess_duration_ms=preprocess_duration_ms,
                    decode_duration_ms=decode_duration_ms,
                    enhance_duration_ms=enhance_duration_ms,
                    segment_duration_ms=segment_duration_ms,
                    decorate_duration_ms=decorate_duration_ms,
                    prepared_audio_path=prepared_audio_path if self.keep_prepared_audio else None,
                    quality_flags=quality_flags_json,
                )
            self._log_event(
                event="job_failed",
                job_id=job_id,
                stage=JobStage.FAILED,
                status=JobStatus.FAILED,
                progress=100.0,
                duration_ms=wall_duration_ms,
                error_code=self._infer_error_code(exc),
                error_message=self._safe_error_message(exc),
                extra_fields={
                    "preprocess_duration_ms": preprocess_duration_ms,
                    "decode_duration_ms": decode_duration_ms,
                    "enhance_duration_ms": enhance_duration_ms,
                    "segment_duration_ms": segment_duration_ms,
                },
            )
        finally:
            self._cleanup_prepared_audio(prepared_audio_path)
            self._log_resource_snapshot(job_id=job_id, stage=JobStage.COMPLETED, force=True)
            reset_log_context(log_context_token)

    def _update_job(
        self,
        conn,
        job_id: str,
        *,
        status: JobStatus | None = None,
        stage: JobStage | None = None,
        progress: float | None = None,
        status_message: str | None = None,
        started_at: str | None = None,
        finished_at: str | None = None,
        error: str | None = None,
        error_code: str | None = None,
        duration_sec: float | None = None,
        processing_duration_ms: int | None = None,
        transcribe_duration_ms: int | None = None,
        preprocess_duration_ms: int | None = None,
        decode_duration_ms: int | None = None,
        enhance_duration_ms: int | None = None,
        segment_duration_ms: int | None = None,
        decorate_duration_ms: int | None = None,
        persist_duration_ms: int | None = None,
        prepared_audio_path: str | None = None,
        quality_flags: str | None = None,
        clear_error: bool = False,
    ) -> None:
        fields: list[str] = []
        values: list[object] = []
        updates = {
            "status": status,
            "stage": stage,
            "progress": clamp_progress(progress),
            "status_message": status_message,
            "started_at": started_at,
            "finished_at": finished_at,
            "error": error,
            "error_code": error_code,
            "duration_sec": duration_sec,
            "processing_duration_ms": processing_duration_ms,
            "transcribe_duration_ms": transcribe_duration_ms,
            "preprocess_duration_ms": preprocess_duration_ms,
            "decode_duration_ms": decode_duration_ms,
            "enhance_duration_ms": enhance_duration_ms,
            "segment_duration_ms": segment_duration_ms,
            "decorate_duration_ms": decorate_duration_ms,
            "persist_duration_ms": persist_duration_ms,
            "prepared_audio_path": prepared_audio_path,
            "quality_flags": quality_flags,
        }
        for key, value in updates.items():
            if value is not None:
                fields.append(f"{key} = ?")
                values.append(value)
        if clear_error:
            fields.append("error = NULL")
            fields.append("error_code = NULL")
        if not fields:
            return
        values.append(job_id)
        conn.execute(
            f"""
            UPDATE transcription_jobs
            SET {", ".join(fields)}
            WHERE id = ?
            """,
            values,
        )

    def _infer_error_code(self, exc: Exception) -> str:
        if isinstance(exc, AudioPreprocessError):
            return "AUDIO_PREPROCESS_FAILED"
        if isinstance(exc, QualityGuardError):
            return "LOW_QUALITY_TRANSCRIPTION"
        error_type = type(exc).__name__.lower()
        message = str(exc).lower()
        if "cuda" in message or "out of memory" in message:
            return "MODEL_RUNTIME_ERROR"
        if "no such file" in message or "not found" in message:
            return "AUDIO_FILE_NOT_FOUND"
        if "sqlite" in error_type or "database" in message:
            return "DB_WRITE_FAILED"
        return "TRANSCRIPTION_FAILED"

    def _safe_error_message(self, exc: Exception) -> str:
        if isinstance(exc, AudioPreprocessError):
            return "Audio preprocessing failed"
        if isinstance(exc, QualityGuardError):
            return "Quality guard failed"
        return type(exc).__name__

    def _emit_sla_event_if_needed(
        self,
        *,
        rtf: float | None,
        job_id: str,
        stage: JobStage,
        fallback_used: bool,
    ) -> None:
        if rtf is None:
            return
        if rtf > self.sla_rtf_threshold:
            reason_code = "decode_slowdown"
            queue_stats = get_queue_stats()
            if queue_stats["queued"] > 0:
                reason_code = "cpu_saturation"
            if fallback_used:
                reason_code = "audio_complexity_high"
            self._log_event(
                event="sla_drift_detected",
                job_id=job_id,
                stage=stage,
                status=JobStatus.DONE,
                extra_fields={
                    "reason_code": reason_code,
                    "rtf": rtf,
                    "rtf_threshold": self.sla_rtf_threshold,
                    "queue_backlog": queue_stats["queued"],
                    "processing_jobs": queue_stats["processing"],
                    "fallback_used": fallback_used,
                },
            )

    def _log_resource_snapshot(self, *, job_id: str, stage: JobStage, force: bool = False) -> None:
        if psutil is None:
            return
        now = time.monotonic()
        if not force and (now - self._last_resource_snapshot_ts) < self.resource_snapshot_interval_sec:
            return
        self._last_resource_snapshot_ts = now
        process = psutil.Process(os.getpid())
        mem = process.memory_info()
        cpu_percent = process.cpu_percent(interval=None)
        num_threads = process.num_threads()
        open_files_count = len(process.open_files())
        ctx_switches = process.num_ctx_switches()
        max_rss_kb = None
        if resource is not None:
            ru = resource.getrusage(resource.RUSAGE_SELF)
            max_rss_kb = int(getattr(ru, "ru_maxrss", 0))
        self._log_event(
            event="worker_resource_snapshot",
            job_id=job_id,
            stage=stage,
            extra_fields={
                "worker_pid": os.getpid(),
                "cpu_percent": cpu_percent,
                "rss_bytes": mem.rss,
                "vms_bytes": mem.vms,
                "threads": num_threads,
                "open_files_count": open_files_count,
                "ctx_switches_voluntary": ctx_switches.voluntary,
                "ctx_switches_involuntary": ctx_switches.involuntary,
                "max_rss_kb": max_rss_kb,
            },
        )
        high_watermark_mb = getenv_int("LOG_RESOURCE_RSS_WARN_MB", 12288)
        if mem.rss > high_watermark_mb * 1024 * 1024:
            self._log_event(
                event="resource_pressure_warning",
                job_id=job_id,
                stage=stage,
                status=JobStatus.PROCESSING,
                extra_fields={
                    "reason_code": "rss_high_watermark",
                    "rss_bytes": mem.rss,
                    "rss_warn_mb": high_watermark_mb,
                },
            )

    def _log_event(
        self,
        *,
        event: str,
        job_id: str,
        stage: JobStage,
        status: JobStatus | None = None,
        progress: float | None = None,
        duration_ms: int | None = None,
        error_code: str | None = None,
        error_message: str | None = None,
        queue_size: int | None = None,
        segment_count: int | None = None,
        audio_duration_sec: float | None = None,
        persist_duration_ms: int | None = None,
        extra_fields: dict[str, object] | None = None,
    ) -> None:
        payload = {
            "event": event,
            "job_id": job_id,
            "stage": str(stage),
            "status": str(status) if status else None,
            "progress": clamp_progress(progress),
            "duration_ms": duration_ms,
            "error_code": error_code,
            "error_message": error_message,
            "queue_size": queue_size,
            "segment_count": segment_count,
            "audio_duration_sec": audio_duration_sec,
            "persist_duration_ms": persist_duration_ms,
        }
        if extra_fields:
            payload.update(extra_fields)
        clean_payload = {
            key: value for key, value in payload.items() if value is not None
        }
        clean_payload.pop("event", None)
        log_event(
            self.logger,
            event=event,
            component="worker.pipeline",
            **clean_payload,
        )
