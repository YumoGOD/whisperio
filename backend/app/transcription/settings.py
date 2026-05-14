import logging
import os
from dataclasses import dataclass

from app.logging_utils import log_event

CONFIG_LOGGER = logging.getLogger("whisperio.transcription.settings")


@dataclass(frozen=True, slots=True)
class DecodeSettings:
    beam_size: int
    best_of: int
    temperature: float
    condition_on_previous_text: bool
    no_speech_threshold: float
    log_prob_threshold: float
    compression_ratio_threshold: float


@dataclass(frozen=True, slots=True)
class VadSettings:
    threshold: float
    min_speech_duration_ms: int
    min_silence_duration_ms: int
    speech_pad_ms: int
    max_segment_duration_sec: float
    merge_gap_sec: float


@dataclass(frozen=True, slots=True)
class QualitySettings:
    min_confidence: float
    high_no_speech_prob: float
    low_logprob: float
    high_compression_ratio: float
    rescue_confidence_threshold: float
    rescue_no_speech_threshold: float
    rescue_logprob_threshold: float
    rescue_max_segments: int
    rescue_padding_sec: float
    rescue_min_score_gain: float


@dataclass(frozen=True, slots=True)
class PreprocessSettings:
    sample_rate: int
    timeout_sec: int
    keep_prepared_audio: bool
    ffmpeg_filters: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ProfileSettings:
    name: str
    preprocess_filters: tuple[str, ...]
    vad: VadSettings
    primary_decode: DecodeSettings
    rescue_decode: DecodeSettings
    quality: QualitySettings


@dataclass(frozen=True, slots=True)
class TranscriptionSettings:
    model_name: str
    model_device: str
    model_compute_type: str
    model_cpu_threads: int
    model_workers: int
    language: str
    task: str
    profile: ProfileSettings
    preprocess: PreprocessSettings


def _env_int(name: str, default: int, *, min_value: int = 0) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        parsed = int(value)
    except ValueError:
        log_event(
            CONFIG_LOGGER,
            event="invalid_transcribe_config",
            level=logging.WARNING,
            component="transcription.settings",
            name=name,
            raw_value=value,
            fallback=default,
        )
        return default
    if parsed < min_value:
        return max(min_value, default)
    return parsed


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return default


def _env_choice(name: str, default: str, *, allowed: set[str]) -> str:
    value = (os.getenv(name) or "").strip().lower()
    if not value:
        return default
    if value in allowed:
        return value
    return default


def _legacy_language_default() -> str:
    legacy = (os.getenv("WHISPER_LANGUAGE") or "").strip()
    if legacy:
        return legacy
    return "ru"


def _build_profiles() -> dict[str, ProfileSettings]:
    robust = ProfileSettings(
        name="robust_long_noisy",
        preprocess_filters=(
            "highpass=f=55",
            "lowpass=f=7600",
            "afftdn=nr=20:nf=-32:tn=1",
            "dynaudnorm=f=120:g=16:p=0.95:m=14",
            "alimiter=limit=0.98",
        ),
        vad=VadSettings(
            threshold=0.32,
            min_speech_duration_ms=120,
            min_silence_duration_ms=250,
            speech_pad_ms=700,
            max_segment_duration_sec=30.0,
            merge_gap_sec=0.35,
        ),
        primary_decode=DecodeSettings(
            beam_size=2,
            best_of=2,
            temperature=0.0,
            condition_on_previous_text=False,
            no_speech_threshold=0.72,
            log_prob_threshold=-1.3,
            compression_ratio_threshold=2.8,
        ),
        rescue_decode=DecodeSettings(
            beam_size=5,
            best_of=5,
            temperature=0.2,
            condition_on_previous_text=False,
            no_speech_threshold=0.9,
            log_prob_threshold=-2.4,
            compression_ratio_threshold=4.2,
        ),
        quality=QualitySettings(
            min_confidence=0.16,
            high_no_speech_prob=0.9,
            low_logprob=-2.7,
            high_compression_ratio=3.8,
            rescue_confidence_threshold=0.24,
            rescue_no_speech_threshold=0.95,
            rescue_logprob_threshold=-3.0,
            rescue_max_segments=120,
            rescue_padding_sec=0.6,
            rescue_min_score_gain=0.02,
        ),
    )
    balanced = ProfileSettings(
        name="balanced",
        preprocess_filters=(
            "highpass=f=70",
            "lowpass=f=7600",
            "afftdn=nr=16:nf=-30:tn=1",
            "dynaudnorm=f=140:g=12:p=0.95:m=12",
        ),
        vad=VadSettings(
            threshold=0.4,
            min_speech_duration_ms=150,
            min_silence_duration_ms=350,
            speech_pad_ms=500,
            max_segment_duration_sec=28.0,
            merge_gap_sec=0.25,
        ),
        primary_decode=DecodeSettings(
            beam_size=2,
            best_of=2,
            temperature=0.0,
            condition_on_previous_text=False,
            no_speech_threshold=0.7,
            log_prob_threshold=-1.1,
            compression_ratio_threshold=2.6,
        ),
        rescue_decode=DecodeSettings(
            beam_size=4,
            best_of=4,
            temperature=0.1,
            condition_on_previous_text=False,
            no_speech_threshold=0.88,
            log_prob_threshold=-2.2,
            compression_ratio_threshold=3.8,
        ),
        quality=QualitySettings(
            min_confidence=0.18,
            high_no_speech_prob=0.92,
            low_logprob=-2.5,
            high_compression_ratio=3.6,
            rescue_confidence_threshold=0.22,
            rescue_no_speech_threshold=0.96,
            rescue_logprob_threshold=-2.8,
            rescue_max_segments=80,
            rescue_padding_sec=0.5,
            rescue_min_score_gain=0.03,
        ),
    )
    return {robust.name: robust, balanced.name: balanced}


def load_transcription_settings() -> TranscriptionSettings:
    profiles = _build_profiles()
    profile_name = _env_choice(
        "TRANSCRIBE_PROFILE",
        default="robust_long_noisy",
        allowed=set(profiles),
    )
    profile = profiles[profile_name]
    language = (os.getenv("TRANSCRIBE_LANGUAGE") or _legacy_language_default()).strip() or "ru"
    model_name = (os.getenv("WHISPER_MODEL_SIZE") or "large-v3").strip() or "large-v3"
    task = _env_choice("WHISPER_TASK", "transcribe", allowed={"transcribe", "translate"})
    preprocess = PreprocessSettings(
        sample_rate=_env_int("WHISPER_PREPROCESS_SAMPLE_RATE", 16000, min_value=8000),
        timeout_sec=_env_int("WHISPER_PREPROCESS_TIMEOUT_SEC", 1800, min_value=30),
        keep_prepared_audio=_env_bool("WHISPER_KEEP_PREPARED_AUDIO", True),
        ffmpeg_filters=profile.preprocess_filters,
    )
    settings = TranscriptionSettings(
        model_name=model_name,
        model_device=(os.getenv("WHISPER_DEVICE") or "cpu").strip() or "cpu",
        model_compute_type=(os.getenv("WHISPER_COMPUTE_TYPE") or "int8").strip() or "int8",
        model_cpu_threads=_env_int("WHISPER_CPU_THREADS", 8, min_value=1),
        model_workers=_env_int("WHISPER_MODEL_WORKERS", 1, min_value=1),
        language=language,
        task=task,
        profile=profile,
        preprocess=preprocess,
    )
    log_event(
        CONFIG_LOGGER,
        event="transcription_settings_loaded",
        component="transcription.settings",
        profile=settings.profile.name,
        model=settings.model_name,
        device=settings.model_device,
        compute_type=settings.model_compute_type,
        cpu_threads=settings.model_cpu_threads,
        task=settings.task,
        language=settings.language,
    )
    return settings

