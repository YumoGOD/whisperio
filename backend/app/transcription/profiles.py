from __future__ import annotations

from copy import deepcopy
from typing import Any

from app.config import Settings


PROFILES: dict[str, dict[str, Any]] = {
    "accuracy_first": {
        "description": "Noisy lecture profile. Prefer recall and boundary safety over speed.",
        "beam_size": 5,
        "best_of": 5,
        "patience": 1.2,
        "temperature": [0.0, 0.2, 0.4, 0.6],
        "compression_ratio_threshold": 2.4,
        "log_prob_threshold": -1.2,
        "no_speech_threshold": 0.85,
        "condition_on_previous_text": False,
        "word_timestamps": True,
        "vad_filter": False,
        "vad_parameters": {
            "threshold": 0.35,
            "min_silence_duration_ms": 1200,
            "speech_pad_ms": 600,
        },
    },
    "speed_balanced": {
        "description": "Cleaner audio profile. Faster decoding with cautious VAD enabled.",
        "beam_size": 3,
        "best_of": 3,
        "patience": 1.0,
        "temperature": [0.0, 0.2],
        "compression_ratio_threshold": 2.4,
        "log_prob_threshold": -1.0,
        "no_speech_threshold": 0.75,
        "condition_on_previous_text": False,
        "word_timestamps": True,
        "vad_filter": True,
        "vad_parameters": {
            "threshold": 0.45,
            "min_silence_duration_ms": 800,
            "speech_pad_ms": 400,
        },
    },
}


def resolve_profile(profile_name: str, settings: Settings) -> dict[str, Any]:
    if profile_name not in PROFILES:
        available = ", ".join(sorted(PROFILES))
        raise ValueError(f"Неизвестный профиль транскрибации '{profile_name}'. Доступные профили: {available}")

    profile = deepcopy(PROFILES[profile_name])
    if settings.vad_filter is not None:
        profile["vad_filter"] = settings.vad_filter
    vad_parameters = profile.setdefault("vad_parameters", {})
    if settings.vad_threshold is not None:
        vad_parameters["threshold"] = settings.vad_threshold
    if settings.vad_min_silence_ms is not None:
        vad_parameters["min_silence_duration_ms"] = settings.vad_min_silence_ms
    if settings.vad_speech_pad_ms is not None:
        vad_parameters["speech_pad_ms"] = settings.vad_speech_pad_ms
    return profile
