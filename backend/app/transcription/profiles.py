from __future__ import annotations

from copy import deepcopy
from typing import Any

from app.config import Settings


PROFILES: dict[str, dict[str, Any]] = {
    "accuracy_first": {
        # Тихое/зашумлённое аудио низкого качества. Приоритет — полнота транскрипта.
        #
        # no_speech_threshold=0.90: Whisper пропускает 30-сек окно только если уверен на 90%,
        #   что там тишина — иначе тихая речь с no_speech_prob~0.87 терялась бы.
        # log_prob_threshold=-2.0: принимаем сегменты с низкой уверенностью (деградированный
        #   сигнал даёт плохие log prob); VAD уже отфильтровал чистую тишину.
        # temperature=[0.0..0.8]: 5 ступеней фоллбека — Whisper пробует более высокую
        #   температуру когда compression_ratio слишком высокий (петля/шум).
        # condition_on_previous_text=False: каждое 30-сек окно независимо — плохое окно
        #   не отравляет контекст для следующих.
        # VAD threshold=0.2: Silero считает речью всё с вероятностью ≥20% (вместо 30%),
        #   speech_pad_ms=1000 захватывает 1 с вокруг каждого сегмента.
        "description": "Quiet/low-quality audio profile. Maximum recall, safe segment boundaries.",
        "batch_size": 16,
        "beam_size": 5,
        "best_of": 5,
        "patience": 1.0,
        "temperature": [0.0, 0.2, 0.4, 0.6, 0.8],
        "compression_ratio_threshold": 2.4,
        "log_prob_threshold": -2.0,
        "no_speech_threshold": 0.90,
        "condition_on_previous_text": False,
        "word_timestamps": False,
        "vad_filter": True,
        "vad_parameters": {
            "threshold": 0.2,
            "min_silence_duration_ms": 600,
            "speech_pad_ms": 1000,
        },
    },
    "speed_balanced": {
        "description": "Cleaner audio profile. Faster decoding with VAD enabled.",
        "batch_size": 16,
        "beam_size": 2,
        "best_of": 2,
        "patience": 1.0,
        "temperature": [0.0, 0.2],
        "compression_ratio_threshold": 2.4,
        "log_prob_threshold": -1.5,
        "no_speech_threshold": 0.85,
        "condition_on_previous_text": False,
        "word_timestamps": False,
        "vad_filter": True,
        "vad_parameters": {
            "threshold": 0.25,
            "min_silence_duration_ms": 400,
            "speech_pad_ms": 800,
        },
    },
}


def resolve_profile(profile_name: str, settings: Settings) -> dict[str, Any]:
    if profile_name not in PROFILES:
        available = ", ".join(sorted(PROFILES))
        raise ValueError(f"Неизвестный профиль транскрибации '{profile_name}'. Доступные профили: {available}")

    profile = deepcopy(PROFILES[profile_name])

    # Decode parameter overrides from .env (apply only when explicitly set).
    if settings.whisper_batch_size is not None:
        profile["batch_size"] = settings.whisper_batch_size
    if settings.whisper_no_speech_threshold is not None:
        profile["no_speech_threshold"] = settings.whisper_no_speech_threshold
    if settings.whisper_log_prob_threshold is not None:
        profile["log_prob_threshold"] = settings.whisper_log_prob_threshold
    if settings.whisper_compression_ratio_threshold is not None:
        profile["compression_ratio_threshold"] = settings.whisper_compression_ratio_threshold
    if settings.whisper_beam_size is not None:
        profile["beam_size"] = settings.whisper_beam_size
    if settings.whisper_best_of is not None:
        profile["best_of"] = settings.whisper_best_of
    if settings.whisper_patience is not None:
        profile["patience"] = settings.whisper_patience
    if settings.whisper_condition_on_previous_text is not None:
        profile["condition_on_previous_text"] = settings.whisper_condition_on_previous_text
    if settings.whisper_word_timestamps is not None:
        profile["word_timestamps"] = settings.whisper_word_timestamps

    # VAD overrides.
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
