import math
from collections import Counter

from app.transcription.segmenter import AudioWindow
from app.transcription.settings import QualitySettings


def segment_confidence(avg_logprob: float) -> float:
    return max(0.0, min(1.0, math.exp(min(0.0, avg_logprob))))


def find_rescue_windows(
    segment_rows: list[dict[str, float | str]],
    *,
    quality: QualitySettings,
) -> tuple[list[AudioWindow], dict[str, int]]:
    sorted_rows = sorted(
        segment_rows,
        key=lambda item: (float(item.get("start_sec", 0.0)), float(item.get("end_sec", 0.0))),
    )
    windows: list[AudioWindow] = []
    reasons = Counter()
    for segment in sorted_rows:
        avg_logprob = float(segment.get("avg_logprob", 0.0))
        no_speech_prob = float(segment.get("no_speech_prob", 0.0))
        confidence = segment_confidence(avg_logprob)
        is_rescue = False
        if confidence < quality.rescue_confidence_threshold:
            reasons["LOW_CONFIDENCE"] += 1
            is_rescue = True
        if no_speech_prob > quality.rescue_no_speech_threshold:
            reasons["HIGH_NO_SPEECH_PROB"] += 1
            is_rescue = True
        if avg_logprob < quality.rescue_logprob_threshold:
            reasons["LOW_LOGPROB"] += 1
            is_rescue = True
        if not is_rescue:
            continue
        start_sec = max(0.0, float(segment.get("start_sec", 0.0)) - quality.rescue_padding_sec)
        end_sec = max(start_sec, float(segment.get("end_sec", start_sec)) + quality.rescue_padding_sec)
        windows.append(AudioWindow(start_sec=start_sec, end_sec=end_sec))

    if not windows:
        return [], dict(reasons)

    windows.sort(key=lambda item: (item.start_sec, item.end_sec))
    merged: list[AudioWindow] = [windows[0]]
    for window in windows[1:]:
        prev = merged[-1]
        if window.start_sec <= prev.end_sec + 0.05:
            merged[-1] = AudioWindow(start_sec=prev.start_sec, end_sec=max(prev.end_sec, window.end_sec))
            continue
        merged.append(window)
    if len(merged) > quality.rescue_max_segments:
        merged = merged[: quality.rescue_max_segments]
        reasons["RESCUE_LIMIT_REACHED"] += 1
    return merged, dict(reasons)


def _rows_score(rows: list[dict[str, float | str]]) -> float:
    if not rows:
        return -1.0
    confidence_values: list[float] = []
    coverage = 0.0
    char_count = 0
    for row in rows:
        start_sec = float(row.get("start_sec", 0.0))
        end_sec = float(row.get("end_sec", start_sec))
        coverage += max(0.0, end_sec - start_sec)
        text = str(row.get("text", "")).strip()
        char_count += len(text)
        confidence_values.append(segment_confidence(float(row.get("avg_logprob", 0.0))))
    avg_confidence = sum(confidence_values) / max(1, len(confidence_values))
    length_bonus = min(1.0, char_count / 140.0)
    coverage_bonus = min(1.0, coverage / 12.0)
    return (avg_confidence * 2.0) + length_bonus + coverage_bonus


def choose_best_window_segments(
    *,
    original_rows: list[dict[str, float | str]],
    rescue_rows: list[dict[str, float | str]],
    min_gain: float,
) -> tuple[list[dict[str, float | str]], bool, dict[str, float]]:
    original_score = _rows_score(original_rows)
    rescue_score = _rows_score(rescue_rows)
    selected_rescue = False
    chosen_rows = original_rows
    if rescue_rows and rescue_score >= (original_score + min_gain):
        selected_rescue = True
        chosen_rows = rescue_rows
    elif rescue_rows:
        original_chars = sum(len(str(row.get("text", "")).strip()) for row in original_rows)
        rescue_chars = sum(len(str(row.get("text", "")).strip()) for row in rescue_rows)
        if rescue_chars > original_chars and rescue_score >= (original_score - 0.02):
            selected_rescue = True
            chosen_rows = rescue_rows
    return chosen_rows, selected_rescue, {
        "original_score": round(original_score, 4),
        "rescue_score": round(rescue_score, 4),
    }


def replace_rows_in_window(
    *,
    all_rows: list[dict[str, float | str]],
    window: AudioWindow,
    replacement_rows: list[dict[str, float | str]],
) -> list[dict[str, float | str]]:
    retained: list[dict[str, float | str]] = []
    for row in all_rows:
        start_sec = float(row.get("start_sec", 0.0))
        end_sec = float(row.get("end_sec", start_sec))
        overlaps = start_sec < window.end_sec and end_sec > window.start_sec
        if not overlaps:
            retained.append(row)
    retained.extend(replacement_rows)
    retained.sort(key=lambda item: (float(item.get("start_sec", 0.0)), float(item.get("end_sec", 0.0))))
    return retained


def decorate_segments(
    segment_rows: list[dict[str, float | str]],
    *,
    quality: QualitySettings,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    decorated: list[dict[str, object]] = []
    flagged = 0
    reason_counts = Counter()

    for segment in sorted(
        segment_rows,
        key=lambda item: (float(item.get("start_sec", 0.0)), float(item.get("end_sec", 0.0))),
    ):
        text = str(segment.get("text", "")).strip()
        if not text:
            continue
        start_sec = max(0.0, float(segment.get("start_sec", 0.0)))
        end_sec = max(start_sec, float(segment.get("end_sec", start_sec)))
        avg_logprob = float(segment.get("avg_logprob", 0.0))
        no_speech_prob = float(segment.get("no_speech_prob", 0.0))
        compression_ratio = float(segment.get("compression_ratio", 0.0))
        confidence = segment_confidence(avg_logprob)

        reasons: list[str] = []
        if confidence < quality.min_confidence:
            reasons.append("LOW_CONFIDENCE")
        if no_speech_prob > quality.high_no_speech_prob:
            reasons.append("HIGH_NO_SPEECH_PROB")
        if avg_logprob < quality.low_logprob:
            reasons.append("LOW_LOGPROB")
        if compression_ratio > quality.high_compression_ratio:
            reasons.append("HIGH_COMPRESSION_RATIO")
        if reasons:
            flagged += 1
            for reason in reasons:
                reason_counts[reason] += 1

        flags: dict[str, object] = {
            "avg_logprob": round(avg_logprob, 4),
            "no_speech_prob": round(no_speech_prob, 4),
            "compression_ratio": round(compression_ratio, 4),
        }
        if reasons:
            flags["quality_reasons"] = reasons

        decorated.append(
            {
                "start_sec": start_sec,
                "end_sec": end_sec,
                "label": "speech",
                "text": text,
                "confidence": round(confidence, 4),
                "quality_flags": flags,
            }
        )

    stats = {
        "inserted_labels": {"silence": 0, "music": 0, "unclear": 0, "speech": len(decorated)},
        "filtered_segments": 0,
        "flagged_segments": flagged,
        "flagged_reason_counts": dict(sorted(reason_counts.items())),
        "drop_filtered_segments": False,
    }
    return decorated, stats

