from __future__ import annotations

import re
from dataclasses import dataclass


UNINTELLIGIBLE_TEXT = "НЕРАЗБОРЧИВО"

_ARTIFACT_PATTERNS = (
    re.compile(r"^продолжение\s+следует$"),
    re.compile(
        r"^субтитры\s+(?:делал|сделал|сделала|сделали|создавал|создал|создала|создали|подготовил|подготовила|подготовили|подогнал|подогнала|подогнали)\b"
    ),
)


@dataclass(frozen=True, slots=True)
class AudioChunk:
    index: int
    start: float
    end: float

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


def build_chunks(duration_seconds: float, chunk_seconds: int, overlap_seconds: int) -> list[AudioChunk]:
    if duration_seconds <= 0:
        return []
    if chunk_seconds <= 0:
        return [AudioChunk(index=0, start=0.0, end=duration_seconds)]
    if overlap_seconds < 0 or overlap_seconds >= chunk_seconds:
        raise ValueError("Overlap фрагментов должен быть >= 0 и меньше длины фрагмента")

    chunks: list[AudioChunk] = []
    start = 0.0
    index = 0
    step = chunk_seconds - overlap_seconds
    while start < duration_seconds:
        end = min(duration_seconds, start + chunk_seconds)
        chunks.append(AudioChunk(index=index, start=start, end=end))
        if end >= duration_seconds:
            break
        start += step
        index += 1
    return chunks


def normalize_text(text: str) -> str:
    return " ".join(text.split())


def normalize_artifact_text(text: str) -> str:
    text = normalize_text(text).casefold().replace("ё", "е")
    text = re.sub(r"[^\w\s]+", " ", text)
    return normalize_text(text)


def is_transcript_artifact(text: str) -> bool:
    normalized = normalize_artifact_text(text)
    return any(pattern.search(normalized) for pattern in _ARTIFACT_PATTERNS)


def replace_transcript_artifacts(segments: list[dict]) -> tuple[list[dict], dict[str, int]]:
    processed: list[dict] = []
    stats = {"replaced_segments": 0, "collapsed_segments": 0}
    previous_was_artifact = False

    for segment in segments:
        text = normalize_text(segment.get("text", ""))
        if not text:
            previous_was_artifact = False
            continue

        candidate = {**segment, "text": text}
        if not is_transcript_artifact(text):
            processed.append(candidate)
            previous_was_artifact = False
            continue

        stats["replaced_segments"] += 1
        candidate["text"] = UNINTELLIGIBLE_TEXT
        if previous_was_artifact and processed:
            processed[-1]["end"] = max(processed[-1]["end"], candidate["end"])
            stats["collapsed_segments"] += 1
        else:
            processed.append(candidate)
        previous_was_artifact = True

    return processed, stats


GAP_THRESHOLD_SECONDS = 2.0


def _sentence_ends(text: str) -> bool:
    stripped = text.rstrip()
    return bool(stripped) and stripped[-1] in ".!?…"


def _merge_group(group: list[dict]) -> dict:
    texts = [normalize_text(s["text"]) for s in group]
    logprobs = [s["avg_logprob"] for s in group if s.get("avg_logprob") is not None]
    return {
        "id": group[0]["id"],
        "start": group[0]["start"],
        "end": group[-1]["end"],
        "text": " ".join(texts),
        "avg_logprob": round(sum(logprobs) / len(logprobs), 4) if logprobs else None,
        "no_speech_prob": None,
        "compression_ratio": None,
    }


def group_segments_by_sentence(
    segments: list[dict],
    gap_threshold_seconds: float = GAP_THRESHOLD_SECONDS,
) -> list[dict]:
    """Groups consecutive segments into sentence-level blocks.
    Inserts UNINTELLIGIBLE_TEXT marker for silent gaps between blocks."""
    if not segments:
        return []

    result: list[dict] = []
    current_group: list[dict] = []

    for segment in segments:
        text = normalize_text(segment.get("text", ""))
        if not text:
            continue

        if current_group:
            gap = segment["start"] - current_group[-1]["end"]
            if gap > gap_threshold_seconds:
                result.append(_merge_group(current_group))
                result.append({
                    "id": -1,
                    "start": current_group[-1]["end"],
                    "end": segment["start"],
                    "text": UNINTELLIGIBLE_TEXT,
                    "avg_logprob": None,
                    "no_speech_prob": None,
                    "compression_ratio": None,
                })
                current_group = []
            elif _sentence_ends(current_group[-1]["text"]):
                result.append(_merge_group(current_group))
                current_group = []

        current_group.append({**segment, "text": text})

    if current_group:
        result.append(_merge_group(current_group))

    return result


def merge_segments(segments: list[dict], overlap_seconds: int) -> list[dict]:
    merged: list[dict] = []
    for segment in sorted(segments, key=lambda item: (item["start"], item["end"])):
        text = normalize_text(segment.get("text", ""))
        if not text:
            continue
        candidate = {**segment, "text": text}
        if not merged:
            merged.append(candidate)
            continue

        previous = merged[-1]
        boundary_window = max(2.0, float(overlap_seconds) + 2.0)
        starts_near_previous = candidate["start"] <= previous["end"] + boundary_window
        same_text = text.lower() == normalize_text(previous.get("text", "")).lower()
        contained_text = text.lower() in normalize_text(previous.get("text", "")).lower()
        previous_contained = normalize_text(previous.get("text", "")).lower() in text.lower()

        if starts_near_previous and (same_text or contained_text):
            previous["end"] = max(previous["end"], candidate["end"])
            continue
        if starts_near_previous and previous_contained:
            previous.update(candidate)
            continue
        merged.append(candidate)
    return merged
