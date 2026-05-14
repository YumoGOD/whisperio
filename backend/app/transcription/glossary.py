from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml

from app.config import Settings

logger = logging.getLogger(__name__)

TermMode = Literal["hard", "soft"]


@dataclass(slots=True)
class GlossaryTerm:
    canonical: str
    mode: TermMode = "soft"
    category: str = "term"
    spoken_forms: list[str] = field(default_factory=list)
    description: str = ""
    replacements: list[dict[str, str]] = field(default_factory=list)
    source: Literal["global", "dynamic"] = "global"
    min_words: int = 0


@dataclass(slots=True)
class ReplacementStats:
    total: int = 0
    by_term: dict[str, int] = field(default_factory=dict)

    def add(self, canonical: str, count: int) -> None:
        if count <= 0:
            return
        self.total += count
        self.by_term[canonical] = self.by_term.get(canonical, 0) + count


@dataclass(slots=True)
class GlossaryContext:
    terms: list[GlossaryTerm]
    initial_prompt: str | None
    hotwords: list[str]
    prompted_terms: list[str] = field(default_factory=list)
    replacement_stats: ReplacementStats = field(default_factory=ReplacementStats)
    hallucination_stats: dict[str, Any] = field(
        default_factory=lambda: {"dropped_segments": 0, "dropped_terms": {}}
    )

    def diagnostics(self) -> dict[str, Any]:
        hard_terms = [term for term in self.terms if term.mode == "hard"]
        soft_terms = [term for term in self.terms if term.mode == "soft"]
        return {
            "prompt": self.initial_prompt,
            "hotwords": self.hotwords,
            "prompted_terms": self.prompted_terms,
            "terms_total": len(self.terms),
            "hard_terms": len(hard_terms),
            "soft_terms": len(soft_terms),
            "global_terms": sum(1 for term in self.terms if term.source == "global"),
            "dynamic_terms": sum(1 for term in self.terms if term.source == "dynamic"),
            "terms": [term_to_dict(term) for term in self.terms],
            "replacement_counts": {
                "total": self.replacement_stats.total,
                "by_term": self.replacement_stats.by_term,
            },
            "hallucination_filter": self.hallucination_stats,
        }


def term_to_dict(term: GlossaryTerm) -> dict[str, Any]:
    return {
        "canonical": term.canonical,
        "mode": term.mode,
        "category": term.category,
        "spoken_forms": term.spoken_forms,
        "description": term.description,
        "source": term.source,
    }


def load_global_terms(path: Path) -> list[GlossaryTerm]:
    if not path.exists():
        logger.info("Файл словаря не найден, словарь пропущен: %s", path)
        return []
    with path.open("r", encoding="utf-8") as file:
        payload = yaml.safe_load(file) or {}
    terms_payload = payload.get("terms") or []
    if not isinstance(terms_payload, list):
        raise ValueError("Файл словаря должен содержать список terms")
    return [parse_term(item, source="global") for item in terms_payload if isinstance(item, dict)]


def parse_term(payload: dict[str, Any], source: Literal["global", "dynamic"]) -> GlossaryTerm:
    canonical = str(payload.get("canonical") or "").strip()
    if not canonical:
        raise ValueError("Термин словаря должен содержать canonical")
    mode = str(payload.get("mode") or "soft").strip().lower()
    if mode not in {"hard", "soft"}:
        raise ValueError(f"Некорректный mode для термина {canonical!r}: {mode!r}")
    # "forms" is a short alias for "spoken_forms"; "note" is a short alias for "description"
    spoken_forms = payload.get("forms") or payload.get("spoken_forms") or []
    description = str(payload.get("note") or payload.get("description") or "").strip()
    replacements = payload.get("replacements") or []
    try:
        min_words = int(payload.get("min_words") or 0)
    except (TypeError, ValueError):
        min_words = 0
    return GlossaryTerm(
        canonical=canonical,
        mode=mode,  # type: ignore[arg-type]
        category=str(payload.get("category") or "term").strip() or "term",
        spoken_forms=[str(item).strip() for item in spoken_forms if str(item).strip()],
        description=description,
        replacements=[
            {"from": str(item.get("from") or ""), "to": str(item.get("to") or canonical)}
            for item in replacements
            if isinstance(item, dict) and str(item.get("from") or "").strip()
        ],
        source=source,
        min_words=max(0, min_words),
    )


def parse_dynamic_terms(raw_text: str | None) -> list[GlossaryTerm]:
    if not raw_text:
        return []
    terms: list[GlossaryTerm] = []
    for line_number, line in enumerate(raw_text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = [part.strip() for part in stripped.split("|")]
        if len(parts) < 3:
            logger.warning("Динамический термин пропущен, строка %s: %s", line_number, stripped)
            continue
        canonical, mode, forms = parts[:3]
        terms.append(
            GlossaryTerm(
                canonical=canonical,
                mode="hard" if mode.lower() == "hard" else "soft",
                spoken_forms=[item.strip() for item in forms.split(",") if item.strip()],
                category="dynamic",
                source="dynamic",
            )
        )
    return terms


def build_glossary_context(settings: Settings, params: dict[str, Any]) -> GlossaryContext:
    global_terms = load_global_terms(settings.glossary_path)
    dynamic_terms = parse_dynamic_terms(params.get("dynamic_terms"))
    terms = merge_terms(dynamic_terms + global_terms)
    prompt_terms = select_prompt_terms(params, terms)
    prompt = build_prompt(settings, params, prompt_terms)
    hotwords = build_hotwords(settings, prompt_terms) if settings.glossary_enable_hotwords else []
    return GlossaryContext(
        terms=terms,
        initial_prompt=prompt,
        hotwords=hotwords,
        prompted_terms=[term.canonical for term in prompt_terms],
    )


def merge_terms(terms: list[GlossaryTerm]) -> list[GlossaryTerm]:
    result: list[GlossaryTerm] = []
    seen: set[str] = set()
    for term in terms:
        key = term.canonical.casefold()
        if key in seen:
            continue
        seen.add(key)
        result.append(term)
    return result


def select_prompt_terms(params: dict[str, Any], terms: list[GlossaryTerm]) -> list[GlossaryTerm]:
    context = " ".join(
        [
            str(params.get("audio_type") or ""),
            str(params.get("audio_context") or ""),
            str(params.get("expected_content") or ""),
            str(params.get("dynamic_terms") or ""),
        ]
    ).casefold()
    selected: list[GlossaryTerm] = []
    for term in terms:
        if term.source == "dynamic" or term.category in {"brand", "abbreviation", "person", "department"}:
            selected.append(term)
            continue
        haystack = " ".join([term.canonical, term.category, *term.spoken_forms, term.description]).casefold()
        if context and any(token and token in context for token in glossary_tokens(haystack)):
            selected.append(term)
    return selected


def glossary_tokens(text: str) -> list[str]:
    return [token for token in re.findall(r"[\wа-яА-ЯёЁ]{4,}", text.casefold()) if token]


def build_prompt(settings: Settings, params: dict[str, Any], terms: list[GlossaryTerm]) -> str | None:
    lines = [
        "Это русскоязычная запись. Распознавай только то, что реально произнесено.",
        "Если бренд, модель, имя, отдел, аббревиатура или термин действительно прозвучали, сохраняй указанное написание. Не вставляй термины из списка, если они не прозвучали.",
    ]
    audio_type = clip_text(params.get("audio_type"), 160)
    audio_context = clip_text(params.get("audio_context"), settings.glossary_context_max_chars)
    expected_content = clip_text(params.get("expected_content"), settings.glossary_context_max_chars)
    if audio_type:
        lines.append(f"Тип записи: {audio_type}.")
    if audio_context:
        lines.append(f"Контекст записи: {audio_context}")
    if expected_content:
        lines.append(f"Ожидаемое наполнение: {expected_content}")

    term_hints: list[str] = []
    for term in terms:
        if term.category == "abbreviation" and term.description:
            term_hints.append(f"{term.canonical} — {term.description}")
        else:
            term_hints.append(term.canonical)

    if term_hints:
        lines.append("Возможные термины и написание: " + "; ".join(term_hints))

    prompt = "\n".join(lines)
    return clip_text(prompt, settings.glossary_prompt_max_chars) or None


def build_hotwords(settings: Settings, terms: list[GlossaryTerm]) -> list[str]:
    hotwords: list[str] = []
    for term in terms:
        hotwords.append(term.canonical)
        hotwords.extend(term.spoken_forms)
    deduped: list[str] = []
    seen: set[str] = set()
    for word in hotwords:
        key = word.casefold()
        if not word.strip() or key in seen:
            continue
        seen.add(key)
        deduped.append(word.strip())
        if len(deduped) >= settings.glossary_hotwords_max:
            break
    return deduped


def apply_hard_normalization(
    text: str,
    context: GlossaryContext,
    enabled: bool = True,
    min_segment_words: int = 0,
) -> str:
    if not enabled or not text:
        return text
    segment_words = len(text.split())
    normalized = text
    for term in context.terms:
        if term.mode != "hard":
            continue
        effective_min = max(term.min_words, min_segment_words)
        if effective_min > 0 and segment_words < effective_min:
            continue
        replacements = term.replacements or generated_replacements(term)
        for replacement in replacements:
            pattern = replacement["from"]
            target = replacement.get("to") or term.canonical
            normalized, count = re.subn(pattern, target, normalized, flags=re.IGNORECASE)
            context.replacement_stats.add(term.canonical, count)
    return normalized


def should_drop_glossary_repetition(
    text: str,
    context: GlossaryContext,
    compression_ratio: float | None,
    threshold: float,
) -> bool:
    if not text or compression_ratio is None or compression_ratio < threshold:
        return False
    repeated_phrase = find_repeated_phrase(text)
    if not repeated_phrase:
        return False
    if not phrase_looks_like_glossary(repeated_phrase, context):
        return False
    term = matched_glossary_term(repeated_phrase, context) or repeated_phrase
    context.hallucination_stats["dropped_segments"] += 1
    dropped_terms = context.hallucination_stats["dropped_terms"]
    dropped_terms[term] = dropped_terms.get(term, 0) + 1
    return True


def find_repeated_phrase(text: str) -> str | None:
    normalized = re.sub(r"\s+", " ", text).strip()
    if not normalized:
        return None
    comma_parts = [part.strip() for part in normalized.split(",") if part.strip()]
    if len(comma_parts) >= 4:
        counts: dict[str, int] = {}
        originals: dict[str, str] = {}
        for part in comma_parts:
            key = re.sub(r"[^\wа-яА-ЯёЁ]+", " ", part.casefold()).strip()
            counts[key] = counts.get(key, 0) + 1
            originals.setdefault(key, part)
        key, count = max(counts.items(), key=lambda item: item[1])
        if count >= 4:
            return originals[key]

    words = normalized.split()
    for size in range(2, min(7, len(words) // 3 + 1)):
        chunks = [" ".join(words[index : index + size]) for index in range(0, len(words) - size + 1, size)]
        if not chunks:
            continue
        first = re.sub(r"[^\wа-яА-ЯёЁ]+", " ", chunks[0].casefold()).strip()
        repeats = sum(1 for chunk in chunks if re.sub(r"[^\wа-яА-ЯёЁ]+", " ", chunk.casefold()).strip() == first)
        if repeats >= 4:
            return chunks[0]
    return None


def phrase_looks_like_glossary(phrase: str, context: GlossaryContext) -> bool:
    normalized_phrase = phrase.casefold()
    for term in context.terms:
        candidates = [term.canonical, *term.spoken_forms]
        for candidate in candidates:
            candidate_tokens = glossary_tokens(candidate)
            if candidate.casefold() in normalized_phrase:
                return True
            if candidate_tokens and sum(1 for token in candidate_tokens if token in normalized_phrase) >= 1:
                return True
    return False


def matched_glossary_term(phrase: str, context: GlossaryContext) -> str | None:
    normalized_phrase = phrase.casefold()
    for term in context.terms:
        candidates = [term.canonical, *term.spoken_forms]
        if any(candidate.casefold() in normalized_phrase for candidate in candidates):
            return term.canonical
    return None


def generated_replacements(term: GlossaryTerm) -> list[dict[str, str]]:
    replacements: list[dict[str, str]] = []
    for form in term.spoken_forms:
        if form.casefold() == term.canonical.casefold():
            continue
        replacements.append({"from": rf"(?<!\w){re.escape(form)}(?!\w)", "to": term.canonical})
    return replacements


def clip_text(value: Any, max_chars: int) -> str:
    text = " ".join(str(value or "").split())
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[: max_chars - 1].rstrip() + "…"
