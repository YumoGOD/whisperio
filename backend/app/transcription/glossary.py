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
    replacement_stats: ReplacementStats = field(default_factory=ReplacementStats)

    def diagnostics(self) -> dict[str, Any]:
        hard_terms = [term for term in self.terms if term.mode == "hard"]
        soft_terms = [term for term in self.terms if term.mode == "soft"]
        return {
            "prompt": self.initial_prompt,
            "hotwords": self.hotwords,
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
    spoken_forms = payload.get("spoken_forms") or []
    replacements = payload.get("replacements") or []
    return GlossaryTerm(
        canonical=canonical,
        mode=mode,  # type: ignore[arg-type]
        category=str(payload.get("category") or "term").strip() or "term",
        spoken_forms=[str(item).strip() for item in spoken_forms if str(item).strip()],
        description=str(payload.get("description") or "").strip(),
        replacements=[
            {"from": str(item.get("from") or ""), "to": str(item.get("to") or canonical)}
            for item in replacements
            if isinstance(item, dict) and str(item.get("from") or "").strip()
        ],
        source=source,
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
    prompt = build_prompt(settings, params, terms)
    hotwords = build_hotwords(settings, terms)
    return GlossaryContext(terms=terms, initial_prompt=prompt, hotwords=hotwords)


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


def build_prompt(settings: Settings, params: dict[str, Any], terms: list[GlossaryTerm]) -> str | None:
    lines = [
        "Это русскоязычная запись. Сохраняй бренды, модели, имена, отделы, аббревиатуры и термины в указанном написании.",
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
        forms = ", ".join(term.spoken_forms[:4])
        description = f" — {term.description}" if term.description else ""
        if forms:
            term_hints.append(f"{term.canonical} ({forms}){description}")
        else:
            term_hints.append(f"{term.canonical}{description}")

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


def apply_hard_normalization(text: str, context: GlossaryContext, enabled: bool = True) -> str:
    if not enabled or not text:
        return text
    normalized = text
    for term in context.terms:
        if term.mode != "hard":
            continue
        replacements = term.replacements or generated_replacements(term)
        for replacement in replacements:
            pattern = replacement["from"]
            target = replacement.get("to") or term.canonical
            normalized, count = re.subn(pattern, target, normalized, flags=re.IGNORECASE)
            context.replacement_stats.add(term.canonical, count)
    return normalized


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
