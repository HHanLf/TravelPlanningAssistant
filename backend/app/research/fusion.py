from __future__ import annotations

from collections import Counter
from typing import Any

from app.research.models import EvidenceItem, KnowledgeSummary


class InformationFusion:
    """Deduplicates, scores, and groups evidence into travel knowledge buckets."""

    def fuse(self, evidence: list[EvidenceItem], destination: str = "") -> KnowledgeSummary:
        deduped = self._dedupe(evidence)
        coverage = Counter(item.source_type for item in deduped)
        conflicts = self._detect_conflicts(deduped)
        confidence = self._confidence(deduped, conflicts)

        summary = KnowledgeSummary(
            destination=destination,
            attractions=[self._public_item(item) for item in self._top(deduped, "attraction", 10)],
            restaurants=[self._public_item(item) for item in self._top(deduped, "restaurant", 8)],
            hotels=[self._public_item(item) for item in self._top(deduped, "hotel", 6)],
            transport=[self._public_item(item) for item in self._top(deduped, "transport", 4)],
            weather=self._weather(deduped),
            xiaohongshu_insights=[item.content for item in self._top(deduped, "social_insight", 6)],
            web_findings=[self._public_item(item) for item in self._top(deduped, "web", 5)],
            conflicts=conflicts,
            source_coverage=dict(coverage),
            confidence_score=confidence,
            research_notes=self._notes(deduped, conflicts, confidence),
            evidence_count=len(deduped),
        )
        return summary

    def _dedupe(self, evidence: list[EvidenceItem]) -> list[EvidenceItem]:
        by_key: dict[str, EvidenceItem] = {}
        for item in evidence:
            key = self._key(item)
            current = by_key.get(key)
            if current is None or item.confidence > current.confidence:
                by_key[key] = item
        return list(by_key.values())

    @staticmethod
    def _key(item: EvidenceItem) -> str:
        if item.url:
            return f"url:{item.url}"
        title = item.title.lower().replace(" ", "")
        category = "social" if item.category.startswith("social") else item.category
        return f"{category}:{title}"

    @staticmethod
    def _top(evidence: list[EvidenceItem], category: str, limit: int) -> list[EvidenceItem]:
        items = [item for item in evidence if item.category == category]
        return sorted(items, key=lambda item: (-item.confidence, item.title))[:limit]

    @staticmethod
    def _public_item(item: EvidenceItem) -> dict[str, Any]:
        return {
            "title": item.title,
            "content": item.content,
            "source": item.source,
            "source_type": item.source_type,
            "url": item.url,
            "confidence": round(item.confidence, 2),
            "location": item.location,
            "raw": item.raw,
        }

    def _weather(self, evidence: list[EvidenceItem]) -> dict[str, Any]:
        items = self._top(evidence, "weather", 1)
        if not items:
            return {}
        item = items[0]
        return {
            "title": item.title,
            "summary": item.content,
            "confidence": round(item.confidence, 2),
            "raw": item.raw,
        }

    @staticmethod
    def _detect_conflicts(evidence: list[EvidenceItem]) -> list[str]:
        conflicts: list[str] = []
        title_sources: dict[str, set[str]] = {}
        for item in evidence:
            if not item.title:
                continue
            key = item.title.lower().replace(" ", "")
            title_sources.setdefault(key, set()).add(item.source_type)
        single_source_titles = [key for key, sources in title_sources.items() if len(sources) == 1]
        if len(single_source_titles) == len(title_sources) and len(title_sources) > 8:
            conflicts.append("Most findings come from single-source evidence; keep uncertain claims conservative.")
        return conflicts[:5]

    @staticmethod
    def _confidence(evidence: list[EvidenceItem], conflicts: list[str]) -> float:
        if not evidence:
            return 0.0
        base = sum(item.confidence for item in evidence) / len(evidence)
        source_bonus = min(len({item.source_type for item in evidence}) * 0.04, 0.16)
        conflict_penalty = min(len(conflicts) * 0.08, 0.2)
        return round(max(0.0, min(base + source_bonus - conflict_penalty, 1.0)), 2)

    @staticmethod
    def _notes(evidence: list[EvidenceItem], conflicts: list[str], confidence: float) -> list[str]:
        notes: list[str] = []
        if not evidence:
            return ["No usable research evidence was collected."]
        coverage = Counter(item.category for item in evidence)
        notes.append("Research coverage: " + ", ".join(f"{key}={value}" for key, value in sorted(coverage.items())))
        notes.append(f"Overall research confidence: {confidence:.2f}")
        notes.extend(conflicts)
        return notes

