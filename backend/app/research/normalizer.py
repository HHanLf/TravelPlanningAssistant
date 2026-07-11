from __future__ import annotations

import hashlib
from typing import Any

from app.agent.state import ToolResult
from app.research.models import EvidenceItem, ResearchResult


class EvidenceNormalizer:
    """Converts heterogeneous tool payloads into comparable evidence items."""

    def normalize_many(self, results: list[ResearchResult]) -> list[EvidenceItem]:
        items: list[EvidenceItem] = []
        for result in results:
            tool_result = result.tool_result
            if not isinstance(tool_result, ToolResult) or not tool_result.success:
                continue
            items.extend(self.normalize(result))
        return items

    def normalize(self, result: ResearchResult) -> list[EvidenceItem]:
        task = result.task
        tool_result = result.tool_result
        payload = tool_result.payload or {}
        if task.category in {"attraction", "restaurant"}:
            return self._from_places(task.id, task.category, tool_result.name, payload)
        if task.category == "hotel":
            return self._from_hotels(task.id, tool_result.name, payload)
        if task.category == "social":
            return self._from_xiaohongshu(task.id, tool_result.name, payload)
        if task.category == "weather":
            return [self._single(task.id, "weather", tool_result.name, self._weather_title(payload), tool_result.summary, payload, 0.88)]
        if task.category == "transport":
            return [self._single(task.id, "transport", tool_result.name, self._route_title(payload), tool_result.summary, payload, 0.84)]
        if task.category == "web":
            return self._from_web(task.id, tool_result.name, payload)
        return [self._single(task.id, task.category, tool_result.name, tool_result.summary[:80], tool_result.summary, payload, 0.5)]

    def _from_places(self, task_id: str, category: str, source: str, payload: dict[str, Any]) -> list[EvidenceItem]:
        places = payload.get("places") if isinstance(payload, dict) else []
        items: list[EvidenceItem] = []
        for place in places if isinstance(places, list) else []:
            if not isinstance(place, dict):
                continue
            title = str(place.get("name") or "").strip()
            if not title:
                continue
            content = " ".join(
                str(value)
                for value in (
                    place.get("address"),
                    place.get("category"),
                    place.get("semantic_score"),
                    place.get("matched_keyword"),
                )
                if value not in (None, "")
            )
            items.append(
                self._single(
                    task_id,
                    category,
                    source,
                    title,
                    content,
                    place,
                    self._place_confidence(place, category),
                    location=self._location(place),
                )
            )
        return items

    def _from_hotels(self, task_id: str, source: str, payload: dict[str, Any]) -> list[EvidenceItem]:
        hotels = payload.get("hotels") if isinstance(payload, dict) else []
        items: list[EvidenceItem] = []
        for hotel in hotels if isinstance(hotels, list) else []:
            if not isinstance(hotel, dict):
                continue
            title = str(hotel.get("name") or "").strip()
            if not title:
                continue
            content = " ".join(
                str(value)
                for value in (
                    hotel.get("area"),
                    hotel.get("location"),
                    hotel.get("price"),
                    hotel.get("rating"),
                    hotel.get("distance_hint"),
                )
                if value not in (None, "")
            )
            items.append(self._single(task_id, "hotel", source, title, content, hotel, self._hotel_confidence(hotel)))
        return items

    def _from_xiaohongshu(self, task_id: str, source: str, payload: dict[str, Any]) -> list[EvidenceItem]:
        notes = payload.get("notes") if isinstance(payload, dict) else []
        items: list[EvidenceItem] = []
        for note in notes if isinstance(notes, list) else []:
            if not isinstance(note, dict):
                continue
            title = str(note.get("title") or "").strip()
            content = str(note.get("summary") or "").strip()
            if not title and not content:
                continue
            items.append(
                self._single(
                    task_id,
                    "social",
                    source,
                    title or content[:40],
                    content,
                    note,
                    self._social_confidence(note),
                    url=note.get("url"),
                    timestamp=note.get("publish_time"),
                )
            )
        for insight in payload.get("insights") or []:
            if insight:
                items.append(self._single(task_id, "social_insight", source, str(insight)[:60], str(insight), {"insight": insight}, 0.7))
        return items

    def _from_web(self, task_id: str, source: str, payload: dict[str, Any]) -> list[EvidenceItem]:
        results = payload.get("results") if isinstance(payload, dict) else []
        items: list[EvidenceItem] = []
        for item in results if isinstance(results, list) else []:
            if not isinstance(item, dict):
                continue
            title = str(item.get("title") or "").strip()
            content = str(item.get("snippet") or "").strip()
            if not title and not content:
                continue
            items.append(self._single(task_id, "web", source, title or content[:40], content, item, 0.55, url=item.get("url")))
        return items

    def _single(
        self,
        task_id: str,
        category: str,
        source: str,
        title: str,
        content: str,
        raw: dict[str, Any],
        confidence: float,
        *,
        url: str | None = None,
        location: dict[str, Any] | None = None,
        timestamp: str | None = None,
    ) -> EvidenceItem:
        evidence_id = self._id(category, source, title, content, url)
        return EvidenceItem(
            id=evidence_id,
            category=category,
            source=source,
            source_type=self._source_type(source, category),
            title=title,
            content=content,
            url=url,
            location=location,
            timestamp=timestamp,
            confidence=max(0.0, min(confidence, 1.0)),
            task_id=task_id,
            raw=raw,
        )

    @staticmethod
    def _id(category: str, source: str, title: str, content: str, url: str | None) -> str:
        text = "|".join([category, source, title.lower(), content[:120].lower(), url or ""])
        return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]

    @staticmethod
    def _source_type(source: str, category: str) -> str:
        if "xiaohongshu" in source or category.startswith("social"):
            return "social"
        if source in {"place_search", "restaurant_recommendation", "route_planning"}:
            return "map"
        if source == "weather_lookup":
            return "weather"
        if source == "hotel_search":
            return "hotel"
        return "web"

    @staticmethod
    def _location(item: dict[str, Any]) -> dict[str, Any] | None:
        lat = item.get("lat") or item.get("latitude")
        lng = item.get("lng") or item.get("longitude")
        if lat is None and lng is None:
            return None
        return {"lat": lat, "lng": lng}

    @staticmethod
    def _place_confidence(place: dict[str, Any], category: str) -> float:
        score = 0.72 if category == "attraction" else 0.68
        if place.get("semantic_score"):
            score += min(float(place.get("semantic_score") or 0) / 100, 0.16)
        if place.get("address"):
            score += 0.05
        return min(score, 0.92)

    @staticmethod
    def _hotel_confidence(hotel: dict[str, Any]) -> float:
        score = 0.62
        if hotel.get("rating"):
            score += 0.12
        if hotel.get("price"):
            score += 0.08
        if hotel.get("distance_hint"):
            score += 0.06
        return min(score, 0.86)

    @staticmethod
    def _social_confidence(note: dict[str, Any]) -> float:
        score = 0.48
        if note.get("is_relevant"):
            score += 0.18
        if not note.get("is_marketing"):
            score += 0.14
        if int(note.get("engagement_score") or 0) > 0:
            score += 0.08
        return min(score, 0.82)

    @staticmethod
    def _weather_title(payload: dict[str, Any]) -> str:
        city = payload.get("resolved_city") or payload.get("city") or "weather"
        label = payload.get("date_label") or ""
        return f"{city} {label}".strip()

    @staticmethod
    def _route_title(payload: dict[str, Any]) -> str:
        origin = payload.get("origin") or ""
        destination = payload.get("destination") or ""
        return f"{origin} to {destination}".strip(" to") or "route"

