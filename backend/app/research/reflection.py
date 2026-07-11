from __future__ import annotations

from typing import Any


class ResearchReflection:
    """Checks information quality before final answer generation."""

    REQUIRED_FOR_TRIP = {"attraction", "weather"}

    def evaluate(self, knowledge_summary: dict[str, Any], intent_type: str = "") -> dict[str, Any]:
        issues: list[str] = []
        suggestions: list[str] = []
        coverage = knowledge_summary.get("source_coverage") or {}
        confidence = float(knowledge_summary.get("confidence_score") or 0)

        if intent_type == "trip_plan":
            missing = [
                category
                for category in self.REQUIRED_FOR_TRIP
                if not knowledge_summary.get(self._payload_key(category))
            ]
            if missing:
                issues.append("Missing required research categories: " + ", ".join(missing))
                suggestions.append("Run focused research for missing categories before treating the plan as reliable.")

        if confidence and confidence < 0.55:
            issues.append(f"Research confidence is low: {confidence:.2f}")
            suggestions.append("Use conservative language and avoid presenting uncertain details as facts.")

        if not coverage:
            issues.append("No source coverage is available.")
            suggestions.append("Collect at least one reliable source before generating a detailed travel plan.")

        conflicts = knowledge_summary.get("conflicts") or []
        if conflicts:
            issues.extend(str(item) for item in conflicts[:3])
            suggestions.append("Mention uncertainty where sources are thin or conflicting.")

        return {
            "passed": not issues,
            "score": self._score(confidence, issues, coverage),
            "issues": issues,
            "suggestions": suggestions,
        }

    @staticmethod
    def _payload_key(category: str) -> str:
        return {
            "attraction": "attractions",
            "weather": "weather",
            "transport": "transport",
            "hotel": "hotels",
            "restaurant": "restaurants",
        }.get(category, category)

    @staticmethod
    def _score(confidence: float, issues: list[str], coverage: dict[str, Any]) -> int:
        score = int((confidence or 0) * 100)
        score += min(len(coverage) * 4, 16)
        score -= len(issues) * 12
        return max(0, min(score, 100))

