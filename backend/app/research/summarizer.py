from __future__ import annotations

from app.research.models import KnowledgeSummary


class KnowledgeSummarizer:
    """Final shaping point for the knowledge object consumed by answer generation."""

    def summarize(self, summary: KnowledgeSummary) -> dict:
        payload = summary.to_dict()
        payload["brief"] = self._brief(summary)
        return payload

    @staticmethod
    def _brief(summary: KnowledgeSummary) -> str:
        pieces: list[str] = []
        if summary.attractions:
            names = "、".join(item["title"] for item in summary.attractions[:5] if item.get("title"))
            pieces.append(f"景点候选：{names}")
        if summary.weather:
            pieces.append(f"天气：{summary.weather.get('summary') or summary.weather.get('title')}")
        if summary.transport:
            pieces.append(f"交通：{summary.transport[0].get('content') or summary.transport[0].get('title')}")
        if summary.xiaohongshu_insights:
            pieces.append("小红书经验：" + "；".join(summary.xiaohongshu_insights[:3]))
        if summary.conflicts:
            pieces.append("信息风险：" + "；".join(summary.conflicts))
        return "\n".join(piece for piece in pieces if piece)

