from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class IntentDefinition:
    name: str
    trigger_keywords: tuple[str, ...] = ()
    requires_destination: bool = False
    preferred_tools: tuple[str, ...] = ()


class IntentAnalyzer:
    DESTINATIONS = (
        "北京",
        "上海",
        "广州",
        "深圳",
        "成都",
        "杭州",
        "重庆",
        "西安",
        "南京",
        "苏州",
        "厦门",
        "长沙",
        "武汉",
        "三亚",
    )
    PREFERENCES = (
        "自然",
        "风景",
        "美食",
        "亲子",
        "历史",
        "购物",
        "摄影",
        "夜景",
        "博物馆",
        "徒步",
        "海边",
        "温泉",
    )

    def __init__(self) -> None:
        self._definitions = (
            IntentDefinition(
                name="travel_planning",
                trigger_keywords=("攻略", "规划", "旅行", "旅游", "行程"),
                requires_destination=True,
                preferred_tools=("weather_lookup", "route_planning", "poi_recommendation"),
            ),
            IntentDefinition(
                name="route_planning",
                trigger_keywords=("路线", "怎么走", "顺路", "地铁", "打车"),
                requires_destination=True,
                preferred_tools=("route_planning",),
            ),
            IntentDefinition(
                name="hotel_recommendation",
                trigger_keywords=("酒店", "住宿", "民宿"),
                requires_destination=True,
                preferred_tools=("hotel_search",),
            ),
            IntentDefinition(
                name="weather_analysis",
                trigger_keywords=("天气", "下雨", "气温", "穿什么"),
                requires_destination=True,
                preferred_tools=("weather_lookup",),
            ),
            IntentDefinition(
                name="food_recommendation",
                trigger_keywords=("吃", "餐厅", "美食", "小吃"),
                requires_destination=True,
                preferred_tools=("poi_recommendation",),
            ),
            IntentDefinition(
                name="itinerary_adjustment",
                trigger_keywords=("修改", "调整", "优化", "改一下"),
                requires_destination=False,
                preferred_tools=("weather_lookup", "route_planning", "poi_recommendation"),
            ),
        )

    def analyze(self, message: str) -> dict[str, Any]:
        destination = next((city for city in self.DESTINATIONS if city in message), None)
        days = self._extract_days(message)
        budget = self._extract_budget(message)
        companions = self._extract_companions(message)
        preferences = [keyword for keyword in self.PREFERENCES if keyword in message]
        intent = self._resolve_intent(message)

        return {
            "type": intent.name,
            "destination": destination,
            "days": days,
            "budget": budget,
            "companions": companions,
            "preferences": preferences,
            "preferred_tools": list(intent.preferred_tools),
            "requires_destination": intent.requires_destination,
            "requires_tools": bool(intent.preferred_tools),
            "missing_information": self._missing_information(intent, destination, days),
        }

    def _resolve_intent(self, message: str) -> IntentDefinition:
        for definition in self._definitions:
            if any(keyword in message for keyword in definition.trigger_keywords):
                return definition
        return self._definitions[0]

    @staticmethod
    def _missing_information(
        intent: IntentDefinition,
        destination: str | None,
        days: int | None,
    ) -> list[str]:
        missing: list[str] = []
        if intent.requires_destination and not destination:
            missing.append("destination")
        if intent.name == "travel_planning" and not days:
            missing.append("days")
        return missing

    @staticmethod
    def _extract_days(message: str) -> int | None:
        patterns = (r"(\d+)天", r"(\d+)日")
        for pattern in patterns:
            match = re.search(pattern, message)
            if match:
                return int(match.group(1))
        return None

    @staticmethod
    def _extract_budget(message: str) -> int | None:
        match = re.search(r"预算\s*(\d+)", message)
        return int(match.group(1)) if match else None

    @staticmethod
    def _extract_companions(message: str) -> int | None:
        match = re.search(r"(\d+)人", message)
        return int(match.group(1)) if match else None
