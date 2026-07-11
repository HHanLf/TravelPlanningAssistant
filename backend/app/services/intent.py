from __future__ import annotations

from dataclasses import dataclass

from app.agent.state import IntentResult, PlanningProblem


@dataclass(frozen=True, slots=True)
class IntentDefinition:
    type: str
    keywords: tuple[str, ...]
    preferred_tools: tuple[str, ...] = ()
    reason: str = ""


class IntentAnalyzer:
    """Rule-first intent classifier with stable labels for the frontend."""

    DEFINITIONS = (
        IntentDefinition(
            type="trip_plan",
            keywords=("攻略", "规划", "旅行", "旅游", "行程", "几日游", "自由行", "安排", "玩", "游玩"),
            preferred_tools=(
                "weather_lookup",
                "place_search",
                "restaurant_recommendation",
                "hotel_search",
                "xiaohongshu_search",
            ),
            reason="用户需要综合旅行规划。",
        ),
        IntentDefinition(
            type="weather",
            keywords=("天气", "下雨", "气温", "穿什么", "冷不冷", "热不热"),
            preferred_tools=("weather_lookup",),
            reason="用户需要天气和穿搭/行程影响判断。",
        ),
        IntentDefinition(
            type="hotel_search",
            keywords=("酒店", "住宿", "住哪", "民宿", "宾馆"),
            preferred_tools=("hotel_search",),
            reason="用户需要住宿候选或住宿区域建议。",
        ),
        IntentDefinition(
            type="restaurant_recommendation",
            keywords=("美食", "餐厅", "吃什么", "小吃", "夜宵", "早茶", "咖啡"),
            preferred_tools=("restaurant_recommendation", "xiaohongshu_search"),
            reason="用户需要餐饮推荐和本地经验。",
        ),
        IntentDefinition(
            type="place_search",
            keywords=("景点", "去哪玩", "打卡", "拍照", "博物馆", "公园", "夜景"),
            preferred_tools=("place_search", "xiaohongshu_search"),
            reason="用户需要景点或 POI 推荐。",
        ),
        IntentDefinition(
            type="transport_advice",
            keywords=("怎么去", "路线", "交通", "高铁", "飞机", "地铁", "打车", "自驾", "出发"),
            preferred_tools=("route_planning",),
            reason="用户在询问出行路线或交通方式。",
        ),
    )

    def analyze(self, message: str, problem: PlanningProblem | None = None) -> IntentResult:
        text = message or ""
        lowered = text.lower()

        if self._is_comprehensive_trip_request(text, problem):
            definition = self._definition("trip_plan")
            return IntentResult(
                type=definition.type,
                confidence=0.9,
                requires_tools=True,
                preferred_tools=list(definition.preferred_tools),
                reason="用户提供了目的地，并同时包含天数、预算或偏好，属于综合旅行规划需求。",
            )

        if self._is_explicit_transport_request(text):
            definition = self._definition("transport_advice")
            return IntentResult(
                type=definition.type,
                confidence=0.88,
                requires_tools=True,
                preferred_tools=list(definition.preferred_tools),
                reason=definition.reason,
            )

        best = None
        for definition in self.DEFINITIONS:
            if any(keyword in lowered or keyword in text for keyword in definition.keywords):
                best = definition
                break

        if best is None:
            if problem and (problem.destination or problem.preferences or problem.days):
                best = self._definition("trip_plan")
            else:
                return IntentResult(
                    type="chat",
                    confidence=0.55,
                    requires_tools=False,
                    preferred_tools=[],
                    reason="未识别到明确旅行工具诉求，按普通对话处理。",
                )

        confidence = 0.85 if any(keyword in text for keyword in best.keywords) else 0.7
        return IntentResult(
            type=best.type,
            confidence=confidence,
            requires_tools=bool(best.preferred_tools),
            preferred_tools=list(best.preferred_tools),
            reason=best.reason,
        )

    def _definition(self, intent_type: str) -> IntentDefinition:
        for definition in self.DEFINITIONS:
            if definition.type == intent_type:
                return definition
        raise ValueError(f"Unknown intent type: {intent_type}")

    @staticmethod
    def _is_comprehensive_trip_request(text: str, problem: PlanningProblem | None) -> bool:
        if not problem or not problem.destination:
            return False
        has_trip_context = any(keyword in text for keyword in ("玩", "游", "旅游", "旅行", "行程", "攻略", "规划", "自由行"))
        has_planning_signal = bool(problem.days or problem.budget or problem.preferences or problem.group_size)
        explicit_transport_only = IntentAnalyzer._is_explicit_transport_request(text) and not has_planning_signal
        return has_trip_context and has_planning_signal and not explicit_transport_only

    @staticmethod
    def _is_explicit_transport_request(text: str) -> bool:
        transport_phrases = (
            "怎么去",
            "如何去",
            "路线",
            "交通",
            "高铁",
            "飞机",
            "地铁",
            "打车",
            "自驾",
            "开车",
            "坐车",
            "火车",
            "动车",
            "航班",
            "机票",
        )
        if not any(keyword in text for keyword in transport_phrases):
            return False
        planning_phrases = ("玩", "旅游", "旅行", "行程", "攻略", "规划", "预算", "偏好")
        return not any(keyword in text for keyword in planning_phrases)
