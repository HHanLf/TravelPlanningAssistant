from __future__ import annotations

from typing import Any

from app.agent.state import ExecutionPlan, IntentResult, PlanningProblem
from app.research.models import ResearchTask


class ResearchPlanner:
    """Builds focused travel research tasks from the existing agent plan."""

    TOOL_TO_CATEGORY = {
        "weather_lookup": "weather",
        "place_search": "attraction",
        "restaurant_recommendation": "restaurant",
        "hotel_search": "hotel",
        "route_planning": "transport",
        "xiaohongshu_search": "social",
        "web_search": "web",
    }

    SOURCE_POLICY = {
        "weather": "realtime_api",
        "attraction": "map_poi",
        "restaurant": "map_poi",
        "hotel": "hotel_provider",
        "transport": "map_route",
        "social": "ugc_cross_check",
        "web": "public_web",
    }

    def create_tasks(
        self,
        *,
        message: str,
        intent: IntentResult | None,
        problem: PlanningProblem | None,
        plan: ExecutionPlan | None,
        available_tools: list[dict[str, Any]],
        reflection_notes: list[str] | None = None,
    ) -> list[ResearchTask]:
        if not intent or intent.type == "chat" or not problem:
            return []

        registered = {item.get("name") for item in available_tools}
        tasks: list[ResearchTask] = []
        for call in plan.tool_calls if plan else []:
            if call.name not in registered:
                continue
            category = self.TOOL_TO_CATEGORY.get(call.name, "general")
            tasks.append(
                ResearchTask(
                    id=f"{category}:{len(tasks) + 1}",
                    category=category,
                    tool_name=call.name,
                    arguments=dict(call.arguments),
                    query=self._query_text(message, problem, category, call.arguments),
                    priority=self._priority(intent.type, category),
                    required=self._is_required(intent.type, category),
                    expected_output=self._expected_output(category),
                    source_policy=self.SOURCE_POLICY.get(category, ""),
                    reason=call.reason,
                )
            )

        tasks.extend(self._supplemental_tasks(message, intent, problem, registered, len(tasks)))
        if reflection_notes:
            tasks = self._raise_priorities_for_reflection(tasks, reflection_notes)
        return self._dedupe(tasks)

    def _supplemental_tasks(
        self,
        message: str,
        intent: IntentResult,
        problem: PlanningProblem,
        registered: set[str],
        offset: int,
    ) -> list[ResearchTask]:
        if not problem.destination:
            return []

        destination = self._tool_destination(problem)
        tasks: list[ResearchTask] = []
        wanted: list[tuple[str, str, dict[str, Any], bool]] = []
        if intent.type == "trip_plan":
            wanted = [
                ("weather", "weather_lookup", {"destination": destination, "date_label": (problem.date_range or {}).get("label", "")}, True),
                ("attraction", "place_search", {"destination": destination, "keyword": self._place_keyword(problem)}, True),
                ("restaurant", "restaurant_recommendation", {"destination": destination, "keyword": self._restaurant_keyword(problem, message)}, False),
                ("hotel", "hotel_search", {"destination": destination, "budget_per_night": self._hotel_budget(problem)}, False),
                ("social", "xiaohongshu_search", {"keyword": self._xiaohongshu_keyword(problem, message), "limit": 8}, False),
                ("web", "web_search", {"query": f"{problem.destination} 旅行 攻略 交通 住宿 避坑"}, False),
            ]
            if problem.origin:
                wanted.append(
                    (
                        "transport",
                        "route_planning",
                        {"origin": problem.origin, "destination": destination, "mode": self._route_mode(problem)},
                        True,
                    )
                )
        elif intent.type == "place_search":
            wanted = [
                ("attraction", "place_search", {"destination": destination, "keyword": self._place_keyword(problem)}, True),
                ("social", "xiaohongshu_search", {"keyword": f"{problem.destination} 景点 避坑 攻略", "limit": 8}, False),
            ]
        elif intent.type == "restaurant_recommendation":
            wanted = [
                ("restaurant", "restaurant_recommendation", {"destination": destination, "keyword": self._restaurant_keyword(problem, message)}, True),
                ("social", "xiaohongshu_search", {"keyword": f"{problem.destination} 美食 餐厅 避坑", "limit": 8}, False),
            ]

        for category, tool_name, arguments, required in wanted:
            if tool_name not in registered:
                continue
            tasks.append(
                ResearchTask(
                    id=f"{category}:{offset + len(tasks) + 1}",
                    category=category,
                    tool_name=tool_name,
                    arguments={key: value for key, value in arguments.items() if value not in (None, "")},
                    query=self._query_text(message, problem, category, arguments),
                    priority=self._priority(intent.type, category),
                    required=required,
                    expected_output=self._expected_output(category),
                    source_policy=self.SOURCE_POLICY.get(category, ""),
                    reason=f"Supplemental {category} research for travel planning.",
                )
            )
        return tasks

    @staticmethod
    def _query_text(message: str, problem: PlanningProblem, category: str, arguments: dict[str, Any]) -> str:
        destination = problem.destination or arguments.get("destination") or ""
        keyword = arguments.get("keyword") or arguments.get("query") or category
        return f"{destination} {keyword}".strip() or message[:80]

    @staticmethod
    def _priority(intent_type: str, category: str) -> int:
        if intent_type == "trip_plan":
            weights = {
                "attraction": 100,
                "weather": 90,
                "transport": 85,
                "hotel": 75,
                "restaurant": 70,
                "social": 65,
                "web": 55,
            }
            return weights.get(category, 50)
        return 100 if category in {"weather", "transport", "hotel", "restaurant", "attraction"} else 60

    @staticmethod
    def _is_required(intent_type: str, category: str) -> bool:
        if intent_type == "trip_plan":
            return category in {"attraction", "weather"}
        mapping = {
            "weather": "weather",
            "transport_advice": "transport",
            "hotel_search": "hotel",
            "restaurant_recommendation": "restaurant",
            "place_search": "attraction",
        }
        return mapping.get(intent_type) == category

    @staticmethod
    def _expected_output(category: str) -> str:
        return {
            "weather": "Weather constraints and travel timing advice.",
            "attraction": "Relevant POIs with addresses, ranking hints, and semantic fit.",
            "transport": "Route distance, duration, and mode advice.",
            "social": "Recent UGC tips, pitfalls, queues, reservations, and area hints.",
            "hotel": "Hotel or lodging area candidates with budget hints.",
            "restaurant": "Food places and meal planning hints.",
            "web": "Public travel facts and supplemental findings.",
        }.get(category, "Useful travel evidence.")

    @staticmethod
    def _tool_destination(problem: PlanningProblem) -> str:
        return str((problem.constraints or {}).get("tool_destination") or problem.destination or "").strip()

    @staticmethod
    def _place_keyword(problem: PlanningProblem) -> str:
        constraints = problem.constraints or {}
        if constraints.get("avoid_paid_attractions"):
            return "免费景点"
        if constraints.get("family_friendly"):
            return "亲子景点"
        for keyword in ("博物馆", "夜景", "海边", "温泉", "亲子", "摄影", "自然", "历史"):
            if keyword in problem.preferences:
                return keyword
        return "景点"

    @staticmethod
    def _restaurant_keyword(problem: PlanningProblem, message: str) -> str:
        text = " ".join([message or "", " ".join(problem.preferences or [])])
        if any(token in text for token in ("咖啡", "下午茶")):
            return "咖啡"
        if any(token in text for token in ("早餐", "早饭", "早点")):
            return "早餐"
        if any(token in text for token in ("小吃", "夜市", "夜宵")):
            return "小吃"
        if any(token in text for token in ("火锅", "羊肉", "羊锅")):
            return "火锅"
        city = problem.destination or ""
        city_keywords = {
            "杭州": "杭帮菜",
            "广州": "早茶",
            "成都": "川菜",
            "上海": "本帮菜",
            "北京": "北京菜",
            "厦门": "沙茶面",
            "西安": "小吃",
            "重庆": "火锅",
        }
        return city_keywords.get(city, "当地特色餐厅")

    @staticmethod
    def _hotel_budget(problem: PlanningProblem) -> int | None:
        if not problem.budget:
            return None
        days = max(problem.days or 3, 1)
        group_size = max(problem.group_size or 2, 1)
        nights = max(days - 1, 1)
        return max(int(problem.budget * 0.35 / nights / max(group_size / 2, 1)), 180)

    @staticmethod
    def _route_mode(problem: PlanningProblem) -> str:
        transport_mode = (problem.constraints or {}).get("transport_mode")
        if transport_mode == "public":
            return "transit"
        if transport_mode == "driving":
            return "driving"
        return "transit" if problem.origin and problem.destination else "driving"

    @staticmethod
    def _xiaohongshu_keyword(problem: PlanningProblem, message: str) -> str:
        parts = [problem.destination or ""]
        if problem.days:
            parts.append(f"{problem.days}日游")
        parts.extend(problem.preferences[:3])
        parts.append("攻略 避坑")
        keyword = " ".join(part for part in parts if part).strip()
        return keyword or message[:40]

    @staticmethod
    def _raise_priorities_for_reflection(tasks: list[ResearchTask], notes: list[str]) -> list[ResearchTask]:
        text = " ".join(notes)
        for task in tasks:
            if task.category in text or task.tool_name in text:
                task.priority = min(task.priority + 20, 120)
        return tasks

    @staticmethod
    def _dedupe(tasks: list[ResearchTask]) -> list[ResearchTask]:
        seen: set[tuple[str, tuple[tuple[str, Any], ...]]] = set()
        unique: list[ResearchTask] = []
        for task in sorted(tasks, key=lambda item: (-item.priority, item.id)):
            key = (task.tool_name, tuple(sorted((name, repr(value)) for name, value in task.arguments.items())))
            if key in seen:
                continue
            seen.add(key)
            unique.append(task)
        return unique
