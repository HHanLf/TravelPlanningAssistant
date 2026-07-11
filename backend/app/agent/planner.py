from __future__ import annotations

from typing import Any

from app.agent.state import ExecutionPlan, IntentResult, PlanningProblem, ToolCall
from app.agent.travel_semantics import get_poi_theme_profile, infer_poi_theme


class Planner:
    def create_plan(
        self,
        message: str,
        intent: IntentResult,
        problem: PlanningProblem,
        memory_context: dict[str, Any],
        available_tools: list[dict[str, Any]],
        reflection_notes: list[str] | None = None,
    ) -> ExecutionPlan:
        del memory_context
        registered = {item["name"] for item in available_tools}
        tool_calls: list[ToolCall] = []

        destination = problem.destination
        tool_destination = self._tool_destination(problem)
        constraints = problem.constraints or {}
        if destination:
            for tool_name in intent.preferred_tools:
                call = self._build_tool_call(tool_name, message, problem)
                if call and call.name in registered:
                    tool_calls.append(call)

            if intent.type == "trip_plan" and problem.origin and "route_planning" in registered:
                tool_calls.append(
                    ToolCall(
                        name="route_planning",
                        arguments={
                            "origin": problem.origin,
                            "destination": tool_destination,
                            "mode": self._route_mode(problem),
                        },
                        reason="用户提供了出发地，补充起终点交通耗时参考。",
                    )
                )

        if intent.type == "chat":
            response_goal = "用中文自然回应用户，并在合适时引导补充旅行信息。"
        else:
            response_goal = self._response_goal(intent.type, problem)

        steps = [
            {"stage": "normalize", "title": "统一输入", "detail": "合并文本、语音转写和图片摘要。"},
            {"stage": "understand", "title": "识别意图", "detail": intent.reason},
            {
                "stage": "analyze",
                "title": "分析规划问题",
                "detail": (
                    f"目的地={destination or '待确认'}，缺失={', '.join(problem.missing_info) or '无'}，"
                    f"约束={self._constraint_summary(constraints)}。"
                ),
            },
            {
                "stage": "plan",
                "title": "生成执行计划",
                "detail": f"计划调用 {len(tool_calls)} 个工具，回答目标：{response_goal}",
            },
            {
                "stage": "execute",
                "title": "执行工具",
                "detail": "只执行已注册且参数通过校验的工具。",
            },
            {
                "stage": "respond",
                "title": "生成回答",
                "detail": "汇总问题、记忆、画像、规划、工具结果和缺失信息。",
            },
            {
                "stage": "reflect",
                "title": "反思质量",
                "detail": "检查完整度、工具使用、核心诉求覆盖和关键旅行要素。",
            },
        ]
        if reflection_notes:
            steps.append(
                {
                    "stage": "revise",
                    "title": "吸收反思意见",
                    "detail": "；".join(reflection_notes),
                }
            )

        return ExecutionPlan(
            response_goal=response_goal,
            missing_info=list(problem.missing_info),
            tool_calls=self._dedupe(tool_calls),
            assumptions=list(problem.assumptions),
            steps=steps,
        )

    def _build_tool_call(self, tool_name: str, message: str, problem: PlanningProblem) -> ToolCall | None:
        destination = problem.destination
        tool_destination = self._tool_destination(problem)
        if not destination and tool_name != "web_search":
            return None

        if tool_name == "weather_lookup":
            return ToolCall(
                name=tool_name,
                arguments={
                    "destination": tool_destination,
                    "date_label": (problem.date_range or {}).get("label", ""),
                },
                reason="天气会影响行程顺序、穿搭和室内外安排。",
            )
        if tool_name == "route_planning":
            if not problem.origin:
                return None
            return ToolCall(
                name=tool_name,
                arguments={"origin": problem.origin, "destination": tool_destination, "mode": self._route_mode(problem)},
                reason="需要估算出发地到目的地的交通耗时。",
            )
        if tool_name == "place_search":
            return ToolCall(
                name=tool_name,
                arguments={"destination": tool_destination, "keyword": self._place_keyword(problem)},
                reason="需要检索可落地的景点和区域候选。",
            )
        if tool_name == "restaurant_recommendation":
            return ToolCall(
                name=tool_name,
                arguments={"destination": tool_destination, "keyword": self._restaurant_keyword(problem, message)},
                reason="需要补充当地餐饮和小吃候选。",
            )
        if tool_name == "hotel_search":
            return ToolCall(
                name=tool_name,
                arguments={
                    "destination": tool_destination,
                    "budget_per_night": self._hotel_budget(problem),
                    "near_metro": bool(problem.constraints.get("hotel_near_metro")) if problem.constraints else False,
                },
                reason="需要根据预算补充住宿候选或区域建议。",
            )
        if tool_name == "xiaohongshu_search":
            return ToolCall(
                name=tool_name,
                arguments={"keyword": self._xiaohongshu_keyword(problem, message), "limit": 5},
                reason="需要补充近期旅行笔记中的避坑和体验线索。",
            )
        if tool_name == "web_search":
            return ToolCall(
                name=tool_name,
                arguments={"query": f"{destination or message} 旅行 攻略"},
                reason="需要补充公开信息。",
            )
        return None

    @staticmethod
    def _response_goal(intent_type: str, problem: PlanningProblem) -> str:
        destination = problem.destination or "目的地"
        mapping = {
            "trip_plan": f"给出 {destination} 的可执行旅行规划，覆盖行程节奏、交通、住宿、美食、预算和注意事项。",
            "transport_advice": "比较交通方式并给出适合当前条件的建议。",
            "weather": "说明天气对行程和穿搭的影响。",
            "hotel_search": "给出住宿区域和酒店选择建议。",
            "restaurant_recommendation": "给出餐饮推荐和安排时段建议。",
            "place_search": "给出景点/区域候选及游玩顺序建议。",
        }
        return mapping.get(intent_type, "围绕用户旅行问题给出中文建议。")

    @staticmethod
    def _place_keyword(problem: PlanningProblem) -> str:
        constraints = problem.constraints or {}
        theme = infer_poi_theme(preferences=problem.preferences, constraints=constraints)
        profile = get_poi_theme_profile(theme)
        if profile:
            return profile.query_keywords[0]
        if constraints.get("avoid_paid_attractions"):
            return "免费景点"
        if constraints.get("family_friendly"):
            return "亲子景点"
        for keyword in ("博物馆", "夜景", "海边", "温泉", "亲子", "摄影"):
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
        return city_keywords.get(problem.destination or "", "当地特色餐厅")

    @staticmethod
    def _tool_destination(problem: PlanningProblem) -> str | None:
        constraints = problem.constraints or {}
        return str(constraints.get("tool_destination") or problem.destination or "").strip() or None

    @staticmethod
    def _route_mode(problem: PlanningProblem) -> str:
        transport_mode = (problem.constraints or {}).get("transport_mode")
        if transport_mode == "public":
            return "transit"
        if transport_mode == "driving":
            return "driving"
        if problem.origin and problem.destination:
            return "transit"
        return "driving"

    @staticmethod
    def _constraint_summary(constraints: dict[str, Any]) -> str:
        if not constraints:
            return "无"
        labels = {
            "travel_theme": "旅行主题",
            "transport_mode": "交通方式",
            "family_friendly": "亲子友好",
            "low_walking": "少步行",
            "pace": "节奏",
            "hotel_near_metro": "住宿近地铁",
            "avoid_paid_attractions": "避开收费景点",
            "dietary_preference": "饮食偏好",
        }
        hidden = {"requested_region", "tool_destination", "region_resolution_note"}
        return "、".join(f"{labels.get(key, key)}={value}" for key, value in constraints.items() if key not in hidden)

    @staticmethod
    def _hotel_budget(problem: PlanningProblem) -> int | None:
        if not problem.budget:
            return None
        days = max(problem.days or 3, 1)
        group_size = max(problem.group_size or 2, 1)
        nights = max(days - 1, 1)
        return max(int(problem.budget * 0.35 / nights / max(group_size / 2, 1)), 180)

    @staticmethod
    def _xiaohongshu_keyword(problem: PlanningProblem, message: str) -> str:
        constraints = problem.constraints or {}
        parts = [problem.destination or ""]
        tool_destination = str(constraints.get("tool_destination") or "")
        if tool_destination and tool_destination not in parts:
            parts.append(tool_destination)
        if any(keyword in message for keyword in ("餐厅", "饭店", "美食", "小吃", "咖啡")):
            parts.extend(["餐厅", "美食"])
        if problem.days:
            parts.append(f"{problem.days}日游")
        parts.extend(problem.preferences[:3])
        parts.append("攻略")
        keyword = " ".join(part for part in parts if part).strip()
        return keyword or message[:40]

    @staticmethod
    def _dedupe(tool_calls: list[ToolCall]) -> list[ToolCall]:
        seen: set[tuple[str, tuple[tuple[str, Any], ...]]] = set()
        unique: list[ToolCall] = []
        for call in tool_calls:
            key = (call.name, tuple(sorted(call.arguments.items())))
            if key in seen:
                continue
            seen.add(key)
            unique.append(call)
        return unique
