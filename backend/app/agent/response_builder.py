from __future__ import annotations

from backend.app.domain.models import AgentContext, AgentResponse, ToolResult


class ResponseBuilder:
    def build(self, context: AgentContext, intent: dict, tool_results: list[ToolResult], plan: dict) -> AgentResponse:
        destination = intent.get("destination") or context.user_profile.destination or "待确认目的地"
        days = intent.get("days") or context.user_profile.days or 3
        budget = intent.get("budget") or context.user_profile.budget
        preferences = intent.get("preferences") or context.user_profile.preferences
        missing_information = intent.get("missing_information", [])

        weather = self._find(tool_results, "weather_lookup")
        route = self._find(tool_results, "route_planning")
        poi = self._find(tool_results, "poi_recommendation")

        answer = self._build_answer(
            destination=destination,
            days=days,
            budget=budget,
            preferences=preferences,
            weather=weather,
            route=route,
            poi=poi,
            missing_information=missing_information,
        )
        structured_plan = {
            "summary": f"为 {destination} 生成的 {days} 天旅行规划思路",
            "steps": plan.get("steps", []),
            "missing_information": missing_information,
            "cards": [
                {
                    "title": f"{destination} {days} 天行程建议",
                    "subtitle": f"预算 {budget if budget else '待补充'}；偏好 {', '.join(preferences) if preferences else '综合体验'}",
                    "details": [
                        "第 1 天建议优先安排城市地标与经典区域。",
                        "第 2 天结合用户偏好补充主题玩法与特色餐饮。",
                        "最后一天预留返程、购物或轻松体验时段。",
                    ],
                },
                {
                    "title": "工具协同结果",
                    "subtitle": "基于天气、路线与点位推荐的综合结论",
                    "details": [
                        weather.get("forecast", "天气信息暂未获取") if weather else "天气信息暂未获取",
                        route.get("route_summary", "路线信息暂未获取") if route else "路线信息暂未获取",
                    ],
                },
            ],
            "route_notes": route.get("segments", []) if route else [],
            "next_actions": self._next_actions(missing_information),
        }
        tool_payload = {item.name: item.payload if item.success else {"error": item.error} for item in tool_results}
        reflection = {
            "completeness": "medium" if missing_information else "high",
            "risks": [
                "真实天气、酒店价格和景区营业时间仍需接入外部 API 校验",
                "当前演示版本的工具结果为模拟数据，生产环境需替换为真实供应商实现",
            ],
            "recommendation": "建议下一步引入统一 ToolFactory、Redis 持久记忆以及真实地图/酒店 Provider。",
        }

        return AgentResponse(
            answer=answer,
            intent=intent,
            plan=structured_plan,
            tool_results=tool_payload,
            memory_context=context.memory,
            reflection_result=reflection,
            retrieved_docs=[],
        )

    @staticmethod
    def _find(tool_results: list[ToolResult], name: str) -> dict:
        for item in tool_results:
            if item.name == name and item.success:
                return item.payload
        return {}

    @staticmethod
    def _next_actions(missing_information: list[str]) -> list[str]:
        if not missing_information:
            return [
                "确认是否需要细化到每天上午/下午/晚上的时间段安排。",
                "确认是否需要补充酒店、交通或预算拆分。",
                "确认是否需要生成可修改版本的行程草案。",
            ]

        mapping = {
            "destination": "请先确认目的地城市。",
            "days": "请补充旅行天数，便于细化行程节奏。",
        }
        return [mapping.get(item, f"请补充 {item}。") for item in missing_information]

    @staticmethod
    def _build_answer(
        destination: str,
        days: int,
        budget: int | None,
        preferences: list[str],
        weather: dict,
        route: dict,
        poi: dict,
        missing_information: list[str],
    ) -> str:
        if missing_information:
            missing_text = "、".join(missing_information)
            return (
                f"我已经开始为你准备旅行规划，但目前还缺少 {missing_text}。"
                "你补充这些信息后，我可以继续把行程、路线和推荐内容细化到可执行层。"
            )

        preference_text = "、".join(preferences) if preferences else "综合体验"
        budget_text = f"预算约 {budget} 元" if budget else "预算暂未明确"
        weather_text = weather.get("forecast", "天气情况还需要进一步确认")
        route_text = route.get("route_summary", "建议按区域拆分行程，减少折返")
        poi_items = poi.get("recommendations", [])[:3]
        poi_text = "、".join(poi_items) if poi_items else "可进一步补充景点与美食清单"
        return (
            f"我已经为你整理了一版 {destination} {days} 天旅行规划框架。"
            f"整体会围绕 {preference_text} 展开，当前 {budget_text}。"
            f"天气方面，{weather_text}。路线方面，{route_text}。"
            f"推荐优先关注 {poi_text}。如果你愿意，我下一步可以继续细化到逐日行程。"
        )
