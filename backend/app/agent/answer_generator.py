from __future__ import annotations

import json
from typing import Any

from app.agent.state import AgentState, ToolResult
from app.agent.travel_semantics import is_polluted_poi
from app.services.llm_service import DashScopeLLMService


class AnswerGenerator:
    CRITICAL_TOOL_NAMES = {
        "weather_lookup",
        "place_search",
        "route_planning",
        "hotel_search",
        "restaurant_recommendation",
    }
    OPTIONAL_TOOL_NAMES = {"xiaohongshu_search", "web_search"}

    def __init__(self, llm: DashScopeLLMService | None = None) -> None:
        self._llm = llm or DashScopeLLMService()

    def generate(self, state: AgentState, force_fallback: bool = False) -> str:
        blocking_errors = self._blocking_tool_errors(state)
        intent_type = state.intent.type if state.intent else ""
        if blocking_errors and intent_type == "trip_plan":
            return self._fallback_answer(state)
        if blocking_errors:
            return self._tool_error_answer(state, blocking_errors)
        if force_fallback:
            return self._fallback_answer(state)
        if self._llm.available():
            answer = self._generate_with_llm(state)
            if answer:
                return answer
        return self._fallback_answer(state)

    def _blocking_tool_errors(self, state: AgentState) -> list[ToolResult]:
        intent_type = state.intent.type if state.intent else ""
        if intent_type == "chat":
            return []
        critical_names = set(self.CRITICAL_TOOL_NAMES)
        if intent_type in {"weather", "transport_advice", "hotel_search", "restaurant_recommendation", "place_search"}:
            intent_tool = {
                "weather": "weather_lookup",
                "transport_advice": "route_planning",
                "hotel_search": "hotel_search",
                "restaurant_recommendation": "restaurant_recommendation",
                "place_search": "place_search",
            }.get(intent_type)
            critical_names = {intent_tool} if intent_tool else critical_names
        blocking: list[ToolResult] = []
        for result in state.tool_results:
            if result.name in self.OPTIONAL_TOOL_NAMES:
                continue
            if result.name in critical_names and not result.success and result.error:
                blocking.append(result)
        return blocking

    def _tool_error_answer(self, state: AgentState, errors: list[ToolResult]) -> str:
        problem = state.problem
        destination = problem.destination if problem and problem.destination else "本次目的地"
        intent_type = state.intent.type if state.intent else ""
        labels = "、".join(self._tool_label(result.name) for result in errors[:3])
        title = "实时信息暂时不可用" if intent_type != "trip_plan" else "先给你一版保守行程"
        lines = [
            title,
            "",
            f"我暂时没有拿到 {destination} 的{labels or '实时'}结果，所以不会把具体班次、实时价格或开放状态当成确定信息。",
            "",
            "你现在可以这样处理：",
            "- 如果只是先做规划，我可以按常规路线、预算和偏好给一版保守建议。",
            "- 出发前再核对天气、交通班次、酒店价格和景区开放时间。",
        ]
        if intent_type in {"weather", "transport_advice", "hotel_search", "restaurant_recommendation", "place_search"}:
            lines.append(f"- 你也可以补充日期、出发地或预算，我会先按 {destination} 的常规经验继续收敛方案。")
        return "\n".join(lines)

    @staticmethod
    def _tool_label(name: str) -> str:
        return {
            "weather_lookup": "天气查询",
            "place_search": "地点搜索",
            "route_planning": "路线规划",
            "hotel_search": "酒店搜索",
            "restaurant_recommendation": "餐厅推荐",
            "xiaohongshu_search": "小红书搜索",
            "web_search": "网页搜索",
        }.get(name, name)

    @staticmethod
    def _format_arguments(arguments: dict[str, Any]) -> str:
        if not arguments:
            return ""
        visible = []
        for key, value in arguments.items():
            if value in (None, ""):
                continue
            visible.append(f"{key}={value}")
        return "，".join(visible[:6])

    def _generate_with_llm(self, state: AgentState) -> str:
        prompt_payload = {
            "user_question": state.effective_message,
            "memory_context": state.memory_context,
            "profile": state.profile,
            "intent": state.intent.to_dict() if state.intent else {},
            "problem": state.problem.to_dict() if state.problem else {},
            "plan": state.plan.to_dict() if state.plan else {},
            "knowledge_summary": state.knowledge_summary,
            "tool_results": self._safe_tool_results(state.tool_results),
            "recommended_itinerary": self._recommended_itinerary_payload(state),
            "retrieved_docs": state.retrieved_docs,
            "reflection_notes": state.reflection_notes,
        }
        messages = [
            {
                "role": "system",
                "content": (
                    "你是一个中文智能旅游规划 Agent 的最终回答生成器，不是普通聊天机器人。"
                    "你只能基于给定 Agent 状态、工具结果、记忆和明确假设输出方案，不要编造实时事实。"
                    "不要暴露内部工具名、success/failed、原始错误码、JSON、字段名或调试信息。"
                    "如果某个外部信息源失败，只能自然说明“部分实时笔记暂时未取到”，并用已取得的信息保守规划。"
                    "回答风格参考 ChatGPT：先给结论，再分段给可执行方案；语气自然、具体、干净。"
                    "必须覆盖：规划假设、天气/出行提醒、行程总览、每日安排、住宿建议、美食建议、预算拆分、注意事项。"
                    "必须遵守 problem.constraints 中的预算、交通、节奏、亲子、步行、住宿和饮食约束。"
                    "行程应按 Day 组织，包含上午/下午/晚上，不要只罗列地点；优先同片区串联，并解释为什么这样安排。"
                    "如果 provided recommended_itinerary 非空，必须优先按照其中的路线骨架生成，不要被地图候选的原始排序牵着走。"
                    "景点必须使用工具结果中已筛选的 places；不要把 filtered_out、调试信息或仅名称命中偏好词的机构当作旅行景点。"
                    "如果 reflection_notes 非空，说明上一版没有通过质量门禁；你必须按这些问题重写整份答案，而不是只补几句话。"
                    "如果人数、日期等信息缺失，先说明默认假设，再给可用初版，不要只追问。"
                    "输出纯文本，使用短标题和项目符号，适合直接显示在聊天气泡中。"
                ),
            },
            {
                "role": "user",
                "content": "请根据以下 Agent 状态生成最终中文旅行方案：\n"
                + json.dumps(prompt_payload, ensure_ascii=False, default=str),
            },
        ]
        return self._llm.chat(messages).strip()

    def _recommended_itinerary_payload(self, state: AgentState) -> list[dict[str, Any]]:
        problem = state.problem
        if problem is None:
            return []
        return self._itinerary_days(
            destination=problem.destination or "目的地",
            days=max(problem.days or 3, 1),
            preferences=problem.preferences or ["综合体验"],
            places=self._places(state.tool_results, "place_search"),
            restaurants=self._places(state.tool_results, "restaurant_recommendation"),
        )

    @staticmethod
    def _safe_tool_results(tool_results: list[ToolResult]) -> list[dict[str, Any]]:
        safe_results: list[dict[str, Any]] = []
        for item in tool_results:
            result = item.to_dict()
            if result.get("error"):
                result["error"] = "实时信息暂时不可用"
            payload = result.get("payload")
            if isinstance(payload, dict):
                sanitized_payload = dict(payload)
                sanitized_payload.pop("filtered_out", None)
                semantic_theme = str(sanitized_payload.get("semantic_theme") or "")
                places = sanitized_payload.get("places")
                if semantic_theme and isinstance(places, list):
                    sanitized_payload["places"] = [
                        place
                        for place in places
                        if isinstance(place, dict) and not is_polluted_poi(place, semantic_theme)
                    ]
                result["payload"] = sanitized_payload
            safe_results.append(result)
        return safe_results

    def _fallback_answer(self, state: AgentState) -> str:
        problem = state.problem
        intent_type = state.intent.type if state.intent else "chat"
        if not problem or intent_type == "chat":
            return "我在。你可以直接告诉我目的地、天数、预算、人数和偏好，我会按 Agent 流程先整理需求，再调用工具生成一版可执行行程。"

        if intent_type == "weather":
            return self._weather_answer(state)
        if intent_type == "transport_advice":
            return self._transport_answer(state)
        if intent_type == "hotel_search":
            return self._hotel_answer(state)
        if intent_type in {"restaurant_recommendation", "place_search"}:
            return self._focused_recommendation_answer(state)
        return self._trip_plan_answer(state)

    def _trip_plan_answer(self, state: AgentState) -> str:
        problem = state.problem
        assert problem is not None

        destination = problem.destination or "目的地"
        days = max(problem.days or 3, 1)
        nights = max(days - 1, 0)
        group_size = problem.group_size or 2
        preferences = problem.preferences or ["综合体验"]
        preference_text = "、".join(preferences[:4])
        constraints = self._constraint_text(problem.constraints)
        weather = self._weather_brief(state.tool_results)
        places = self._places(state.tool_results, "place_search")
        restaurants = self._places(state.tool_results, "restaurant_recommendation")
        hotels = self._hotels(state.tool_results)
        route = self._route_brief(state.tool_results)
        xhs = self._xiaohongshu_brief(state.tool_results)
        tool_limit_notice = self._tool_limit_notice(state.tool_results)
        budget_lines = self._budget_lines(problem.budget, days, group_size)
        missing = self._missing_info_text(problem.missing_info)
        assumptions = self._assumption_lines(problem, group_size)
        itinerary_days = self._itinerary_days(destination, days, preferences, places, restaurants)

        lines = [
            f"{destination} {days} 天 {nights} 晚{self._trip_theme(preferences)}之旅（{self._budget_title(problem.budget)}）",
            "",
            f"可以。我会按「{group_size} 人、偏好 {preference_text}、少折返」来安排。整体节奏偏休闲：每天抓一个主主题，上午放核心体验，下午接近距离补充点，晚上留给夜景或美食。",
        ]
        if assumptions:
            lines.extend(["", "规划假设", *assumptions])
        if constraints:
            lines.extend(["", "本次偏好与约束", f"- {constraints}"])

        lines.extend(["", "先说结论", f"- 这趟更适合做「{self._trip_theme(preferences)}」路线，不建议只挤热门商圈；把山林、湿地、湖泊和城市夜景分开安排，体验会更舒服。"])
        if tool_limit_notice:
            lines.append(f"- {tool_limit_notice}")
        if weather:
            lines.append(f"- 天气提醒：{weather}")
        if route:
            lines.append(f"- 交通提醒：{route}")
        if xhs:
            lines.append(f"- 实时经验参考：{xhs}")

        lines.extend(["", "行程总览"])
        lines.extend(self._overview_table(itinerary_days))

        lines.extend(["", "每日安排"])
        for item in itinerary_days:
            lines.extend(self._formatted_day_plan(item, problem.constraints))

        if hotels:
            lines.extend(["", "住宿建议"])
            lines.append("- 优先住在交通方便、能衔接主要景点的区域；如果你要少换乘，酒店靠近地铁会更省力。")
            for hotel in hotels[:4]:
                price = f"约 {hotel['price']} 元/晚" if hotel.get("price") else "价格待确认"
                area = hotel.get("area") or hotel.get("location") or "核心活动区"
                hint = hotel.get("distance_hint") or "适合作为行程落点"
                lines.append(f"- {hotel.get('name')}：{area}，{price}，{hint}。")
        else:
            lines.extend(["", "住宿建议"])
            lines.extend(self._lodging_suggestions(destination, problem.budget, nights))

        if restaurants:
            lines.extend(["", "美食安排"])
            for item in restaurants[:5]:
                address = f"（{item.get('address')}）" if item.get("address") else ""
                lines.append(f"- {item.get('name')}{address}")
            lines.append("- 餐饮建议放在每天晚间片区内解决，不为了单个餐厅跨区跑，体验会更稳。")
        else:
            food_lines = self._food_suggestions(destination, problem.budget)
            if food_lines:
                lines.extend(["", "美食建议", *food_lines])

        lines.extend(["", "预算参考"])
        lines.extend(budget_lines)

        suitability = self._suitability_lines(destination, preferences, itinerary_days)
        if suitability:
            lines.extend(["", "这条路线适合你吗？", *suitability])

        lines.extend(["", "注意事项"])
        lines.extend(self._tips(problem, weather, places, restaurants, hotels))
        if missing:
            lines.extend(["", missing])
        return "\n".join(lines)

    def _weather_answer(self, state: AgentState) -> str:
        problem = state.problem
        destination = problem.destination if problem else "目的地"
        weather = self._first_tool(state.tool_results, "weather_lookup")
        if weather and weather.success:
            payload = weather.payload
            return "\n".join(
                [
                    f"{destination}天气建议",
                    f"- 当前参考：{payload.get('date_label') or '近期'}，{payload.get('forecast') or weather.summary}",
                    f"- 温度：{self._temperature_text(payload)}",
                    f"- 出行判断：{payload.get('recommendation') or '建议出发前再确认实时天气。'}",
                    "- 行程建议：把受天气影响大的户外项目安排在上午或天气较稳时段，室内项目作为机动备选。",
                ]
            )
        return f"我暂时没有拿到 {destination} 的实时天气。建议出发前再确认天气；如果你给出具体日期，我可以继续把穿搭和室内外顺序排细。"

    def _transport_answer(self, state: AgentState) -> str:
        problem = state.problem
        route = self._first_tool(state.tool_results, "route_planning")
        if route and route.success:
            return "\n".join(
                [
                    "交通建议",
                    f"- {self._route_brief(state.tool_results)}",
                    "- 如果你更看重省心，优先选择直达或少换乘方案；如果更看重预算，再比较高铁、飞机、自驾的总成本。",
                    "- 到达后建议先把酒店选在核心交通节点附近，减少第一天和最后一天拖行李的成本。",
                ]
            )
        origin = problem.origin if problem else None
        destination = problem.destination if problem else None
        if origin and destination:
            return f"我暂时没有拿到 {origin} 到 {destination} 的实时路线结果。可以先按少换乘原则规划：优先选择直达交通，到达后住在地铁/核心交通节点附近。"
        return "要给出准确交通建议，我还需要出发地和目的地。你可以告诉我从哪里出发、到哪里，以及更看重省时还是省钱。"

    def _hotel_answer(self, state: AgentState) -> str:
        problem = state.problem
        destination = problem.destination if problem else "目的地"
        hotels = self._hotels(state.tool_results)
        lines = [f"{destination}住宿建议"]
        if problem and problem.budget:
            lines.append(f"- 按总预算 {problem.budget} 元估算，住宿建议控制在总预算的 30%-40%。")
        if problem and problem.constraints.get("hotel_near_metro"):
            lines.append("- 你偏好靠近地铁，优先选地铁站步行 10 分钟内、能直达核心景区的酒店。")
        if hotels:
            lines.append("")
            lines.append("可优先看的候选：")
            for hotel in hotels[:5]:
                price = f"约 {hotel['price']} 元/晚" if hotel.get("price") else "价格待确认"
                lines.append(f"- {hotel.get('name')}：{hotel.get('area') or hotel.get('location') or '位置待确认'}，{price}。")
        else:
            lines.append("- 暂时没有拿到可靠酒店候选，建议先锁定核心交通片区，再按评分、通勤和预算筛选。")
        lines.append("- 订房前重点看：距离地铁/景点、取消政策、房间面积、近期差评里的卫生和隔音。")
        return "\n".join(lines)

    def _focused_recommendation_answer(self, state: AgentState) -> str:
        problem = state.problem
        destination = problem.destination if problem else "目的地"
        intent_type = state.intent.type if state.intent else ""
        tool_name = "restaurant_recommendation" if intent_type == "restaurant_recommendation" else "place_search"
        title = "美食推荐" if tool_name == "restaurant_recommendation" else "景点推荐"
        items = self._places(state.tool_results, tool_name)
        lines = [f"{destination}{title}"]
        if items:
            for item in items[:6]:
                address = f"（{item.get('address')}）" if item.get("address") else ""
                lines.append(f"- {item.get('name')}{address}")
            lines.append("- 建议按片区串联，不要为了单个点频繁跨区。")
        else:
            lines.append("- 暂时没有拿到稳定候选，我建议先按你的偏好确定片区，再补充具体点位。")
        return "\n".join(lines)

    @staticmethod
    def _first_tool(tool_results: list[ToolResult], name: str) -> ToolResult | None:
        for item in tool_results:
            if item.name == name:
                return item
        return None

    def _weather_brief(self, tool_results: list[ToolResult]) -> str:
        result = self._first_tool(tool_results, "weather_lookup")
        if not result or not result.success:
            return ""
        payload = result.payload
        forecast = payload.get("forecast") or result.summary
        recommendation = payload.get("recommendation")
        temp = self._temperature_text(payload)
        pieces = [str(forecast)]
        if temp:
            pieces.append(temp)
        if recommendation:
            pieces.append(str(recommendation))
        return "；".join(piece for piece in pieces if piece)

    def _route_brief(self, tool_results: list[ToolResult]) -> str:
        result = self._first_tool(tool_results, "route_planning")
        if not result or not result.success:
            return ""
        payload = result.payload
        distance = payload.get("distance_km")
        duration = payload.get("duration_minutes")
        mode = self._mode_label(payload.get("mode"))
        pieces = []
        if distance:
            pieces.append(f"约 {distance} 公里")
        if duration:
            pieces.append(f"约 {duration} 分钟")
        if mode:
            pieces.append(mode)
        return "，".join(pieces) or result.summary

    def _places(self, tool_results: list[ToolResult], name: str) -> list[dict[str, Any]]:
        result = self._first_tool(tool_results, name)
        if not result or not isinstance(result.payload, dict):
            return []
        places = result.payload.get("places") or []
        filtered = [item for item in places if isinstance(item, dict) and item.get("name")]
        semantic_theme = str(result.payload.get("semantic_theme") or "")
        if name == "place_search" and semantic_theme:
            filtered = [item for item in filtered if not is_polluted_poi(item, semantic_theme)]
        return filtered

    def _hotels(self, tool_results: list[ToolResult]) -> list[dict[str, Any]]:
        result = self._first_tool(tool_results, "hotel_search")
        if not result or not isinstance(result.payload, dict):
            return []
        hotels = result.payload.get("hotels") or []
        return [item for item in hotels if isinstance(item, dict) and item.get("name")]

    def _xiaohongshu_brief(self, tool_results: list[ToolResult]) -> str:
        result = self._first_tool(tool_results, "xiaohongshu_search")
        if not result:
            return ""
        if result.success:
            insights = result.payload.get("insights") if isinstance(result.payload, dict) else None
            if isinstance(insights, list) and insights:
                return "；".join(str(item) for item in insights[:2])
            notes = result.payload.get("notes") if isinstance(result.payload, dict) else None
            if isinstance(notes, list) and notes:
                titles = "、".join(str(item.get("title")) for item in notes[:3] if item.get("title"))
                return f"已参考近期旅行笔记：{titles}"
        return "部分实时笔记暂时未取到，本版先依据天气、地图和预算做保守安排。"

    @staticmethod
    def _tool_limit_notice(tool_results: list[ToolResult]) -> str:
        if any(
            item.name in AnswerGenerator.CRITICAL_TOOL_NAMES and not item.success and item.error
            for item in tool_results
        ):
            return "部分实时地图、交通、住宿或餐饮信息暂时未取到；下面先按常规动线和预算给可执行初版，出发前再核对班次、开放时间和价格。"
        return ""

    def _itinerary_days(
        self,
        destination: str,
        days: int,
        preferences: list[str],
        places: list[dict[str, Any]],
        restaurants: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        curated = self._curated_itinerary(destination, preferences)
        if curated:
            return curated[:days]
        return self._generic_itinerary_days(destination, days, preferences, places, restaurants)

    @staticmethod
    def _curated_itinerary(destination: str, preferences: list[str]) -> list[dict[str, Any]]:
        preference_text = "、".join(preferences)
        if destination == "云南":
            return [
                {
                    "day": 1,
                    "title": "昆明抵达与城市慢逛",
                    "morning": {
                        "name": "抵达昆明 / 翠湖公园",
                        "description": "第一天先把节奏放轻，抵达后去翠湖、公园周边和老街适应海拔与天气。",
                        "highlights": ["翠湖散步", "云南大学周边", "昆明老街", "轻松拍照"],
                        "duration": "2-3 小时",
                    },
                    "afternoon": {
                        "name": "昆明老街 / 南强街",
                        "description": "适合情侣慢逛、喝咖啡、拍街景，强度低，也方便安排第一顿云南菜。",
                        "highlights": ["老街建筑", "咖啡小店", "鲜花饼", "过桥米线"],
                        "duration": "2 小时左右",
                    },
                    "evening": {
                        "name": "昆明夜市 / 早休息",
                        "description": "晚上吃本地小吃后早点休息，为第二天去大理留体力。",
                        "highlights": ["菌菇火锅", "烧烤小吃", "少折返"],
                        "duration": "1.5 小时左右",
                    },
                },
                {
                    "day": 2,
                    "title": "大理洱海与古城",
                    "morning": {
                        "name": "昆明到大理 / 洱海生态廊道",
                        "description": "上午高铁到大理后直奔洱海边，选择一段生态廊道骑行或散步，不必环完整圈。",
                        "highlights": ["洱海风景", "骑行", "海边合照", "节奏舒适"],
                        "duration": "3 小时左右",
                    },
                    "afternoon": {
                        "name": "喜洲古镇 / 双廊二选一",
                        "description": "想轻松拍照选喜洲，想看海景民宿氛围选双廊；情侣游建议不要两边都赶。",
                        "highlights": ["白族建筑", "麦田风景", "海景咖啡", "慢节奏"],
                        "duration": "2-3 小时",
                    },
                    "evening": {
                        "name": "大理古城",
                        "description": "晚上回古城吃饭和散步，住宿也建议放在古城或洱海西线附近。",
                        "highlights": ["古城夜景", "白族菜", "酒吧街外圈散步"],
                        "duration": "2 小时左右",
                    },
                },
                {
                    "day": 3,
                    "title": "丽江古城与雪山远景",
                    "morning": {
                        "name": "大理到丽江 / 束河古镇",
                        "description": "上午去丽江后先逛束河，氛围比大研古城更安静，适合情侣慢游。",
                        "highlights": ["纳西古镇", "溪流巷道", "咖啡小院", "拍照"],
                        "duration": "2-3 小时",
                    },
                    "afternoon": {
                        "name": "白沙古镇 / 玉龙雪山远眺",
                        "description": "预算 5000 元内不强行安排高成本雪山大套票，先用白沙古镇和雪山远景控制花费。",
                        "highlights": ["雪山远景", "白沙壁画周边", "安静街巷", "情侣合照"],
                        "duration": "2-3 小时",
                    },
                    "evening": {
                        "name": "丽江古城",
                        "description": "晚上进大研古城看夜景和吃饭，避开过度商业化主街，走外围巷子会舒服一些。",
                        "highlights": ["古城夜景", "腊排骨", "纳西烤鱼", "返程缓冲"],
                        "duration": "2 小时左右",
                    },
                },
            ]
        if not any(keyword in preference_text for keyword in ("自然", "风景", "山水", "湿地", "森林", "海边", "徒步")):
            return []
        if destination == "厦门":
            return [
                {
                    "day": 1,
                    "title": "海岸线与城市初体验",
                    "morning": {
                        "name": "厦门园林植物园",
                        "description": "把植物园放在上午更舒服，热带雨林区、多肉植物区和观景点都适合慢慢逛。",
                        "highlights": ["雨林雾森", "多肉植物区", "山海城市视野", "拍照出片"],
                        "duration": "3-4 小时",
                    },
                    "afternoon": {
                        "name": "沙坡尾 / 演武大桥观景平台",
                        "description": "下午转到海边城市片区，强度不高，适合从植物园过渡到厦门的海岸氛围。",
                        "highlights": ["海港街区", "演武大桥海景", "咖啡小店", "轻松散步"],
                        "duration": "2 小时左右",
                    },
                    "evening": {
                        "name": "白城沙滩 - 环岛路",
                        "description": "傍晚沿海边散步或骑行，看日落和海风，第一天不要排太满。",
                        "highlights": ["海边日落", "环岛路骑行", "沙滩散步", "夜景轻体验"],
                        "duration": "1.5-2 小时",
                    },
                },
                {
                    "day": 2,
                    "title": "鼓浪屿慢游",
                    "morning": {
                        "name": "鼓浪屿",
                        "description": "上午坐船上岛，优先走菽庄花园、海边步道和安静小巷，避开只打卡商业街。",
                        "highlights": ["海岛步道", "万国建筑", "菽庄花园", "慢节奏散步"],
                        "duration": "3-4 小时",
                    },
                    "afternoon": {
                        "name": "日光岩 / 港仔后沙滩",
                        "description": "天气好可登日光岩看海岛全景；如果想轻松，就改去港仔后沙滩和海边咖啡。",
                        "highlights": ["海岛全景", "沙滩", "海风", "摄影"],
                        "duration": "2-3 小时",
                    },
                    "evening": {
                        "name": "中山路 / 鹭江道",
                        "description": "返程回本岛后吃晚餐，饭后沿鹭江道看夜景，比继续赶景点更稳。",
                        "highlights": ["厦门夜景", "本地小吃", "海景散步"],
                        "duration": "1.5-2 小时",
                    },
                },
                {
                    "day": 3,
                    "title": "山海收尾",
                    "morning": {
                        "name": "五缘湾湿地公园",
                        "description": "最后一天安排湿地和海湾，路程相对可控，也符合自然风光偏好。",
                        "highlights": ["湿地栈道", "水鸟观赏", "湾区风景", "轻松步行"],
                        "duration": "2-3 小时",
                    },
                    "afternoon": {
                        "name": "集美学村 / 十里长堤",
                        "description": "如果返程时间允许，可去集美看海堤和学村建筑；时间紧就直接回酒店取行李。",
                        "highlights": ["十里长堤海景", "集美学村", "傍海散步", "返程顺路"],
                        "duration": "2 小时左右",
                    },
                    "evening": {
                        "name": "返程",
                        "description": "预留退房、取行李和去机场/车站时间，不建议最后再安排远距离景点。",
                        "highlights": ["返程缓冲", "伴手礼", "减少赶路"],
                        "duration": "按车次/航班调整",
                    },
                },
            ]
        if destination != "广州":
            return []
        return [
            {
                "day": 1,
                "title": "登山看广州全景",
                "morning": {
                    "name": "白云山",
                    "description": "推荐乘索道上山、步行下山，上午空气更清爽，也更适合看城市天际线。",
                    "highlights": ["森林覆盖率高，空气清新", "多条徒步路线可选", "山顶可俯瞰广州城市全景", "适合拍日间风光"],
                    "duration": "3-4 小时",
                },
                "afternoon": {
                    "name": "云台花园",
                    "description": "位于白云山南麓，适合把登山后的下午放轻松。",
                    "highlights": ["欧式园林", "四季花海", "喷泉景观", "摄影出片率高"],
                    "duration": "约 2 小时",
                },
                "evening": {
                    "name": "珠江两岸",
                    "description": "晚餐后去珠江边散步；预算足够可选珠江夜游，更偏自然轻松就沿江步道慢走。",
                    "highlights": ["珠江夜景", "花城广场", "广州塔外观拍照"],
                    "duration": "1.5-2 小时",
                },
            },
            {
                "day": 2,
                "title": "湿地生态一日",
                "morning": {
                    "name": "海珠国家湿地公园",
                    "description": "广州代表性的城市湿地，适合安排一个完整上午慢慢走。",
                    "highlights": ["木栈道漫步", "湖泊风景", "鸟类观赏", "芦苇湿地"],
                    "duration": "3 小时左右",
                },
                "afternoon": {
                    "name": "海珠湖",
                    "description": "距离湿地较近，下午适合骑行或湖边散步，节奏比继续跨区更舒服。",
                    "highlights": ["骑自行车", "湖边散步", "拍日落", "城市水域景观"],
                    "duration": "2 小时左右",
                },
                "evening": {
                    "name": "广州塔周边",
                    "description": "晚上回到广州塔周边，登塔看夜景或沿江散步二选一。",
                    "highlights": ["广州塔夜景", "海心桥/江边步道", "晚餐选择多"],
                    "duration": "1.5-2 小时",
                },
            },
            {
                "day": 3,
                "title": "森林放松",
                "morning": {
                    "name": "流溪河国家森林公园",
                    "description": "更推荐给喜欢森林、湖泊和安静自然环境的人；如果返程较早，可改为离市区更近的大夫山森林公园。",
                    "highlights": ["森林徒步", "湖边漫步", "呼吸新鲜空气", "自然气息浓"],
                    "duration": "半天到大半天",
                },
                "afternoon": {
                    "name": "返回市区 / 大夫山森林公园备选",
                    "description": "如果上午去流溪河，下午预留返程缓冲；如果选大夫山，可半天游玩后回市区拿行李。",
                    "highlights": ["骑行绿道", "湖泊", "森林", "草坪"],
                    "duration": "2-3 小时",
                },
                "evening": {
                    "name": "返程",
                    "description": "最后一天不建议再塞太远景点，避免误车或拖行李赶路。",
                    "highlights": ["退房", "伴手礼", "返程缓冲"],
                    "duration": "按车次/航班调整",
                },
            },
        ]

    def _generic_itinerary_days(
        self,
        destination: str,
        days: int,
        preferences: list[str],
        places: list[dict[str, Any]],
        restaurants: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        del preferences
        result: list[dict[str, Any]] = []
        for day in range(1, days + 1):
            morning = self._pick_name(places, (day - 1) * 2, f"{destination}核心景点")
            afternoon = self._pick_name(places, (day - 1) * 2 + 1, "同片区轻松点位")
            evening = self._pick_name(restaurants, day - 1, "当地特色餐厅 / 夜景散步")
            if day == 1:
                title = "抵达与城市初体验"
            elif day == days:
                title = "轻松收尾与返程缓冲"
            else:
                title = f"{morning}片区深度游"
            result.append(
                {
                    "day": day,
                    "title": title,
                    "morning": {
                        "name": morning,
                        "description": "把当天最重要的景点放在上午，体力和光线都更稳定。",
                        "highlights": ["核心体验", "避开午后高温", "适合拍照"],
                        "duration": "2-3 小时",
                    },
                    "afternoon": {
                        "name": afternoon,
                        "description": "下午接同片区或相邻片区，减少跨城折返。",
                        "highlights": ["顺路串联", "节奏更轻松", "留出休息时间"],
                        "duration": "2 小时左右",
                    },
                    "evening": {
                        "name": evening,
                        "description": "晚餐尽量放在当天片区附近解决，饭后可安排轻松散步。",
                        "highlights": ["就近吃饭", "夜景/散步", "减少通勤"],
                        "duration": "1.5-2 小时",
                    },
                }
            )
        return result

    @staticmethod
    def _overview_table(itinerary_days: list[dict[str, Any]]) -> list[str]:
        lines = ["| 天数 | 上午 | 下午 | 晚上 |", "| --- | --- | --- | --- |"]
        for item in itinerary_days:
            lines.append(
                f"| Day {item.get('day')} | "
                f"{item.get('morning', {}).get('name', '')} | "
                f"{item.get('afternoon', {}).get('name', '')} | "
                f"{item.get('evening', {}).get('name', '')} |"
            )
        return lines

    def _formatted_day_plan(self, item: dict[str, Any], constraints: dict[str, Any]) -> list[str]:
        day = item.get("day")
        title = item.get("title") or "当日安排"
        lines = ["", f"Day {day}：{title}"]
        for label, key in (("上午", "morning"), ("下午", "afternoon"), ("晚上", "evening")):
            slot = item.get(key) if isinstance(item.get(key), dict) else {}
            lines.extend(self._slot_lines(label, slot, constraints))
        return lines

    @staticmethod
    def _slot_lines(label: str, slot: dict[str, Any], constraints: dict[str, Any]) -> list[str]:
        name = str(slot.get("name") or "机动安排")
        description = str(slot.get("description") or "按当天体力和天气灵活安排。")
        duration = str(slot.get("duration") or "")
        highlights = [str(item) for item in slot.get("highlights", []) if item]
        lines = [f"{label}：{name}", f"- {description}"]
        if highlights:
            lines.append("- 亮点：" + "、".join(highlights[:4]))
        if duration:
            lines.append(f"- 建议游玩时间：{duration}")
        if constraints.get("pace") == "relaxed" or constraints.get("low_walking"):
            lines.append("- 节奏提醒：中间留 1 小时左右休息，不要把点位排满。")
        return lines

    @staticmethod
    def _lodging_suggestions(destination: str, budget: int | None, nights: int) -> list[str]:
        if destination == "广州":
            lines = [
                "- 优先住：珠江新城、体育西路、客村、昌岗一带，兼顾地铁、夜景和去白云山/海珠湿地的效率。",
                "- 如果更看重自然安静，可选靠近白云山南门或海珠湖周边，但晚上餐饮选择会比市中心少一些。",
            ]
        elif destination == "厦门":
            lines = [
                "- 优先住：思明区中山路、厦大/沙坡尾、白城沙滩或曾厝垵附近，方便衔接植物园、鼓浪屿码头和环岛路。",
                "- 如果更看重海边氛围，可以选环岛路/曾厝垵；如果更看重交通和吃饭便利，中山路/轮渡附近会更稳。",
            ]
        elif destination == "云南":
            lines = [
                "- 建议分段住：昆明 1 晚住翠湖/老街周边，大理 1 晚住古城或洱海西线，丽江可住束河或古城外圈。",
                "- 情侣出行优先选安静、交通方便、可步行吃饭的客栈，不必追求过贵海景房，把预算留给交通和体验更稳。",
            ]
        else:
            lines = ["- 建议优先选核心景区与地铁/公交之间的中间区域，避免每天跨城折返。"]
        if budget and nights:
            per_night = int(budget * 0.32 / max(nights, 1))
            lines.append(f"- 住宿预算可先按约 {per_night} 元/晚控制，再根据评分、地铁距离和取消政策筛选。")
        return lines

    @staticmethod
    def _food_suggestions(destination: str, budget: int | None) -> list[str]:
        if destination == "广州":
            lines = [
                "- 自然景点游玩之余，可以穿插广州特色：早茶、烧鹅、白切鸡、云吞面、双皮奶、肠粉。",
                "- 早茶适合放在 Day 1 或 Day 3 上午前后；晚餐尽量选当天片区附近，避免为了单个餐厅跨区。",
            ]
        elif destination == "厦门":
            lines = [
                "- 可以穿插厦门特色：沙茶面、海蛎煎、姜母鸭、土笋冻、花生汤、烧肉粽。",
                "- 鼓浪屿当天不建议为单个网红店排太久，晚餐回中山路/鹭江道一带选择会更稳。",
            ]
        elif destination == "云南":
            lines = [
                "- 可以穿插云南特色：过桥米线、菌菇火锅、白族菜、乳扇、鲜花饼、腊排骨火锅。",
                "- 预算 5000 元内建议每天 1 顿特色正餐 + 1 顿轻食小吃，不为网红店长距离跨城绕路。",
            ]
        else:
            return []
        if budget:
            lines.append(f"- 按总预算 {budget} 元估算，餐饮可以预留约 {int(budget * 0.18)}-{int(budget * 0.24)} 元。")
        return lines

    @staticmethod
    def _suitability_lines(destination: str, preferences: list[str], itinerary_days: list[dict[str, Any]]) -> list[str]:
        text = "、".join(preferences)
        if destination == "广州" and any(keyword in text for keyword in ("自然", "风景", "山水", "湿地", "森林")):
            return [
                "- 如果你偏爱自然风景，这条路线重点放在山林、湿地、湖泊和珠江夜景。",
                "- 相比只逛广州塔、北京路、上下九等热门商圈，这样更能体验广州生态和自然的一面。",
                "- 如果返程时间紧，Day 3 建议把流溪河换成大夫山森林公园，整体会更稳。",
            ]
        if destination == "厦门" and any(keyword in text for keyword in ("自然", "风景", "山水", "海边", "湿地", "森林")):
            return [
                "- 如果你偏爱自然风光，这条路线重点放在海岸线、植物园、鼓浪屿海岛和湿地湾区。",
                "- 相比只逛商业街和网红店，这样能更好体验厦门的山海城市气质。",
                "- 如果遇到高温或下雨，Day 1 的植物园和 Day 2 的鼓浪屿都可以压缩时长，把更多时间留给海边散步和咖啡休息。",
            ]
        if destination == "云南":
            return [
                "- 这条线更适合第一次去云南、预算有限又想兼顾风景和氛围的情侣。",
                "- 3 天只能做昆明 + 大理 + 丽江快线初稿；如果能加到 5-6 天，建议把大理和丽江各多留 1 天，体验会明显更松弛。",
            ]
        if itinerary_days:
            return ["- 这版安排按每天 1 个主片区控制通勤，适合先作为初稿，再根据酒店位置和出发时间微调。"]
        return []

    def _day_plan(
        self,
        day: int,
        days: int,
        destination: str,
        places: list[dict[str, Any]],
        restaurants: list[dict[str, Any]],
        preferences: list[str],
        constraints: dict[str, Any],
    ) -> list[str]:
        theme = self._day_area(day, destination, places)
        morning = self._pick_name(places, day - 1, f"{destination}核心景点")
        afternoon = self._pick_name(places, day, "同片区自然/人文点位")
        dinner = self._pick_name(restaurants, day - 1, "当地特色餐厅")
        pace_hint = "中间留 1-1.5 小时机动休息" if constraints.get("pace") == "relaxed" or constraints.get("low_walking") else "尽量按同片区顺路串联"
        if day == 1:
            return [
                f"第 {day} 天：抵达与城市初体验",
                f"- 上午/中午：抵达 {destination}，先去酒店寄存行李，熟悉周边交通。",
                f"- 下午：安排 {morning}，强度不要太高，{pace_hint}。",
                f"- 晚上：在 {dinner} 附近吃饭，饭后可补一个轻松夜景或散步点。",
            ]
        if day == days:
            return [
                f"第 {day} 天：轻松收尾与返程缓冲",
                f"- 上午：安排 {morning} 或补拍照点，避免太远的跨区行程。",
                "- 下午：预留退房、伴手礼和返程时间。",
                f"- 晚上：如果不返程，可选择 {dinner}，行程保持轻量。",
            ]
        return [
            f"第 {day} 天：{theme}",
            f"- 上午：优先去 {morning}，把最想看的自然/风景项目放在体力最好时段。",
            f"- 下午：接 {afternoon}，{pace_hint}。",
            f"- 晚上：选择 {dinner}，不建议为了单个餐厅跨太远。",
        ]

    def _day_area(self, day: int, destination: str, places: list[dict[str, Any]]) -> str:
        if not places:
            defaults = ["核心城区/地标区", "自然风景或主题片区", "轻松收尾/返程缓冲"]
            return defaults[min(day - 1, len(defaults) - 1)]
        first = self._pick_name(places, day - 1, f"{destination}核心景点")
        second = self._pick_name(places, day, "")
        if second:
            return f"{first} + {second}"
        return first

    @staticmethod
    def _pick_name(items: list[dict[str, Any]], index: int, fallback: str) -> str:
        if not items:
            return fallback
        item = items[index % len(items)]
        return str(item.get("name") or fallback)

    @staticmethod
    def _trip_theme(preferences: list[str]) -> str:
        text = "、".join(preferences)
        if any(keyword in text for keyword in ("自然", "风景", "海边", "徒步")):
            return "自然风景 + 城市轻体验"
        if any(keyword in text for keyword in ("历史", "博物馆", "人文")):
            return "历史人文 + 经典地标"
        if "美食" in text:
            return "美食探店 + 轻景点"
        if "亲子" in text:
            return "亲子轻松 + 低强度动线"
        return "经典景点 + 美食 + 休闲"

    @staticmethod
    def _budget_title(budget: int | None) -> str:
        return f"总预算约 {budget} 元" if budget else "预算暂未明确"

    def _budget_lines(self, budget: int | None, days: int, group_size: int) -> list[str]:
        if not budget:
            return [
                "- 住宿：建议占总预算 30%-40%。",
                "- 市内交通：建议占 15%-25%，优先同片区串联来省通勤。",
                "- 餐饮与门票：建议占 30%左右。",
                "- 机动金：至少预留 10%，应对天气、排队或临时改路线。",
            ]
        lodging_low = int(budget * 0.30)
        lodging_high = int(budget * 0.40)
        transport_low = int(budget * 0.15)
        transport_high = int(budget * 0.25)
        food_low = int(budget * 0.20)
        food_high = int(budget * 0.30)
        ticket_low = int(budget * 0.10)
        ticket_high = int(budget * 0.18)
        reserve = max(int(budget * 0.10), 200)
        per_person = int(budget / max(group_size, 1))
        coffee = max(int(budget * 0.06), 120)
        souvenir = max(int(budget * 0.10), 200)
        return [
            f"- 总预算：约 {budget} 元，折合人均约 {per_person} 元。",
            "| 项目 | 建议预算 |",
            "| --- | --- |",
            f"| 住宿 | {lodging_low}-{lodging_high} 元 |",
            f"| 餐饮 | {food_low}-{food_high} 元 |",
            f"| 景点门票/索道/夜游 | {ticket_low}-{ticket_high} 元 |",
            f"| 市内交通 | {transport_low}-{transport_high} 元 |",
            f"| 咖啡/下午茶 | 约 {coffee} 元 |",
            f"| 伴手礼 | 约 {souvenir} 元 |",
            f"| 预留机动 | 至少 {reserve} 元 |",
        ]

    @staticmethod
    def _assumption_lines(problem: Any, group_size: int) -> list[str]:
        lines: list[str] = []
        if not problem.group_size:
            lines.append(f"- 人数未说明，暂按 {group_size} 人估算。")
        if not problem.date_range:
            lines.append("- 日期未说明，天气和开放时间按当前可用信息做保守参考，出发前建议再确认。")
        if problem.assumptions:
            for item in problem.assumptions:
                if not item:
                    continue
                if not problem.group_size and "人数" in item:
                    continue
                if not problem.days and ("3 天" in item or "3天" in item):
                    continue
                formatted = f"- {item}"
                if formatted not in lines:
                    lines.append(formatted)
        return lines

    def _tips(
        self,
        problem: Any,
        weather: str,
        places: list[dict[str, Any]],
        restaurants: list[dict[str, Any]],
        hotels: list[dict[str, Any]],
    ) -> list[str]:
        tips = []
        if weather and any(keyword in weather for keyword in ("雨", "雪", "雾", "霾")):
            tips.append("- 天气不稳时，把户外自然点放在上午，下午准备博物馆/室内展馆备选。")
        else:
            tips.append("- 热门景点尽量上午去，下午安排同片区轻松项目，体验会更稳。")
        if problem.constraints.get("transport_mode") == "public":
            tips.append("- 你偏向公共交通，住宿尽量选地铁/公交换乘少的位置。")
        if places:
            tips.append("- 上面景点建议按片区二次排序，避免每天来回横穿城市。")
        if restaurants:
            tips.append("- 美食不要排得过满，留出排队和临时调整时间。")
        if hotels:
            tips.append("- 酒店下单前重点看近期差评、交通步行距离和取消政策。")
        return tips

    @staticmethod
    def _temperature_text(payload: dict[str, Any]) -> str:
        low = payload.get("temperature_min")
        high = payload.get("temperature_max")
        current = payload.get("current_temperature")
        if low is not None and high is not None:
            return f"{low}-{high}℃"
        if current is not None:
            return f"当前约 {current}℃"
        return ""

    @staticmethod
    def _mode_label(mode: Any) -> str:
        normalized = str(mode or "").lower()
        return {
            "transit": "公共交通",
            "driving": "驾车",
            "walking": "步行",
            "bicycling": "骑行",
        }.get(normalized, str(mode or ""))

    @staticmethod
    def _missing_info_text(missing_info: list[str]) -> str:
        remaining = [item for item in missing_info if item != "destination"]
        if not remaining:
            return ""
        labels = {
            "days": "旅行天数",
            "budget": "预算",
            "group_size": "人数",
        }
        readable = "、".join(labels.get(item, item) for item in remaining)
        return f"为了把方案继续细化，我还需要你补充：{readable}。"

    @staticmethod
    def _constraint_text(constraints: dict[str, Any]) -> str:
        if not constraints:
            return ""
        labels = {
            "transport_mode": {"public": "优先公共交通", "driving": "适合自驾"},
            "family_friendly": "亲子友好",
            "low_walking": "减少步行",
            "pace": {"relaxed": "慢节奏", "intensive": "紧凑高效"},
            "hotel_near_metro": "住宿尽量靠近地铁",
            "avoid_paid_attractions": "优先免费或低门票景点",
            "dietary_preference": "饮食偏好",
        }
        hidden_keys = {"travel_theme", "requested_region", "tool_destination", "region_resolution_note"}
        readable: list[str] = []
        for key, value in constraints.items():
            if key in hidden_keys:
                continue
            if value in (False, None, ""):
                continue
            label = labels.get(key, key)
            if isinstance(label, dict):
                readable.append(label.get(str(value), f"{key}={value}"))
            elif value is True:
                readable.append(str(label))
            else:
                readable.append(f"{label}：{value}")
        return "、".join(readable)
