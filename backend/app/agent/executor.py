from __future__ import annotations

from typing import Any

from backend.app.services.llm_service import DashScopeLLMService


class Executor:
    def __init__(self) -> None:
        self.llm = DashScopeLLMService()

    def synthesize(self, state: dict[str, Any]) -> str:
        if self.llm.available():
            return self._llm_synthesize(state)
        return self._fallback_synthesize(state)

    def _llm_synthesize(self, state: dict[str, Any]) -> str:
        intent = state.get("intent", {})
        plan = state.get("plan", {})
        docs = state.get("retrieved_docs", [])
        tools = state.get("tool_results", {})
        memory_context = state.get("memory_context", {})
        question = state.get("question", "")

        context_lines = [
            f"用户问题：{question}",
            f"任务类型：{intent.get('domain', 'chat')}",
            f"规划摘要：{plan.get('summary', '')}",
            f"步骤：{' / '.join(plan.get('steps', []))}",
            f"检索资料：{self._format_docs(docs)}",
            f"工具结果：{self._format_tools(tools)}",
            f"历史记忆：{self._format_memory(memory_context)}",
        ]
        messages = [
            {
                "role": "system",
                "content": "你是一个专业、务实、会给出可执行建议的中文旅行规划助手。请直接回答用户问题，避免空话套话。若信息不足，先给出合理默认值并明确标注假设。输出要包含：1. 简短结论 2. 具体行程或建议 3. 预算/交通/住宿要点 4. 如有不确定项，说明原因。",
            },
            {
                "role": "user",
                "content": "\n".join(context_lines),
            },
        ]
        try:
            answer = self.llm.chat(messages, temperature=0.3).strip()
        except Exception:
            answer = ""
        return answer or self._fallback_synthesize(state)

    def _fallback_synthesize(self, state: dict[str, Any]) -> str:
        intent = state.get("intent", {})
        docs = state.get("retrieved_docs", [])
        tools = state.get("tool_results", {})
        question = state.get("question", "")
        memory_context = state.get("memory_context", {})
        domain = intent.get("domain", "chat")

        if domain == "plan":
            return self._travel_plan_fallback(question, docs, tools, memory_context)
        if domain == "route":
            return self._route_fallback(question, tools)
        if domain == "hotel":
            return self._hotel_fallback(question, docs)
        if domain == "weather":
            return self._weather_fallback(question, tools)

        return self._general_fallback(question, docs, tools, memory_context)

    def _travel_plan_fallback(self, question: str, docs: list[dict[str, Any]], tools: dict[str, Any], memory_context: dict[str, Any]) -> str:
        days = self._extract_days(question) or self._guess_days(memory_context) or 3
        destination = self._extract_destination(question) or self._guess_destination(memory_context) or "目的地城市"
        budget = self._extract_budget(question) or self._guess_budget(memory_context) or 3000
        origin = self._guess_origin(question, memory_context)
        transport = "自驾" if any(token in question for token in ["自驾", "开车"]) else "公共交通"
        route_text = self._format_route(tools)
        doc_hint = self._pick_doc_hint(docs)

        lines = [
            f"我先按你给的信息做一个可执行方案：从{origin}去{destination}{days}日游，预算约{budget}元，优先按{transport}考虑。",
            f"参考信息：{doc_hint}",
            "建议安排如下：",
        ]
        for day in range(1, days + 1):
            if day == 1:
                lines.append(f"第{day}天：出发到达 {destination}，下午安排市区核心景点，晚上吃当地特色美食。")
            elif day == days:
                lines.append(f"第{day}天：轻松游玩 + 返程，预留机动时间，避免赶路。")
            else:
                lines.append(f"第{day}天：安排一个主题路线，比如自然风景、博物馆或老城区漫游。")
        lines.extend([
            f"预算建议：交通约{int(budget * 0.35)}元，住宿约{int(budget * 0.3)}元，餐饮和门票约{int(budget * 0.25)}元，机动费用约{int(budget * 0.1)}元。",
            f"{route_text}" if route_text else "",
            "如果你愿意，我可以继续细化成‘景点-餐厅-酒店’的详细版。",
        ])
        return "\n".join(line for line in lines if line)

    def _route_fallback(self, question: str, tools: dict[str, Any]) -> str:
        route_text = self._format_route(tools)
        if route_text:
            return f"根据你的需求，我建议优先采用工具给出的路线信息：\n{route_text}\n如果你告诉我起点、终点和出发时间，我还能继续帮你细化到导航级别。"
        origin = self._guess_origin(question, {})
        destination = self._extract_destination(question) or "目的地"
        return f"你这个问题主要是路线/交通类问题。当前我先按常规建议：从{origin}到{destination}时，优先比较时间成本和停车成本；如果是长途自驾，建议中途安排休息点。你补充起点、终点和出发时间后，我可以给你更准确的路线建议。"

    def _hotel_fallback(self, question: str, docs: list[dict[str, Any]]) -> str:
        destination = self._extract_destination(question) or "目的地"
        if docs:
            return f"关于 {destination} 的住宿建议，我建议优先选择市中心或景区地铁沿线，方便出行。参考到的资料里也有相关住宿信息，下一步我可以继续帮你筛选预算和区域。"
        return f"关于 {destination} 的住宿建议，我建议优先选地铁方便、评分稳定、离主要景点不太远的区域。"

    def _weather_fallback(self, question: str, tools: dict[str, Any]) -> str:
        _ = self._guess_destination({})
        if tools:
            return f"关于天气问题，我已经拿到部分工具结果。你如果告诉我具体城市和日期，我可以继续帮你判断是否适合出行、是否要带伞和怎么调整行程。"
        return f"关于天气问题，我建议你补充具体城市和日期，我可以帮你判断当天适不适合出行，以及要不要调整安排。"

    def _general_fallback(self, question: str, docs: list[dict[str, Any]], tools: dict[str, Any], memory_context: dict[str, Any]) -> str:
        parts = ["我理解你的问题。"]
        if docs:
            parts.append(f"我参考了 {len(docs)} 条资料。")
        if tools:
            parts.append(f"也结合了工具结果：{', '.join(tools.keys())}。")
        if memory_context:
            parts.append("同时考虑了当前会话的记忆信息。")
        parts.append("如果你愿意，我可以继续把它整理成更具体、可执行的建议。")
        return "\n".join(parts)

    def _format_docs(self, docs: list[dict[str, Any]]) -> str:
        if not docs:
            return "无"
        return "；".join(doc.get("content", "") for doc in docs[:3])

    def _format_tools(self, tools: dict[str, Any]) -> str:
        if not tools:
            return "无"
        return "; ".join(f"{k}: {v}" for k, v in tools.items())

    def _format_memory(self, memory_context: dict[str, Any]) -> str:
        if not memory_context:
            return "无"
        return str(memory_context)

    def _pick_doc_hint(self, docs: list[dict[str, Any]]) -> str:
        if not docs:
            return "暂无检索资料，以下方案基于常规旅行规划经验。"
        first = docs[0]
        return str(first.get("content") or first.get("title") or "已检索到相关资料")[:180]

    def _guess_destination(self, memory_context: dict[str, Any]) -> str | None:
        if not memory_context:
            return None
        for key in ["destination", "city", "target_city", "trip_city"]:
            value = memory_context.get(key)
            if value:
                return str(value)
        return None

    def _guess_days(self, memory_context: dict[str, Any]) -> int | None:
        if not memory_context:
            return None
        for key in ["days", "trip_days", "duration"]:
            value = memory_context.get(key)
            if isinstance(value, int) and value > 0:
                return value
        return None

    def _guess_budget(self, memory_context: dict[str, Any]) -> int | None:
        if not memory_context:
            return None
        for key in ["budget", "trip_budget"]:
            value = memory_context.get(key)
            if isinstance(value, int) and value > 0:
                return value
        return None

    def _guess_origin(self, question: str, memory_context: dict[str, Any]) -> str:
        for city in ["济南", "北京", "上海", "杭州", "南京", "苏州", "青岛", "天津", "广州", "深圳"]:
            if city in question:
                return city
        origin = memory_context.get("origin") if memory_context else None
        return str(origin) if origin else "出发地"

    def _format_route(self, tools: dict[str, Any]) -> str:
        route = tools.get("route")
        if isinstance(route, dict):
            distance = route.get("distance_km")
            duration = route.get("duration_minutes")
            if distance or duration:
                return f"路线提示：预计距离约{distance if distance is not None else '未知'}公里，耗时约{duration if duration is not None else '未知'}分钟。"
        return ""

    def _extract_days(self, question: str) -> int | None:
        import re

        match = re.search(r"(\d+)\s*日", question)
        return int(match.group(1)) if match else None

    def _extract_budget(self, question: str) -> int | None:
        import re

        match = re.search(r"(\d+)\s*元", question)
        return int(match.group(1)) if match else None

    def _extract_destination(self, question: str) -> str | None:
        import re

        patterns = [
            r"去([\u4e00-\u9fa5A-Za-z0-9]+)",
            r"规划一个([\u4e00-\u9fa5A-Za-z0-9]+)",
            r"([\u4e00-\u9fa5A-Za-z0-9]+)\d+日游",
            r"帮我规划一个([\u4e00-\u9fa5A-Za-z0-9]+)",
            r"([\u4e00-\u9fa5A-Za-z0-9]+)旅行",
        ]
        for pattern in patterns:
            match = re.search(pattern, question)
            if match:
                return match.group(1)
        return None
