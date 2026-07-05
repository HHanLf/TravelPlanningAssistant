from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

from backend.app.core.config import get_settings
from backend.app.services.llm_service import DashScopeLLMService
from backend.app.services.multimodal_service import MultimodalService
from backend.app.services.redis_memory import RedisMemoryService
from backend.app.tools.amap_tool import TencentMapTool
from backend.app.tools.ctrip_tool import CtripTool
from backend.app.tools.search_tool import SearchTool
from backend.app.tools.weather_tool import WeatherTool
from backend.app.tools.xiaohongshu_tool import XiaohongshuTool


settings = get_settings()
logger = logging.getLogger(__name__)


@dataclass
class PlanCard:
    title: str
    subtitle: str | None = None
    details: list[str] | None = None


class TravelAgentService:
    def __init__(self) -> None:
        self.memory = RedisMemoryService()
        self.tencent_map = TencentMapTool(settings.tencent_map_api_key)
        self.ctrip = CtripTool(settings.ctrip_api_key, map_tool=self.tencent_map)
        self.search = SearchTool(settings.search_api_key)
        self.weather = WeatherTool(settings.weather_api_key or settings.tencent_map_api_key)
        self.xiaohongshu = XiaohongshuTool(settings.justoneapi_token, settings.justoneapi_base_url)
        self.llm = DashScopeLLMService()
        self.multimodal = MultimodalService()

    def respond(
        self,
        session_id: str,
        message: str,
        history: list[tuple[str, str]],
        image_path: str | None = None,
        audio_path: str | None = None,
        image_data: str | None = None,
        multimodal: bool = False,
    ) -> dict:
        profile = self.memory.get_profile(session_id)
        multimodal_summary = None
        audio_transcript = None
        effective_message = message

        if audio_path:
            audio_result = self.multimodal.transcribe_audio(audio_path)
            audio_transcript = audio_result.get("text") or None
            if audio_transcript:
                effective_message = f"{message.strip()}\n\n[语音识别补充]\n{audio_transcript}".strip() if message.strip() else audio_transcript
            elif not message.strip():
                answer = audio_result.get("error") or "未识别到有效语音内容。"
                self._save_turn(session_id, message or "[voice]", answer)
                return {
                    "session_id": session_id,
                    "answer": answer,
                    "sources": ["multimodal_service"],
                    "intent": "voice_input",
                    "action_taken": "audio_transcribe_failed",
                    "need_more_info": False,
                    "plan_cards": [],
                    "profile": self.memory.get_profile(session_id),
                    "multimodal_summary": audio_result.get("summary"),
                    "audio_transcript": audio_transcript,
                }

        cleaned_message = self._sanitize_user_message(effective_message)

        if multimodal and image_path:
            multimodal_summary = self.multimodal.describe_image(image_path)
            answer_parts = []
            if audio_transcript:
                answer_parts.append(f"语音识别结果：{audio_transcript}")
            answer_parts.append(multimodal_summary)
            answer = "\n\n".join(part for part in answer_parts if part)
            self._save_turn(session_id, effective_message or message, answer)
            return {
                "session_id": session_id,
                "answer": answer,
                "sources": ["multimodal_service"],
                "intent": "multimodal",
                "action_taken": "multimodal_describe",
                "need_more_info": False,
                "plan_cards": [],
                "profile": self.memory.get_profile(session_id),
                "multimodal_summary": multimodal_summary,
                "audio_transcript": audio_transcript,
            }

        resolved_intent = self._resolve_user_intent(cleaned_message, history, profile)
        self._log_pipeline_step(session_id, "intent_identified", {"intent": resolved_intent.get("intent"), "confidence": resolved_intent.get("confidence")})
        current_message_facts = self._extract_current_message_trip_facts(cleaned_message, profile, history, resolved_intent)
        self._log_pipeline_step(session_id, "extracted_facts", current_message_facts)
        updated_profile = self._infer_user_facts(session_id, cleaned_message, history, profile, resolved_intent)
        self._log_trip_state_transition(session_id, profile, current_message_facts, updated_profile)
        self._log_pipeline_step(session_id, "updated_state", updated_profile)
        updated_profile = self.memory.update_profile(session_id, **updated_profile)
        self._update_user_profile_memory(session_id, cleaned_message, updated_profile)
        long_memory = self.memory.read(session_id).long_memory
        rewritten_message = self._rewrite_question_with_state(cleaned_message, updated_profile, resolved_intent)
        self._log_pipeline_step(session_id, "rewritten_question", {"message": rewritten_message})
        resolved_intent = self._resolve_user_intent(rewritten_message, history, updated_profile)
        self._log_pipeline_step(session_id, "resolved_intent_after_rewrite", {"intent": resolved_intent.get("intent"), "confidence": resolved_intent.get("confidence")})
        updated_profile = self.memory.update_profile(session_id, **resolved_intent.get("profile_updates", {}))
        response_profile = self._build_response_profile(rewritten_message, history, updated_profile, current_message_facts)
        self._log_pipeline_step(session_id, "response_profile", response_profile)

        plan = self._build_workflow_plan(rewritten_message, history, response_profile, resolved_intent)
        self._log_pipeline_step(session_id, "tool_plan", plan or {})
        tool_result = self._execute_tool_calls(plan, response_profile)
        self._log_pipeline_step(session_id, "api_results", tool_result)
        response_profile = self._merge_tool_confirmed_facts(response_profile, tool_result)
        self._log_pipeline_step(session_id, "state_after_api_merge", response_profile)

        if self.llm.available() and plan:
            self._log_llm_state("before_compose_answer", session_id, rewritten_message, response_profile, long_memory)
            final_result = self._compose_answer_with_llm(
                message=rewritten_message,
                history=history,
                profile=response_profile,
                plan=plan,
                tool_result=tool_result,
                long_memory=long_memory,
            )
            self._log_llm_result("after_compose_answer", session_id, final_result, response_profile)
            raw_answer = final_result.get("answer") or self._fallback_response(rewritten_message, history, response_profile, resolved_intent)
            answer = self._reflect_and_rewrite_answer(
                question=rewritten_message,
                answer=raw_answer,
                profile=response_profile,
                tool_result=tool_result,
            )
            intent = str(plan.get("task") or resolved_intent.get("intent") or "agent")
            answer, inferred_need_more = self._ensure_missing_info_closing(
                answer,
                response_profile,
                intent,
            )
            plan_cards = final_result.get("plan_cards") or []
            need_more_info = bool(final_result.get("need_more_info")) or inferred_need_more
            action_taken = str(final_result.get("action_taken") or "llm_agent_reply")
            if inferred_need_more and action_taken == "llm_agent_reply":
                action_taken = "llm_agent_clarify"
            sources = tool_result.get("sources", [])
            followup_updates = self._derive_followup_updates(intent, answer, need_more_info)
            self.memory.update_profile(session_id, last_intent=intent, **followup_updates)
            self._save_turn(session_id, message, answer)
            return {
                "session_id": session_id,
                "answer": answer,
                "sources": sources,
                "intent": intent,
                "action_taken": action_taken,
                "need_more_info": need_more_info,
                "plan_cards": plan_cards,
                "profile": self.memory.get_profile(session_id),
                "multimodal_summary": multimodal_summary,
            }

        fallback_intent = str(plan.get("task") or resolved_intent.get("intent") or "fallback")
        dynamic_result = self._build_dynamic_compose_fallback(
            message=rewritten_message,
            profile=response_profile,
            plan=plan,
            tool_result=tool_result,
        )
        answer = str(dynamic_result.get("answer") or "").strip()
        if not answer:
            answer = self._fallback_response(rewritten_message, history, response_profile, resolved_intent)
        answer, inferred_need_more = self._ensure_missing_info_closing(answer, response_profile, fallback_intent)
        need_more_info = bool(dynamic_result.get("need_more_info", False)) or inferred_need_more
        action_taken = str(dynamic_result.get("action_taken") or "rule_fallback")
        plan_cards = dynamic_result.get("plan_cards") or []
        followup_updates = self._derive_followup_updates(fallback_intent, answer, need_more_info)
        self.memory.update_profile(session_id, last_intent=fallback_intent, **followup_updates)
        self._save_turn(session_id, message, answer)
        return {
            "session_id": session_id,
            "answer": answer,
            "sources": tool_result.get("sources", []),
            "intent": fallback_intent,
            "action_taken": action_taken,
            "need_more_info": need_more_info,
            "plan_cards": plan_cards,
            "profile": self.memory.get_profile(session_id),
            "multimodal_summary": multimodal_summary,
        }

    def _plan_with_llm(
        self,
        message: str,
        history: list[tuple[str, str]],
        profile: dict[str, Any],
        resolved_intent: dict[str, Any],
        long_memory: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        workflow_plan = self._build_workflow_plan(message, history, profile, resolved_intent)
        if workflow_plan:
            return workflow_plan

        history_text = self._serialize_history(history)
        profile_text = json.dumps(profile, ensure_ascii=False)
        long_memory_text = json.dumps(long_memory or {}, ensure_ascii=False)
        intent_text = json.dumps(resolved_intent, ensure_ascii=False)
        planner_messages = [
            {
                "role": "system",
                "content": (
                    "你是旅游规划 Agent 的任务规划器。"
                    "你的职责是先理解用户需求，再决定是否需要调用工具。"
                    "不要直接输出自然语言答案，只能输出 JSON。\n"
                    "可用工具如下：\n"
                    "1. route_plan: 查询两地驾车距离、时长。参数：origin, destination, mode(默认driving)\n"
                    "2. weather: 查询城市天气。参数：city\n"
                    "3. hotel_search: 查询酒店候选。参数：city, budget\n"
                    "4. place_search: 查询景点/地点候选。参数：keyword, city\n"
                    "5. web_search: 做泛化搜索。参数：query\n"
                    "6. xiaohongshu_search: 查询小红书旅行经验笔记。参数：keyword, page(默认1), sort_type(默认general), note_type(默认ALL), time_filter(默认ALL), limit(默认5)\n"
                    "输出 JSON 格式：\n"
                    "{\n"
                    '  "task": "transport_advice|trip_plan|detailed_trip_plan|route|weather|hotel|place_search|restaurant_recommendation|search|chat",\n'
                    '  "need_more_info": true/false,\n'
                    '  "missing_info": ["缺少的关键信息"],\n'
                    '  "reasoning_brief": "一句话说明为什么这样规划",\n'
                    '  "tool_calls": [{"tool": "weather", "args": {"city": "杭州"}}],\n'
                    '  "response_goal": "最终回答要覆盖哪些点"\n'
                    "}\n"
                    "要求：\n"
                    "- 旅游规划问题优先按工作流思维组织：路线、天气、酒店、小红书经验四部分能查就查；当前会话中已经确认并存入状态的目的地、出发地、天数、预算、人数、偏好、日期等字段，都可以作为当前有效事实参与规划与回答。\n"
                    "- 只有当用户在当前轮明确修改了某个字段时，才用新值覆盖旧值；当前轮未提到的字段，默认沿用会话里已确认的旧值。只有目的地明确变更时，才把其他强相关字段视为需要重新确认。\n"
                    "- 如果信息不足，不要卡住；先回答当前可确定的方案，再指出缺什么信息能让建议更精确。\n"
                    "- 如果任务是 trip_plan 或 detailed_trip_plan，优先组合 weather、hotel_search、xiaohongshu_search；如已知出发地和目的地，再补 route_plan；再按偏好补 1-2 个 place_search。\n"
                    "- 如果用户问今天天气、周边吃什么、附近有什么好吃的，也可以调用 weather 或 place_search 回答。\n"
                    "- 不要虚构不存在的工具。"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"用户消息：{message}\n\n"
                    f"历史对话：\n{history_text}\n\n"
                    f"当前状态：{profile_text}\n\n"
                    f"用户画像记忆：{long_memory_text}\n\n"
                    f"预解析意图：{intent_text}"
                ),
            },
        ]
        plan = self.llm.extract_json(planner_messages, temperature=0.0)
        if not isinstance(plan, dict):
            return {}
        tool_calls = plan.get("tool_calls")
        if not isinstance(tool_calls, list):
            plan["tool_calls"] = []
        return plan

    def _execute_tool_calls(self, plan: dict[str, Any], profile: dict[str, Any]) -> dict[str, Any]:
        tool_calls = plan.get("tool_calls") or []
        results: list[dict[str, Any]] = []
        sources: list[str] = []
        if not tool_calls:
            return {"results": [], "sources": [], "message": "no_tool_calls"}
        for item in tool_calls[:6]:
            if not isinstance(item, dict):
                continue
            tool_name = str(item.get("tool") or "").strip()
            args = item.get("args") or {}
            if not isinstance(args, dict):
                args = {}
            executed = self._run_tool(tool_name, args, profile)
            if executed is None:
                results.append({"tool": tool_name, "args": args, "result": None, "status": "skipped"})
                continue
            results.append(executed)
            source = executed.get("source")
            if source and source not in sources:
                sources.append(source)
        return {"results": results, "sources": sources}

    def _run_tool(self, tool_name: str, args: dict[str, Any], profile: dict[str, Any]) -> dict[str, Any] | None:
        try:
            if tool_name == "weather":
                city = str(args.get("city") or profile.get("destination") or settings.default_city)
                day_offset = args.get("day_offset", profile.get("departure_day_offset") or 0)
                date_label = str(args.get("date_label") or profile.get("departure_date_label") or "")
                result = self.weather.get_weather(city, day_offset=int(day_offset or 0), date_label=date_label)
                return {
                    "tool": tool_name,
                    "source": "weather_tool",
                    "args": {"city": city, "day_offset": int(day_offset or 0), "date_label": date_label},
                    "result": result,
                    "planning_hints": self._build_weather_hints(result),
                }
            if tool_name == "hotel_search":
                city = str(args.get("city") or profile.get("destination") or settings.default_city)
                budget = args.get("budget", profile.get("budget"))
                effective_budget = budget if isinstance(budget, int) else self._hotel_search_budget_cap(
                    profile.get("budget"),
                    profile.get("days"),
                    profile.get("group_size"),
                )
                hotels = self.ctrip.search_hotels(city, budget=effective_budget if isinstance(effective_budget, int) else None)
                hotel_dicts = [hotel.__dict__ for hotel in hotels]
                return {
                    "tool": tool_name,
                    "source": "hotel_search_tool",
                    "args": {"city": city, "budget": effective_budget},
                    "result": hotel_dicts,
                    "planning_hints": self._build_hotel_hints(city, hotel_dicts, effective_budget),
                }
            if tool_name == "route_plan":
                origin = str(args.get("origin") or profile.get("origin") or "")
                destination = str(args.get("destination") or profile.get("destination") or "")
                mode = str(args.get("mode") or "driving")
                if not origin or not destination:
                    return None
                route_result = self.tencent_map.route_plan(origin, destination, mode=mode)
                return {
                    "tool": tool_name,
                    "source": "tencent_map_tool",
                    "args": {"origin": origin, "destination": destination, "mode": mode},
                    "result": route_result,
                    "planning_hints": self._build_route_hints(route_result, profile),
                }
            if tool_name == "place_search":
                city = str(args.get("city") or profile.get("destination") or settings.default_city)
                keyword = str(args.get("keyword") or "景点")
                result = self.tencent_map.search_place(keyword, city)
                return {
                    "tool": tool_name,
                    "source": "tencent_map_tool",
                    "args": {"keyword": keyword, "city": city},
                    "result": result,
                    "planning_hints": self._build_place_hints(city, keyword, result),
                }
            if tool_name == "web_search":
                query = str(args.get("query") or "").strip()
                if not query:
                    return None
                return {
                    "tool": tool_name,
                    "source": "search_tool",
                    "args": {"query": query},
                    "result": self.search.search(query),
                }
            if tool_name == "xiaohongshu_search":
                keyword = str(
                    args.get("keyword")
                    or self._build_xiaohongshu_keyword(
                        profile=profile,
                        fallback_message=str(args.get("query") or args.get("topic") or ""),
                    )
                ).strip()
                if not keyword:
                    return None
                page = int(args.get("page") or 1)
                sort_type = str(args.get("sort_type") or "general")
                note_type = str(args.get("note_type") or "ALL")
                time_filter = str(args.get("time_filter") or "ALL")
                limit = int(args.get("limit") or 5)
                result = self.xiaohongshu.search_notes(
                    keyword=keyword,
                    page=page,
                    sort_type=sort_type,
                    note_type=note_type,
                    time_filter=time_filter,
                    limit=limit,
                )
                return {
                    "tool": tool_name,
                    "source": "xiaohongshu_tool",
                    "args": {
                        "keyword": keyword,
                        "page": page,
                        "sort_type": sort_type,
                        "note_type": note_type,
                        "time_filter": time_filter,
                        "limit": limit,
                    },
                    "result": result,
                    "planning_hints": self._build_xiaohongshu_hints(result),
                }
        except Exception as exc:  # noqa: BLE001
            return {
                "tool": tool_name,
                "source": f"{tool_name}_error",
                "args": args,
                "result": {"error": str(exc)},
            }
        return None

    def _compose_answer_with_llm(
        self,
        message: str,
        history: list[tuple[str, str]],
        profile: dict[str, Any],
        plan: dict[str, Any],
        tool_result: dict[str, Any],
        long_memory: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        history_text = self._serialize_history(history)
        profile_text = json.dumps(profile, ensure_ascii=False)
        long_memory_text = json.dumps(long_memory or {}, ensure_ascii=False)
        plan_text = json.dumps(plan, ensure_ascii=False)
        tool_results = tool_result.get("results", [])
        tool_text = json.dumps(tool_results, ensure_ascii=False)
        trip_blueprint = self._build_trip_blueprint(profile, tool_results) if str(plan.get("task") or "") in {"trip_plan", "detailed_trip_plan", "transport_advice"} else {}
        orchestration_hints = self._build_orchestration_hints(tool_results, profile, trip_blueprint)
        composer_messages = [
            {
                "role": "system",
                "content": (
                    "你是一个中文旅游规划助手。你需要基于用户问题、历史上下文、用户画像和工具结果，自主组织最终回答。"
                    "你可以做比较、规划、建议、提醒，也可以在信息不足时明确指出还缺什么。"
                    "不要暴露内部规划过程，不要提到你看到了 JSON 或工具编排。"
                    "如果工具结果不足，就基于已有事实谨慎回答，不要编造精确数据。\n"
                    "这是一个旅游规划 agent，优先按照工作流整合答案：路线、天气、酒店、小红书经验，再结合预算、人数、偏好做结论。\n"
                    "如果拿到了小红书笔记结果，要优先吸收其中的玩法、避坑点、住宿区域、美食建议和游玩节奏，再转化为更接地气的旅行建议。\n"
                    "如果同时拿到了天气、景点、酒店结果，要把它们联合编排：天气决定游玩节奏，景点决定白天动线，酒店决定住宿落点，小红书决定体验细节和避坑提醒。\n"
                    "可以适度提炼‘很多笔记提到’‘常见建议是’这类总结，但不要把小红书原文大段照搬。\n"
                    "如果适合生成完整行程，优先输出早/午/晚节奏、住宿区域建议、交通方式建议和避坑提醒。\n"
                    "如果用户明确要求详细行程，你必须尽量给出具体景点/馆名、推荐美食街区或餐厅类型、建议入住酒店名称或酒店类型+商圈、每天大致时间段（如 09:00-11:30 / 14:00-17:00 / 晚上）以及每天的交通衔接。\n"
                    "如果已经拿到酒店候选和地点候选，不要只给笼统区域，优先点名 2-3 个具体候选并说明适合原因。\n"
                    "如果拿到了路线工具结果，只要这些信息已经在当前会话状态中确认过，或者来自当前问题与工具结果，就可以结合距离、人数、预算、天数给结论；如果出发地未知，就必须先说明结论依赖出发地和门到门总耗时。\n"
                    "如果拿到了天气结果，要把日期和天气绑定说明，例如今天/明天/出发当天更适合安排室内还是户外。\n"
                    "如果信息还不完整，不要只追问；先给当前能确定的方案，再单独说明补充哪些信息后可以进一步细化。\n"
                    "如果出发地、人数或出行日期任一缺失，回答末尾必须明确列出还缺哪些项，并说明补充后能把交通、天气、住宿和行程排得更准；不要只引导用户要‘更详细时间表’而不提缺失项。\n"
                    "如果用户问的是日常问题，比如今天天气怎么样、周围有什么好吃的，也要直接基于工具结果回答，不要强行套旅游行程模板。\n"
                    "餐馆推荐要优先使用地址字段，没有地址时再回退到目的地。\n"
                    "输出 JSON 格式：\n"
                    "{\n"
                    '  "answer": "给用户的最终中文回答",\n'
                    '  "need_more_info": true/false,\n'
                    '  "action_taken": "llm_agent_reply|llm_agent_clarify",\n'
                    '  "plan_cards": [{"title": "第1天", "subtitle": "主题", "details": ["事项1", "事项2"]}]\n'
                    "}\n"
                    "要求：\n"
                    "- 回答自然，不要模板腔。\n"
                    "- 如果用户在问高铁还是飞机、坐什么更合适这类二选一交通比较：必须直接比较总耗时、票价/综合成本、准点性、舒适度、到站/落地后的接驳效率，不要只给模糊结论。\n"
                    "- 如果用户在问怎么去且出发地未知，不要直接下交通方式结论；应先明确告诉用户‘现在还不能直接判断’，再说明需要看出发地、距离和门到门总耗时，随后给通用判断框架。优先给‘什么情况下高铁更合适、什么情况下飞机更合适’，不要把自驾写成主比较对象，除非用户主动问到自驾。\n"
                    "- 如果用户在问怎么去且出发地已知，先给结论，再给理由，再给适用场景；如果目的地是上海，要优先提到虹桥站、虹桥机场、浦东机场的进城便利度差异。\n"
                    "- 不要写‘更建议选高铁或飞机’这种等于没回答的问题句式；要么明确给条件判断，要么明确推荐其一。\n"
                    "- 如果用户在要行程，尽量按天组织，并保持内部逻辑一致。\n"
                    "- plan_cards 只有在确实形成按天行程时再返回，否则返回空数组。"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"用户消息：{message}\n\n"
                    f"历史对话：\n{history_text}\n\n"
                    f"当前状态：{profile_text}\n\n"
                    f"用户画像记忆：{long_memory_text}\n\n"
                    f"规划结果：{plan_text}\n\n"
                    f"工具结果：{tool_text}\n\n"
                    f"结构化规划草案：{json.dumps(trip_blueprint, ensure_ascii=False)}\n\n"
                    f"联合编排提示：\n{orchestration_hints}"
                ),
            },
        ]
        result = self.llm.extract_json(composer_messages, temperature=0.3)
        if not isinstance(result, dict):
            return self._build_dynamic_compose_fallback(message, profile, plan, tool_result)
        if not isinstance(result.get("plan_cards"), list):
            result["plan_cards"] = []
        answer = str(result.get("answer") or "").strip()
        if not answer:
            return self._build_dynamic_compose_fallback(message, profile, plan, tool_result)
        if "need_more_info" not in result:
            result["need_more_info"] = bool(plan.get("need_more_info"))
        if not str(result.get("action_taken") or "").strip():
            result["action_taken"] = "llm_agent_clarify" if result.get("need_more_info") else "llm_agent_reply"
        return result

    def _build_dynamic_compose_fallback(
        self,
        message: str,
        profile: dict[str, Any],
        plan: dict[str, Any],
        tool_result: dict[str, Any],
    ) -> dict[str, Any]:
        intent = str(plan.get("task") or "")
        if intent in {"trip_plan", "detailed_trip_plan"}:
            answer = self._build_dynamic_trip_plan_answer(profile, tool_result)
            return {
                "answer": answer,
                "need_more_info": bool(plan.get("need_more_info")),
                "action_taken": "llm_agent_clarify" if plan.get("need_more_info") else "llm_agent_reply",
                "plan_cards": [],
            }
        fallback_answer = self._fallback_response(message, [], profile, {"intent": intent})
        return {
            "answer": fallback_answer,
            "need_more_info": bool(plan.get("need_more_info")),
            "action_taken": "llm_agent_clarify" if plan.get("need_more_info") else "llm_agent_reply",
            "plan_cards": [],
        }

    def _infer_user_facts(
        self,
        session_id: str,
        message: str,
        history: list[tuple[str, str]],
        profile: dict[str, Any],
        resolved_intent: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return self._resolve_trip_state(session_id, message, history, profile, resolved_intent)

    # ===== 修改开始 =====
    def _build_response_profile(
        self,
        message: str,
        history: list[tuple[str, str]],
        updated_profile: dict[str, Any],
        current_message_facts: dict[str, Any],
    ) -> dict[str, Any]:
        del message, history
        response_profile = dict(updated_profile)
        current_destination = str(current_message_facts.get("destination") or "").strip()
        current_address = str(current_message_facts.get("address") or "").strip()
        if current_destination:
            response_profile["destination"] = current_destination
        if current_address:
            response_profile["address"] = current_address
        response_profile.pop("city", None)
        return response_profile

    def _merge_tool_confirmed_facts(self, profile: dict[str, Any], tool_result: dict[str, Any]) -> dict[str, Any]:
        merged = dict(profile)
        tool_results = tool_result.get("results") or []
        for item in tool_results:
            if not isinstance(item, dict):
                continue
            tool_name = str(item.get("tool") or "")
            args = item.get("args") or {}
            if tool_name == "route_plan":
                if args.get("origin") and not merged.get("origin"):
                    merged["origin"] = args.get("origin")
                if args.get("destination") and not merged.get("destination"):
                    merged["destination"] = args.get("destination")
        return merged
    # ===== 修改结束 =====

    def _resolve_destination(self, profile: dict[str, Any], message: str = "", history: list[tuple[str, str]] | None = None) -> str | None:
        history = history or []
        destination = self._extract_destination(message, history)
        if destination:
            return self._normalize_location_text(str(destination).strip()) or None

        existing = str(profile.get("destination") or "").strip()
        if existing:
            return self._normalize_location_text(existing) or None

        for role, content in reversed(history):
            if role != "user":
                continue
            history_destination = self._extract_destination(content, history)
            if history_destination:
                return self._normalize_location_text(str(history_destination).strip()) or None
            history_city = self._extract_city(content)
            if history_city and not any(keyword in content for keyword in ["怎么去", "怎么走", "推荐我怎么去", "建议我怎么去", "交通方式", "坐什么"]):
                return self._normalize_location_text(str(history_city).strip()) or None
        return None

    # ===== 修改开始 =====
    def _resolve_trip_state(
        self,
        session_id: str,
        message: str,
        history: list[tuple[str, str]],
        profile: dict[str, Any],
        resolved_intent: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        current_facts = self._extract_current_message_trip_facts(message, profile, history, resolved_intent)
        intent = str((resolved_intent or {}).get("intent") or "")
        explicit_destination = str(current_facts.get("destination") or "").strip()
        previous_destination = str(profile.get("destination") or "").strip()
        normalized_previous_destination = self._normalize_location_text(previous_destination) if previous_destination else ""
        normalized_explicit_destination = self._normalize_location_text(explicit_destination) if explicit_destination else ""
        normalized_destination = normalized_explicit_destination or normalized_previous_destination
        destination_changed = bool(
            normalized_explicit_destination
            and normalized_previous_destination
            and normalized_explicit_destination != normalized_previous_destination
        )

        if destination_changed:
            self.memory.clear_profile_fields(
                session_id,
                "origin",
                "days",
                "budget",
                "group_size",
                "preferences",
                "departure_day_offset",
                "departure_date_label",
                "awaiting_followup_action",
                "last_intent",
            )
            profile = self.memory.get_profile(session_id)

        resolved: dict[str, Any] = dict(profile)
        resolved.pop("city", None)

        if intent in {"trip_plan", "detailed_trip_plan", "transport_advice", "route"}:
            destination_value = normalized_destination or str(profile.get("destination") or "").strip()
            if destination_value:
                resolved["destination"] = destination_value

        for key in ("origin", "days", "budget", "group_size", "departure_day_offset", "departure_date_label"):
            value = current_facts.get(key)
            if value is not None:
                resolved[key] = value

        travel_date_range = current_facts.get("travel_date_range")
        if isinstance(travel_date_range, dict):
            start_date = str(travel_date_range.get("start_date") or "").strip()
            end_date = str(travel_date_range.get("end_date") or "").strip()
            if start_date:
                resolved["travel_start_date"] = start_date
            if end_date:
                resolved["travel_end_date"] = end_date
            if start_date or end_date:
                resolved["travel_date_range"] = {"start_date": start_date, "end_date": end_date}

        for key in ("address", "weather_time"):
            value = current_facts.get(key)
            if value is not None and value != "":
                resolved[key] = value

        preferences = current_facts.get("preferences")
        if preferences and intent in {"trip_plan", "detailed_trip_plan"}:
            resolved["preferences"] = preferences
        resolved["destination_changed"] = destination_changed
        return resolved
    # ===== 修改结束 =====

    # ===== 修改开始 =====
    def _extract_current_message_trip_facts(
        self,
        message: str,
        profile: dict[str, Any] | None = None,
        history: list[tuple[str, str]] | None = None,
        resolved_intent: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        profile = profile or {}
        history = history or []
        intent = str((resolved_intent or {}).get("intent") or "")
        origin, route_destination = self._extract_route_points(message, history)
        explicit_city = self._extract_city(message)
        address = self._sanitize_location_candidate(self._extract_location_address(message), field_type="address")
        destination = self._sanitize_location_candidate(
            route_destination or self._extract_destination(message, history) or explicit_city,
            field_type="destination",
        )
        origin = self._sanitize_location_candidate(origin, field_type="origin")

        facts: dict[str, Any] = {}

        if intent in {"trip_plan", "detailed_trip_plan"}:
            if origin is not None:
                facts["origin"] = origin
            if destination:
                facts["destination"] = destination
            travel_date_range = self._extract_travel_date_range(message)
            if travel_date_range:
                facts["travel_date_range"] = travel_date_range
            days = self._extract_days(message)
            if days is not None:
                facts["days"] = days
            budget = self._extract_budget(message)
            if budget is not None:
                facts["budget"] = budget
            group_size = self._extract_group_size(message)
            if group_size is not None:
                facts["group_size"] = group_size
            preferences = self._extract_preference_tags(message, [], profile, inherit_existing=False)
            if preferences:
                facts["preferences"] = preferences
            weather_time = self._extract_weather_time(message)
            if weather_time:
                facts["weather_time"] = weather_time
            departure_day_offset = self._extract_day_offset(message)
            if departure_day_offset is not None:
                facts["departure_day_offset"] = departure_day_offset
            departure_date_label = self._extract_date_label(message)
            if departure_date_label is not None:
                facts["departure_date_label"] = departure_date_label
            return facts

        if intent in {"transport_advice", "route"}:
            if origin is not None:
                facts["origin"] = origin
            if destination:
                facts["destination"] = destination
            group_size = self._extract_group_size(message)
            if group_size is not None:
                facts["group_size"] = group_size
            return facts

        if intent == "weather":
            if address:
                facts["address"] = address
            weather_time = self._extract_weather_time(message)
            if weather_time:
                facts["weather_time"] = weather_time
            departure_day_offset = self._extract_day_offset(message)
            if departure_day_offset is not None:
                facts["departure_day_offset"] = departure_day_offset
            departure_date_label = self._extract_date_label(message)
            if departure_date_label is not None:
                facts["departure_date_label"] = departure_date_label
            return facts

        if intent in {"hotel", "hotel_search", "place_search", "food_search", "restaurant_recommendation"}:
            if address:
                facts["address"] = address
            elif explicit_city:
                sanitized_city = self._sanitize_location_candidate(explicit_city, field_type="address")
                if sanitized_city:
                    facts["address"] = sanitized_city
            return facts

        if origin is not None:
            facts["origin"] = origin
        if destination:
            facts["destination"] = destination
        if address:
            facts["address"] = address
        return facts
    # ===== 修改结束 =====

    def _log_trip_state_transition(
        self,
        session_id: str,
        previous_profile: dict[str, Any],
        current_message_facts: dict[str, Any],
        merged_profile: dict[str, Any],
    ) -> None:
        logger.info(
            "trip_state_transition | session_id=%s | current_message_facts=%s | previous_profile=%s | merged_profile=%s",
            session_id,
            json.dumps(current_message_facts, ensure_ascii=False, sort_keys=True),
            json.dumps(previous_profile, ensure_ascii=False, sort_keys=True),
            json.dumps(merged_profile, ensure_ascii=False, sort_keys=True),
        )

    def _log_pipeline_step(self, session_id: str, step: str, payload: Any) -> None:
        if step == "api_results" and isinstance(payload, dict):
            formatted = self._format_tool_results_for_log(payload)
        elif isinstance(payload, dict):
            formatted = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        elif isinstance(payload, list):
            formatted = json.dumps(payload, ensure_ascii=False)
        else:
            formatted = str(payload)
        logger.info("pipeline_step | session_id=%s | step=%s | payload=%s", session_id, step, formatted)

    def _format_tool_results_for_log(self, payload: dict[str, Any]) -> str:
        grouped = self._group_tool_results_for_log(payload.get("results") or [])
        source_lines: list[str] = []
        for source_name, items in grouped.items():
            if not items:
                continue
            source_lines.append(f"{source_name}：" + "；".join(items))
        if not source_lines:
            source_lines.append("未调用到有效工具结果")
        raw_sources = payload.get("sources") or []
        suffix = f" | sources={','.join(str(item) for item in raw_sources if item)}" if raw_sources else ""
        return " | ".join(source_lines) + suffix

    def _group_tool_results_for_log(self, tool_results: list[dict[str, Any]]) -> dict[str, list[str]]:
        grouped: dict[str, list[str]] = {}
        source_name_map = {
            "tencent_map_tool": "腾讯地图api",
            "weather_tool": "天气api",
            "hotel_search_tool": "酒店查询api",
            "xiaohongshu_tool": "小红书api",
            "search_tool": "搜索api",
        }
        for item in tool_results:
            if not isinstance(item, dict):
                continue
            source = str(item.get("source") or "unknown")
            source_name = source_name_map.get(source, source)
            summary = self._summarize_single_tool_result_for_log(item)
            if not summary:
                continue
            grouped.setdefault(source_name, []).append(summary)
        return grouped

    def _summarize_single_tool_result_for_log(self, item: dict[str, Any]) -> str:
        tool = str(item.get("tool") or "")
        args = item.get("args") or {}
        result = item.get("result") or {}

        if tool == "weather":
            if isinstance(result, dict) and result.get("error"):
                city = str(args.get("city") or result.get("city") or "目的地")
                return f"天气查询：{city}，失败（{result.get('error')}）"
            city = str(args.get("city") or (result.get("city") if isinstance(result, dict) else "") or "目的地")
            forecast = str((result.get("forecast") if isinstance(result, dict) else "") or "天气未知")
            low = result.get("temperature_min") if isinstance(result, dict) else None
            high = result.get("temperature_max") if isinstance(result, dict) else None
            temp_text = f"{low}℃-{high}℃" if low is not None and high is not None else "温度待确认"
            return f"天气查询：{city}，{forecast}，{temp_text}"

        if tool == "route_plan":
            origin = str(args.get("origin") or (result.get("origin") if isinstance(result, dict) else "") or "起点")
            destination = str(args.get("destination") or (result.get("destination") if isinstance(result, dict) else "") or "终点")
            if isinstance(result, dict) and result.get("error"):
                return f"路线查询：{origin}→{destination}，失败（{result.get('error')}）"
            distance = result.get("distance_km") if isinstance(result, dict) else None
            duration = result.get("duration_minutes") if isinstance(result, dict) else None
            distance_text = f"{distance}公里" if distance is not None else "距离待确认"
            duration_text = f"{duration}分钟" if duration is not None else "耗时待确认"
            return f"路线查询：{origin}→{destination}，{distance_text}，{duration_text}"

        if tool == "place_search":
            keyword = str(args.get("keyword") or "地点")
            city = str(args.get("city") or "")
            if not isinstance(result, list) or not result:
                return f"地点查询：{city}{keyword}，无结果"
            names = [str(place.get("name") or "").strip() for place in result[:3] if isinstance(place, dict) and place.get("name")]
            if not names:
                return f"地点查询：{city}{keyword}，无有效结果"
            return f"地点查询：{city}{keyword}，命中" + "、".join(names)

        if tool == "hotel_search":
            city = str(args.get("city") or "")
            if not isinstance(result, list) or not result:
                return f"酒店查询：{city}，无结果"
            hotel_lines: list[str] = []
            for hotel in result[:3]:
                if not isinstance(hotel, dict):
                    continue
                name = str(hotel.get("name") or "酒店").strip()
                price = hotel.get("price")
                price_text = f"{price}元/晚" if price is not None else "价格待确认"
                hotel_lines.append(f"{name}({price_text})")
            return f"酒店查询：{city}，命中" + "、".join(hotel_lines) if hotel_lines else f"酒店查询：{city}，无有效结果"

        if tool == "xiaohongshu_search":
            keyword = str(args.get("keyword") or "")
            if isinstance(result, dict) and result.get("error"):
                return f"攻略查询：{keyword}，失败（{result.get('error')}）"
            notes = result.get("notes") if isinstance(result, dict) else []
            note_count = len(notes) if isinstance(notes, list) else 0
            return f"攻略查询：{keyword}，返回{note_count}条笔记"

        if tool == "web_search":
            query = str(args.get("query") or "")
            return f"搜索查询：{query}"

        return ""

    def _log_llm_state(
        self,
        stage: str,
        session_id: str,
        message: str,
        profile: dict[str, Any],
        long_memory: dict[str, Any] | None = None,
    ) -> None:
        logger.info(
            "llm_state | stage=%s | session_id=%s | message=%s | state=%s | long_memory=%s",
            stage,
            session_id,
            message,
            self._format_state_for_log(profile),
            json.dumps(long_memory or {}, ensure_ascii=False, sort_keys=True),
        )

    def _log_llm_result(
        self,
        stage: str,
        session_id: str,
        result: dict[str, Any],
        profile: dict[str, Any],
    ) -> None:
        logger.info(
            "llm_result | stage=%s | session_id=%s | result=%s | state=%s",
            stage,
            session_id,
            json.dumps(result, ensure_ascii=False, sort_keys=True),
            self._format_state_for_log(profile),
        )

    def _format_state_for_log(self, profile: dict[str, Any]) -> str:
        ordered_keys = [
            "origin",
            "destination",
            "group_size",
            "days",
            "travel_start_date",
            "travel_end_date",
            "travel_date_range",
            "departure_date_label",
            "departure_day_offset",
            "budget",
            "preferences",
            "address",
            "weather_time",
            "destination_changed",
            "last_intent",
            "awaiting_followup_action",
        ]
        ordered_state = {key: profile[key] for key in ordered_keys if key in profile and profile.get(key) not in (None, "", [])}
        return json.dumps(ordered_state, ensure_ascii=False)

    def _update_user_profile_memory(self, session_id: str, message: str, profile: dict[str, Any]) -> dict[str, Any]:
        current_long_memory = self.memory.read(session_id).long_memory or {}
        user_profile = dict(current_long_memory.get("user_profile") or {})

        preferences = profile.get("preferences") or []
        if isinstance(preferences, list) and preferences:
            user_profile["travel_preferences"] = preferences

        budget = profile.get("budget")
        if isinstance(budget, int) and budget > 0:
            user_profile["budget_level"] = self._infer_budget_level(budget)

        group_size = profile.get("group_size")
        if isinstance(group_size, int) and group_size > 0:
            user_profile["group_type"] = "solo" if group_size == 1 else ("pair" if group_size == 2 else "group")

        personality_tags = user_profile.get("personality_tags") or []
        inferred_tags = self._infer_personality_tags(message, preferences)
        merged_tags: list[str] = []
        for tag in [*(personality_tags if isinstance(personality_tags, list) else []), *inferred_tags]:
            if tag and tag not in merged_tags:
                merged_tags.append(tag)
        if merged_tags:
            user_profile["personality_tags"] = merged_tags

        return self.memory.update_long_memory(session_id, user_profile=user_profile)

    def _rewrite_question_with_state(self, message: str, profile: dict[str, Any], resolved_intent: dict[str, Any] | None = None) -> str:
        normalized_message = self._sanitize_user_message(message)
        intent = str((resolved_intent or {}).get("intent") or "")
        destination = str(profile.get("destination") or "").strip()
        origin = str(profile.get("origin") or "").strip()
        address = str(profile.get("address") or "").strip()
        days = profile.get("days")
        budget = profile.get("budget")
        group_size = profile.get("group_size")
        preferences = profile.get("preferences") or []
        departure_date_label = str(profile.get("departure_date_label") or "").strip()
        travel_start_date = str(profile.get("travel_start_date") or "").strip()
        travel_end_date = str(profile.get("travel_end_date") or "").strip()
        weather_time = str(profile.get("weather_time") or "").strip()

        state_parts: list[str] = []
        if intent in {"trip_plan", "detailed_trip_plan"}:
            if origin:
                state_parts.append(f"出发地={origin}")
            if destination:
                state_parts.append(f"目的地={destination}")
            if isinstance(group_size, int) and group_size > 0:
                state_parts.append(f"人数={group_size}")
            if isinstance(days, int) and days > 0:
                state_parts.append(f"天数={days}")
            if travel_start_date and travel_end_date:
                state_parts.append(f"日期={travel_start_date}到{travel_end_date}")
            elif departure_date_label:
                state_parts.append(f"日期={departure_date_label}")
            if isinstance(budget, int) and budget > 0:
                state_parts.append(f"预算={budget}")
            if isinstance(preferences, list) and preferences:
                state_parts.append("偏好=" + "、".join(str(item) for item in preferences if item))
        elif intent in {"transport_advice", "route"}:
            if origin:
                state_parts.append(f"出发地={origin}")
            if destination:
                state_parts.append(f"目的地={destination}")
            if isinstance(group_size, int) and group_size > 0:
                state_parts.append(f"人数={group_size}")
        elif intent == "weather":
            if address:
                state_parts.append(f"地址={address}")
            if weather_time:
                state_parts.append(f"天气时间={weather_time}")
            elif departure_date_label:
                state_parts.append(f"天气时间={departure_date_label}")
        elif intent in {"hotel", "hotel_search", "place_search", "food_search", "restaurant_recommendation"}:
            if address:
                state_parts.append(f"地址={address}")
        else:
            if destination:
                state_parts.append(f"目的地={destination}")
            if address:
                state_parts.append(f"地址={address}")

        if not state_parts:
            return normalized_message
        return f"{normalized_message}\n\n[当前状态]\n" + "；".join(state_parts)

    def _infer_budget_level(self, budget: int) -> str:
        if budget < 1000:
            return "low"
        if budget < 5000:
            return "medium"
        return "high"

    def _infer_personality_tags(self, message: str, preferences: list[Any]) -> list[str]:
        text = f"{message} {' '.join(str(item) for item in preferences if item)}"
        tags: list[str] = []
        mapping = {
            "休闲": ["轻松", "休闲", "慢游", "不赶"],
            "自然偏好": ["自然", "风景", "山水", "湿地"],
            "人文偏好": ["历史", "人文", "博物馆", "古建"],
            "美食偏好": ["美食", "吃", "餐厅", "小吃"],
            "拍照偏好": ["拍照", "出片", "机位"],
        }
        for tag, keywords in mapping.items():
            if any(keyword in text for keyword in keywords):
                tags.append(tag)
        return tags

    def _reflect_and_rewrite_answer(
        self,
        question: str,
        answer: str,
        profile: dict[str, Any],
        tool_result: dict[str, Any],
    ) -> str:
        if not answer or not self.llm.available():
            return answer

        allowed_facts = self._build_allowed_fact_summary(question, profile, tool_result)
        reflection_messages = [
            {
                "role": "system",
                "content": (
                    "你是旅游助手的回答质检器。你的任务是检查一段答案是否使用了当前会话中尚未确认、且也没有被工具结果直接确认的事实。"
                    "如果答案存在这类越界事实，你必须重写答案，删掉这些事实，或者改写成条件句。"
                    "允许保留当前会话状态里已经确认过的具体人数、预算、天数、偏好、出发日期、出发地、目的地等信息。"
                    "如果答案本身已经合规，就原样返回。"
                    "不要解释过程，只输出最终修正后的答案文本。"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"用户当前问题：{question}\n\n"
                    f"允许使用的事实：\n{allowed_facts}\n\n"
                    f"待检查答案：\n{answer}"
                ),
            },
        ]
        rewritten = self.llm.chat(reflection_messages, temperature=0.0).strip()
        return rewritten or answer

    def _build_allowed_fact_summary(
        self,
        question: str,
        profile: dict[str, Any],
        tool_result: dict[str, Any],
    ) -> str:
        lines = [f"- 用户当前问题原文：{question}"]

        current_fact_lines: list[str] = []
        fact_labels = {
            "origin": "出发地",
            "destination": "目的地",
            "days": "天数",
            "budget": "预算",
            "group_size": "人数",
            "departure_day_offset": "相对日期",
            "departure_date_label": "日期标签",
        }
        for key, label in fact_labels.items():
            value = profile.get(key)
            if value is None or value == "":
                continue
            current_fact_lines.append(f"- {label}：{value}")

        preferences = profile.get("preferences") or []
        if isinstance(preferences, list) and preferences:
            current_fact_lines.append("- 偏好：" + "、".join(str(item) for item in preferences if item))

        if current_fact_lines:
            lines.append("- 当前会话中已确认、当前回答可直接使用的事实：")
            lines.extend(current_fact_lines)
        else:
            lines.append("- 当前会话里尚未确认出发地、人数、预算、天数、偏好、出行日期等具体信息。")

        tool_lines: list[str] = []
        tool_results = tool_result.get("results") or []
        for item in tool_results:
            if not isinstance(item, dict):
                continue
            tool_name = str(item.get("tool") or "")
            args = item.get("args") or {}
            result = item.get("result") or {}
            if tool_name == "route_plan" and isinstance(result, dict):
                tool_lines.append(
                    f"- 路线工具确认：{result.get('origin') or args.get('origin') or ''} → {result.get('destination') or args.get('destination') or ''}"
                    f"，距离{result.get('distance_km')}公里，耗时{result.get('duration_minutes')}分钟"
                )
            elif tool_name == "weather" and isinstance(result, dict):
                tool_lines.append(
                    f"- 天气工具确认：{result.get('city') or args.get('city') or ''}，{result.get('forecast') or '天气待确认'}"
                )
            elif tool_name in {"hotel_search", "place_search", "xiaohongshu_search"}:
                tool_lines.append(f"- 工具{tool_name}已返回候选结果，可概括其结论，但不能虚构额外具体事实。")

        if tool_lines:
            lines.append("- 工具直接确认的事实：")
            lines.extend(tool_lines)

        lines.append("- 允许使用当前会话状态里已经确认过的字段；但不能虚构未确认的新事实，也不能把默认值写成用户已明确提供的信息。")
        return "\n".join(lines)

    def _fallback_response(
        self,
        message: str,
        history: list[tuple[str, str]],
        profile: dict[str, Any],
        resolved_intent: dict[str, Any],
    ) -> str:
        intent = str(resolved_intent.get("intent") or "")

        if intent == "detailed_trip_plan" and self._profile_ready_for_trip_plan(profile):
            return "我已经拿到这次详细行程需求了。你这次的目的地、日期和已有约束我都记住了，我可以继续结合交通、天气、住宿和景点结果，为你生成一版动态的详细行程安排。"

        if intent == "transport_advice" or self._looks_like_transport_advice_request(message):
            destination = self._resolve_destination(profile, message, history)
            if destination and not profile.get("destination"):
                profile = {**profile, "destination": destination}
            if self._profile_ready_for_transport_advice(profile):
                return self._build_transport_advice_fallback(profile, message=message)
            origin = profile.get("origin")
            missing = self._missing_transport_fields(profile)
            if origin and destination:
                route = self.tencent_map.route_plan(origin, destination, mode="driving")
                line = self._format_route_line(route) if not route.get("fallback") else f"{origin} → {destination}"
                if missing:
                    return f"我先识别到你的出行需求了：{line}。如果你再补充{''.join(missing)}，我就能更准确地帮你比较高铁、自驾和飞机。"
                return f"我先识别到你的出行需求了：{line}。如果你愿意，我可以继续根据人数、预算和行程重点帮你比较高铁、自驾和飞机。"
            destination = self._resolve_destination(profile, message, history)
            if origin and destination:
                route = self.tencent_map.route_plan(origin, destination, mode="driving")
                line = self._format_route_line(route) if not route.get("fallback") else f"{origin} → {destination}"
                return f"我先帮你识别到路线是：{line}。如果你愿意，我还能继续结合人数、预算和偏好，帮你比较高铁、自驾和飞机。"
            return "我可以帮你判断怎么去更合适。你可以补充出发地、目的地、人数、预算和大概玩几天。"

        if intent == "route" or self._looks_like_route_query(message):
            destination = self._resolve_destination(profile, message)
            if destination and not profile.get("destination"):
                profile = {**profile, "destination": destination}
            origin = profile.get("origin")
            if origin and destination:
                route = self.tencent_map.route_plan(origin, destination, mode="driving")
                line = self._format_route_line(route) if not route.get("fallback") else f"{origin} → {destination}"
                return f"我先帮你看了一下路线：{line}。如果你愿意，我也可以继续帮你比较高铁、自驾和飞机哪个更合适。"
            return "我可以先帮你查路线。你可以补充出发地和目的地。"

        if self._is_followup_confirmation(message, history) and self._profile_ready_for_trip_plan(profile):
            return "可以，我已经接住你的继续细化需求了。接下来我会基于当前状态和工具结果，继续整理成更具体的详细行程。"

        if intent == "weather" or self._looks_like_weather_query(message):
            city = profile.get("address") or self._extract_location_address(message) or profile.get("destination") or self._extract_city(message) or settings.default_city
            day_offset = int(profile.get("departure_day_offset") or self._extract_day_offset(message) or 0)
            date_label = str(profile.get("weather_time") or profile.get("departure_date_label") or self._extract_date_label(message) or "")
            result = self.weather.get_weather(str(city), day_offset=day_offset, date_label=date_label)
            if result.get("fallback"):
                return f"我可以帮你查天气。目前先识别到你关注的是{city}，如果你愿意补充具体出行日期，我可以把天气和行程节奏一起细化。"
            forecast = result.get("forecast") or "天气待确认"
            temp_min = result.get("temperature_min")
            temp_max = result.get("temperature_max")
            temp_text = f"{temp_min}~{temp_max}°C" if temp_min is not None and temp_max is not None else "温度待确认"
            recommendation = result.get("recommendation") or ""
            label_text = result.get("date_label") or "今天"
            return f"{city}{label_text}参考天气是{forecast}，{temp_text}。{recommendation}"

        if intent in {"food_search", "restaurant_recommendation"} or self._looks_like_food_query(message):
            address = self._extract_location_address(message) or profile.get("address") or self._extract_city(message) or profile.get("destination") or settings.default_city
            food_places = self.tencent_map.search_place("美食", str(address))
            if food_places:
                top_foods = "、".join(str(item.get("name") or "") for item in food_places[:4] if item.get("name"))
                return f"如果你想在{address}找吃的，可以先从这些地方看起：{top_foods}。如果你告诉我是想吃火锅、小吃、川菜还是夜宵，我可以继续缩小范围。"
            return f"我可以继续帮你找{address}周边好吃的。你如果补充想吃正餐、小吃、火锅还是夜宵，我能推荐得更准。"

        if intent == "trip_plan" or self._looks_like_trip_plan_request(message):
            if self._profile_ready_for_trip_plan(profile):
                if intent == "detailed_trip_plan" or self._wants_detailed_plan(message, history):
                    return "我已经拿到你的详细行程需求了，会优先围绕交通、天气、住宿、景点和每日节奏来组织一版更细的动态安排。"
                return "我已经拿到你的行程规划需求了，会优先围绕怎么去、住哪里、去哪玩、怎么玩和注意事项来整理一版动态建议。"
            city = profile.get("destination") or settings.default_city
            days = profile.get("days") or settings.default_trip_days
            missing = self._missing_trip_fields(profile)
            if missing:
                missing_text = "、".join(missing)
                return (
                    f"我可以先按现有信息给你做一版 {city}{days} 日的基础玩法思路，"
                    f"并优先围绕‘怎么去、住哪里、去哪玩、怎么玩、注意什么’来整理；"
                    f"目前还缺 {missing_text}，如果你给我更多信息，我会把路线、天气、酒店和攻略安排得更准确、更贴合你这次出行。"
                )
            return f"我可以继续帮你做 {city}{days} 日行程。你如果再告诉我预算、同行人数、日期和偏好，我能把路线、天气、酒店和攻略一起排得更贴合。"

        if history and self._profile_ready_for_trip_plan(profile):
            return "我已经记住当前这次旅行的核心条件了。你可以继续补充日期、人数、预算、偏好或具体诉求，我会基于当前状态继续动态完善建议。"
        if history:
            return "我已经接住你的需求了。你可以继续补充目的地、天数、预算、人数或偏好，我会接着完善建议。"
        return "我是你的旅游规划助手，可以帮你比较交通方式、规划行程、查天气、找酒店，也可以结合上下文给出更完整的旅行建议。"

    def _looks_like_route_or_transport_request(self, message: str) -> bool:
        keywords = ["怎么去", "怎么走", "高铁", "自驾", "飞机", "路线", "路程", "车程", "交通方式", "建议怎么去"]
        return any(keyword in message for keyword in keywords)

    def _looks_like_transport_advice_request(self, message: str) -> bool:
        text = (message or "").strip()
        decision_keywords = ["建议怎么去", "怎么去更合适", "怎么去最好", "高铁还是", "自驾还是", "飞机还是", "交通方式", "建议坐什么"]
        if any(keyword in text for keyword in decision_keywords):
            return True
        has_transport = any(keyword in text for keyword in ["怎么去", "高铁", "自驾", "飞机", "交通"])
        has_constraints = any(keyword in text for keyword in ["预算", "天", "日", "人", "喜欢", "想看", "偏好", "计划"])
        return has_transport and has_constraints

    def _is_highspeed_vs_flight_question(self, message: str) -> bool:
        text = (message or "").strip()
        return (
            ("高铁" in text and "飞机" in text)
            or "高铁还是飞机" in text
            or "飞机还是高铁" in text
            or "坐高铁还是飞机" in text
            or "坐飞机还是高铁" in text
        )

    def _looks_like_route_query(self, message: str) -> bool:
        text = (message or "").strip()
        route_keywords = ["路线", "路程", "车程", "多少公里", "多久", "多远", "怎么走"]
        return any(keyword in text for keyword in route_keywords)

    def _looks_like_weather_query(self, message: str) -> bool:
        text = (message or "").strip()
        return any(keyword in text for keyword in ["天气", "温度", "下雨", "会不会下雨", "今天天气", "明天天气"])

    def _looks_like_food_query(self, message: str) -> bool:
        text = (message or "").strip()
        return any(keyword in text for keyword in ["好吃的", "吃什么", "美食", "餐厅", "小吃", "附近有什么吃的", "周围有什么吃的"])

    def _profile_ready_for_trip_plan(self, profile: dict[str, Any]) -> bool:
        city = profile.get("destination")
        days = profile.get("days")
        return bool(city and isinstance(days, int) and days > 0)

    def _profile_ready_for_transport_advice(self, profile: dict[str, Any]) -> bool:
        origin = profile.get("origin")
        destination = profile.get("destination")
        days = profile.get("days")
        budget = profile.get("budget")
        group_size = profile.get("group_size")
        return bool(origin and destination and isinstance(days, int) and days > 0 and isinstance(budget, int) and budget > 0 and isinstance(group_size, int) and group_size > 0)

    def _missing_transport_fields(self, profile: dict[str, Any]) -> list[str]:
        missing: list[str] = []
        if not profile.get("origin"):
            missing.append("出发地")
        if not profile.get("destination"):
            missing.append("目的地")
        if not (isinstance(profile.get("days"), int) and profile.get("days") > 0):
            missing.append("天数")
        if not (isinstance(profile.get("budget"), int) and profile.get("budget") > 0):
            missing.append("预算")
        if not (isinstance(profile.get("group_size"), int) and profile.get("group_size") > 0):
            missing.append("人数")
        if not str(profile.get("departure_date_label") or "").strip() and not isinstance(profile.get("departure_day_offset"), int):
            missing.append("日期")
        preferences = profile.get("preferences") or []
        if not isinstance(preferences, list) or not any(str(item).strip() for item in preferences):
            missing.append("偏好")
        return missing

    def _is_followup_confirmation(self, message: str, history: list[tuple[str, str]]) -> bool:
        normalized = (message or "").strip().lower()
        confirmation_words = {"可以", "好的", "好", "行", "继续", "那就这样", "来吧", "安排", "嗯", "恩", "展开", "具体一点"}
        if normalized not in confirmation_words:
            return False
        for role, content in reversed(history[-4:]):
            if role != "assistant":
                continue
            if self._assistant_offered_next_step(content):
                return True
        return False

    def _wants_detailed_plan(self, message: str, history: list[tuple[str, str]]) -> bool:
        text = (message or "").strip()
        detail_keywords = ["详细", "具体", "细化", "详细行程", "时间表", "按天", "早中晚", "展开"]
        if any(keyword in text for keyword in detail_keywords):
            return True
        return self._is_followup_confirmation(message, history)

    def _assistant_offered_next_step(self, content: str) -> bool:
        text = self._sanitize_user_message(content)
        offer_keywords = [
            "如果你愿意",
            "下一步可以",
            "我可以继续",
            "给你排一版",
            "详细行程",
            "详细时间表",
            "更细",
            "展开成一版",
        ]
        return any(keyword in text for keyword in offer_keywords)

    def _resolve_user_intent(self, message: str, history: list[tuple[str, str]], profile: dict[str, Any]) -> dict[str, Any]:
        normalized = (message or "").strip()
        profile_updates: dict[str, Any] = {}
        last_requested_action = str(profile.get("awaiting_followup_action") or "").strip()
        has_trip_context = bool(profile.get("destination"))
        has_origin = bool(str(profile.get("origin") or "").strip())

        if self._is_followup_confirmation(message, history):
            if last_requested_action:
                return {"intent": last_requested_action, "confidence": "high", "profile_updates": profile_updates}
            if self._profile_ready_for_trip_plan(profile):
                return {"intent": "detailed_trip_plan", "confidence": "medium", "profile_updates": profile_updates}

        if self._wants_detailed_plan(message, history):
            profile_updates["awaiting_followup_action"] = ""
            return {"intent": "detailed_trip_plan", "confidence": "high", "profile_updates": profile_updates}

        if has_trip_context and (self._looks_like_transport_advice_request(message) or (has_origin and any(keyword in normalized for keyword in ["怎么去", "怎么走", "推荐我怎么去", "建议我怎么去", "坐什么", "交通方式"]))):
            profile_updates["awaiting_followup_action"] = ""
            return {"intent": "transport_advice", "confidence": "high", "profile_updates": profile_updates}

        if has_trip_context and self._looks_like_route_query(message):
            profile_updates["awaiting_followup_action"] = ""
            return {"intent": "route", "confidence": "high", "profile_updates": profile_updates}

        if self._looks_like_weather_query(message):
            profile_updates["awaiting_followup_action"] = ""
            return {"intent": "weather", "confidence": "high", "profile_updates": profile_updates}

        if self._looks_like_food_query(message):
            profile_updates["awaiting_followup_action"] = ""
            return {"intent": "restaurant_recommendation", "confidence": "high", "profile_updates": profile_updates}

        if self._looks_like_trip_plan_request(message):
            profile_updates["awaiting_followup_action"] = ""
            return {"intent": "trip_plan", "confidence": "high", "profile_updates": profile_updates}

        if has_trip_context and has_origin:
            return {"intent": "transport_advice", "confidence": "medium", "profile_updates": profile_updates}

        if normalized in {"可以", "好的", "好", "行", "继续", "安排", "展开"} and last_requested_action:
            return {"intent": last_requested_action, "confidence": "medium", "profile_updates": profile_updates}

        if self._profile_ready_for_trip_plan(profile):
            return {"intent": "trip_plan", "confidence": "low", "profile_updates": profile_updates}
        return {"intent": "chat", "confidence": "low", "profile_updates": profile_updates}

    def _derive_followup_updates(self, intent: str, answer: str, need_more_info: bool) -> dict[str, Any]:
        if need_more_info:
            return {"awaiting_followup_action": ""}
        if intent == "transport_advice" and self._assistant_offered_next_step(answer):
            return {"awaiting_followup_action": "detailed_trip_plan"}
        if intent in {"trip_plan", "detailed_trip_plan"} and self._assistant_offered_next_step(answer):
            return {"awaiting_followup_action": "detailed_trip_plan"}
        return {"awaiting_followup_action": ""}

    def _missing_trip_fields(self, profile: dict[str, Any]) -> list[str]:
        missing: list[str] = []
        if not profile.get("destination"):
            missing.append("目的地")
        if not str(profile.get("origin") or "").strip():
            missing.append("出发地")
        if not (isinstance(profile.get("days"), int) and profile.get("days") > 0):
            missing.append("天数")
        if not (isinstance(profile.get("budget"), int) and profile.get("budget") > 0):
            missing.append("预算")
        if not (isinstance(profile.get("group_size"), int) and profile.get("group_size") > 0):
            missing.append("人数")
        if not str(profile.get("departure_date_label") or "").strip() and not isinstance(profile.get("departure_day_offset"), int):
            missing.append("出行日期")
        preferences = profile.get("preferences") or []
        if not isinstance(preferences, list) or not any(str(item).strip() for item in preferences):
            missing.append("偏好")
        return missing

    def _build_missing_info_closing(self, profile: dict[str, Any]) -> str:
        missing = self._missing_trip_fields(profile)
        if not missing:
            return ""
        missing_text = "、".join(missing)
        return (
            f"目前还缺{missing_text}等信息。"
            f"如果你补充这些，我会把交通方式、天气、住宿和每日安排安排得更准确、更贴合你这次出行。"
        )

    def _build_dynamic_trip_plan_answer(self, profile: dict[str, Any], tool_result: dict[str, Any]) -> str:
        city = str(profile.get("destination") or settings.default_city)
        origin = str(profile.get("origin") or "").strip()
        days = int(profile.get("days") or settings.default_trip_days)
        budget = profile.get("budget")
        preferences = profile.get("preferences") or []
        travel_start_date = str(profile.get("travel_start_date") or "").strip()
        travel_end_date = str(profile.get("travel_end_date") or "").strip()
        nights = max(days - 1, 1)

        tool_results = tool_result.get("results") or []
        blueprint = self._build_trip_blueprint(profile, tool_results)
        weather_summary = blueprint.get("weather_summary") or self._summarize_weather_result(tool_results)
        hotel_summary = blueprint.get("hotel_summary") or self._summarize_hotel_result(tool_results)
        food_summary = blueprint.get("food_summary") or self._summarize_food_result(tool_results)
        area_summary = blueprint.get("area_summary") or ""
        place_summary = blueprint.get("place_summary") or self._summarize_place_result(tool_results)
        xhs_summary = blueprint.get("xhs_summary") or self._summarize_xiaohongshu_result(tool_results)
        route_summary = blueprint.get("route_summary") or self._summarize_route_result(tool_results)
        daily_plan = blueprint.get("daily_plan") or []

        intro_parts = [f"可以，我先按你当前这组信息，给你整理一版{city}{days}天{nights}晚的详细玩法。"]
        if origin:
            intro_parts.append(f"这次先按从{origin}前往{city}来组织。")
        if travel_start_date and travel_end_date:
            intro_parts.append(f"日期按{travel_start_date}到{travel_end_date}理解。")
        if isinstance(budget, int) and budget > 0:
            intro_parts.append(f"总预算参考{budget}元左右。")
        if isinstance(preferences, list) and preferences:
            intro_parts.append("偏好重点放在" + "、".join(str(item) for item in preferences if item) + "。")

        day_cards = self._render_daily_plan_cards(daily_plan)
        if not day_cards:
            fallback_place_groups = blueprint.get("place_groups") or self._collect_place_groups(tool_results)
            if fallback_place_groups:
                day_cards = self._build_daily_dayparts(
                    city,
                    days,
                    fallback_place_groups,
                    blueprint.get("hotel_candidates") or self._collect_hotel_candidates(tool_results),
                    weather_summary,
                    xhs_summary,
                )

        sections = ["".join(intro_parts)]
        if route_summary:
            sections.append("交通与到达：\n- " + route_summary)
        if weather_summary:
            sections.append("天气节奏：\n- " + weather_summary)
        if hotel_summary:
            sections.append("住宿落点：\n- " + hotel_summary)
        if food_summary:
            sections.append("餐饮补位：\n- " + food_summary)
        if area_summary:
            sections.append("片区动线：\n- " + area_summary)
        if xhs_summary:
            sections.append("经验整合：\n- " + xhs_summary)
        if place_summary:
            sections.append("玩法筛选：\n- " + place_summary)
        budget_section = self._build_budget_section(profile, blueprint)
        if budget_section:
            sections.append(budget_section)
        if day_cards:
            sections.append("按天详细安排：\n" + "\n\n".join(day_cards))

        closing = self._build_missing_info_closing(profile)
        if closing:
            sections.append(closing)
        return "\n\n".join(section for section in sections if section.strip())

    def _summarize_weather_result(self, tool_results: list[dict[str, Any]]) -> str:
        for item in tool_results:
            if item.get("tool") != "weather":
                continue
            result = item.get("result") or {}
            if not isinstance(result, dict):
                continue
            if result.get("error"):
                return "天气接口这次没有成功返回结果，我先按常规节奏帮你排，后面你补充出行日期后我可以再把天气因素补进去。"
            city = str(result.get("city") or "目的地")
            forecast = str(result.get("forecast") or "天气待确认")
            temp_min = result.get("temperature_min")
            temp_max = result.get("temperature_max")
            temp_text = f"，{temp_min}~{temp_max}°C" if temp_min is not None and temp_max is not None else ""
            recommendation = str(result.get("recommendation") or "").strip()
            return f"{city}当前参考天气为{forecast}{temp_text}。{recommendation}".strip()
        return ""

    def _summarize_route_result(self, tool_results: list[dict[str, Any]]) -> str:
        for item in tool_results:
            if item.get("tool") != "route_plan":
                continue
            result = item.get("result") or {}
            if not isinstance(result, dict):
                continue
            origin = str(result.get("origin") or (item.get("args") or {}).get("origin") or "出发地")
            destination = str(result.get("destination") or (item.get("args") or {}).get("destination") or "目的地")
            distance = result.get("distance_km")
            duration = result.get("duration_minutes")
            if distance == 0 or destination == origin:
                return f"当前识别到的跨城路线存在异常：系统把路线理解成了“{origin}→{destination}”。修正出发地和目的地后，我会重新给你比较大交通方案。"
            return f"当前识别到的路线是{origin}→{destination}，驾车参考约{distance}公里、{duration}分钟；真实出行建议仍应优先比较高铁和飞机，再结合落地后的接驳。"
        return ""

    def _summarize_xiaohongshu_result(self, tool_results: list[dict[str, Any]]) -> str:
        for item in tool_results:
            if item.get("tool") != "xiaohongshu_search":
                continue
            result = item.get("result") or {}
            if not isinstance(result, dict):
                continue
            notes = result.get("notes") or []
            insights = result.get("insights") or []
            if isinstance(notes, list) and notes:
                note_lines = []
                for note in notes[:3]:
                    if not isinstance(note, dict):
                        continue
                    title = str(note.get("title") or "笔记")
                    summary = str(note.get("summary") or "").strip()
                    note_lines.append(f"{title}：{summary}" if summary else title)
                if note_lines:
                    return "很多笔记集中提到：" + "；".join(note_lines)
            if isinstance(insights, list) and insights:
                return "；".join(str(item) for item in insights[:2] if item)
            raw_count = int(result.get("raw_count") or 0)
            if raw_count > 0:
                return f"小红书接口已经返回了{raw_count}条原始结果，但当前可直接落地的摘要还不够稳定；我已继续用地图和酒店结果补足主行程。"
        return ""

    def _collect_place_groups(self, tool_results: list[dict[str, Any]]) -> list[str]:
        names: list[str] = []
        for item in tool_results:
            if item.get("tool") != "place_search":
                continue
            places = item.get("result") or []
            if not isinstance(places, list):
                continue
            for place in places[:5]:
                if not isinstance(place, dict):
                    continue
                name = str(place.get("name") or "").strip()
                if not name or self._looks_like_fake_poi_name(name):
                    continue
                if name not in names:
                    names.append(name)
        return names[:8]

    def _collect_hotel_candidates(self, tool_results: list[dict[str, Any]]) -> list[str]:
        hotels: list[str] = []
        for item in tool_results:
            if item.get("tool") != "hotel_search":
                continue
            results = item.get("result") or []
            if not isinstance(results, list):
                continue
            for hotel in results[:3]:
                if not isinstance(hotel, dict):
                    continue
                name = str(hotel.get("name") or "").strip()
                if not name or self._looks_like_fake_hotel_name(name):
                    continue
                if name not in hotels:
                    hotels.append(name)
        return hotels[:3]

    def _looks_like_fake_poi_name(self, name: str) -> bool:
        text = str(name or "").strip()
        if not text:
            return True
        reject_keywords = [
            "有限公司", "公司", "集团", "产业园", "园区", "写字楼", "办公楼", "研发中心", "工厂", "学校", "大学",
            "医院", "诊所", "药店", "北门", "南门", "东门", "西门", "停车场", "公交站", "地铁站", "收费站",
            "服务区", "仓库", "物流", "营业厅", "银行", "支行", "派出所", "居委会", "小区", "宿舍", "菜市场",
        ]
        if any(keyword in text for keyword in reject_keywords):
            return True
        suspicious_keywords = ["西湖", "博物馆", "老街"]
        if any(keyword in text for keyword in suspicious_keywords) and "海南安排" in text:
            return True
        return False

    def _looks_like_fake_hotel_name(self, name: str) -> bool:
        text = str(name or "").strip()
        if not text:
            return True
        hotel_markers = ["酒店", "民宿", "客栈", "公寓", "宾馆", "度假"]
        if any(marker in text for marker in hotel_markers):
            return False
        return self._looks_like_fake_poi_name(text)

    def _extract_weather_constraints(self, tool_results: list[dict[str, Any]]) -> dict[str, Any]:
        for item in tool_results:
            if item.get("tool") != "weather":
                continue
            result = item.get("result") or {}
            if not isinstance(result, dict) or result.get("error"):
                continue
            return {
                "forecast": str(result.get("forecast") or "").strip(),
                "date_label": str(result.get("date_label") or "出发当天").strip(),
                "recommendation": str(result.get("recommendation") or "").strip(),
                "indoor_bias": bool(result.get("indoor_bias")),
                "mode": str(result.get("travel_mode_hint") or "balanced").strip() or "balanced",
            }
        return {"forecast": "", "date_label": "", "recommendation": "", "indoor_bias": False, "mode": "balanced"}

    def _collect_valid_hotels(self, tool_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
        hotels: list[dict[str, Any]] = []
        for item in tool_results:
            if item.get("tool") != "hotel_search":
                continue
            results = item.get("result") or []
            if not isinstance(results, list):
                continue
            for hotel in results:
                if not isinstance(hotel, dict):
                    continue
                name = str(hotel.get("name") or "").strip()
                if not name or self._looks_like_fake_hotel_name(name):
                    continue
                hotels.append(hotel)
        return hotels[:5]

    def _collect_valid_places(self, tool_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
        deduped: dict[str, dict[str, Any]] = {}
        keyword_priority = {
            "自然景点": 8,
            "山水风景": 8,
            "湿地公园": 8,
            "公园": 7,
            "景点": 5,
            "古镇": 5,
            "博物馆": 4,
            "艺术馆": 4,
            "夜景": 4,
            "商圈": 2,
            "小吃街": 2,
            "餐厅": 6,
            "美食": 6,
        }
        natural_markers = ["公园", "湿地", "湖", "山", "森林", "绿道", "植物园", "动物园", "海滨", "海滩", "江", "河", "生态"]
        culture_markers = ["博物馆", "艺术馆", "美术馆", "故居", "古镇", "古城", "寺", "祠", "纪念馆"]
        night_markers = ["夜景", "步行街", "街区", "老街", "小吃街", "广场", "商圈"]
        food_markers = ["餐厅", "饭店", "火锅", "小吃", "烧烤", "面馆", "海鲜", "咖啡", "甜品", "茶餐厅", "夜宵", "美食"]
        for item in tool_results:
            if item.get("tool") != "place_search":
                continue
            keyword = str((item.get("args") or {}).get("keyword") or "景点").strip() or "景点"
            results = item.get("result") or []
            if not isinstance(results, list):
                continue
            for index, place in enumerate(results):
                if not isinstance(place, dict):
                    continue
                name = str(place.get("name") or "").strip()
                if not name or self._looks_like_fake_poi_name(name):
                    continue
                address = str(place.get("address") or "").strip()
                combined = f"{keyword} {name} {address}"
                score = keyword_priority.get(keyword, 2) * 10 - index
                if keyword in {"自然景点", "山水风景", "湿地公园", "公园"}:
                    if not any(marker in combined for marker in natural_markers):
                        score -= 25
                    else:
                        score += 10
                if keyword in {"博物馆", "艺术馆", "古镇"} and any(marker in combined for marker in culture_markers):
                    score += 8
                if keyword in {"夜景", "小吃街", "商圈"} and any(marker in combined for marker in night_markers):
                    score += 6
                if keyword in {"餐厅", "美食"}:
                    if not any(marker in combined for marker in food_markers):
                        score -= 20
                    else:
                        score += 12
                dedupe_key = f"{name}|{address}"
                normalized = dict(place)
                normalized["query_keyword"] = keyword
                normalized["place_score"] = score
                current = deduped.get(dedupe_key)
                if current is None or float(normalized.get("place_score") or 0) > float(current.get("place_score") or 0):
                    deduped[dedupe_key] = normalized
        ranked = sorted(
            [item for item in deduped.values() if float(item.get("place_score") or 0) > 0],
            key=lambda item: (
                -float(item.get("place_score") or 0),
                len(str(item.get("address") or "")),
                str(item.get("name") or ""),
            ),
        )
        return ranked[:18]

    def _classify_place_theme(self, place: dict[str, Any], preferences: list[Any]) -> str:
        text = f"{place.get('query_keyword') or ''} {place.get('name') or ''} {place.get('address') or ''} {' '.join(str(item) for item in preferences if item)}"
        if any(token in text for token in ["餐厅", "饭店", "火锅", "小吃", "面馆", "咖啡", "甜品", "夜宵"]):
            return "美食体验"
        if any(token in text for token in ["博物馆", "艺术馆", "古建", "人文"]):
            return "人文漫游"
        if any(token in text for token in ["夜景", "街区", "老街", "小吃街", "美食"]):
            return "夜游休闲"
        if any(token in text for token in ["公园", "湿地", "自然", "景点", "山", "湖"]):
            return "自然景观"
        return "城市经典"

    def _select_hotel_strategy(self, hotels: list[dict[str, Any]], preferences: list[Any], budget: Any, days: int) -> dict[str, Any]:
        if not hotels:
            return {"summary": "住宿结果暂时不稳定，建议优先住在城市核心商圈或交通方便区域，减少每天折返。", "candidates": []}
        sorted_hotels = sorted(
            hotels,
            key=lambda item: (
                -(float(item.get("rating") or 0.0)),
                float(item.get("price") or 9999),
            ),
        )
        selected = sorted_hotels[:3]
        budget_text = ""
        if isinstance(budget, int) and budget > 0:
            per_night_cap = self._hotel_search_budget_cap(budget, days, 2) or 0
            if per_night_cap:
                budget_text = f"，优先控制在约{per_night_cap}元/晚上下"
        summary = "住宿优先放在交通便利、晚间餐饮丰富、回酒店不折返的区域" + budget_text + "。"
        return {"summary": summary, "candidates": selected}

    def _build_daily_plan(self, city: str, days: int, places: list[dict[str, Any]], hotel_strategy: dict[str, Any], weather_constraints: dict[str, Any]) -> list[dict[str, Any]]:
        if days <= 0:
            return []
        grouped: dict[str, list[dict[str, Any]]] = {}
        for place in places:
            theme = str(place.get("theme") or "城市经典")
            grouped.setdefault(theme, []).append(place)

        theme_priority = ["自然景观", "城市经典", "人文漫游", "美食体验", "夜游休闲"]
        ordered_themes = [theme for theme in theme_priority if grouped.get(theme)]
        for theme in grouped:
            if theme not in ordered_themes:
                ordered_themes.append(theme)
        if not ordered_themes:
            ordered_themes = ["城市经典"]

        hotel_names = [str(item.get("name") or "").strip() for item in hotel_strategy.get("candidates") or [] if item.get("name")]
        hotel_hint = hotel_names[0] if hotel_names else f"{city}核心片区酒店"
        weather_mode = str(weather_constraints.get("mode") or "balanced")
        weather_note = str(weather_constraints.get("recommendation") or "").strip()

        used_names: set[str] = set()

        def pick_places(theme_name: str, count: int = 2, region_hint: str | None = None) -> list[dict[str, Any]]:
            theme_places = grouped.get(theme_name) or []
            filtered = theme_places
            if region_hint:
                region_matches = [
                    place for place in theme_places
                    if region_hint in str(place.get("address") or "") or region_hint in str(place.get("name") or "")
                ]
                if region_matches:
                    filtered = region_matches
            unique = [place for place in filtered if str(place.get("name") or "").strip() and str(place.get("name") or "").strip() not in used_names]
            selected = unique[:count]
            if len(selected) < count:
                fallback = [place for place in filtered if str(place.get("name") or "").strip() and place not in selected]
                selected.extend(fallback[: count - len(selected)])
            for place in selected:
                name = str(place.get("name") or "").strip()
                if name:
                    used_names.add(name)
            return selected

        def infer_region(place: dict[str, Any]) -> str:
            address = str(place.get("address") or "").strip()
            if address:
                tokens = [token for token in ["区", "镇", "路", "街", "公园", "景区", "商圈"] if token in address]
                if tokens:
                    token = tokens[0]
                    idx = address.find(token)
                    if idx > 0:
                        return address[max(0, idx - 4): idx + len(token)]
                return address[:8]
            return str(place.get("name") or city)

        def build_day_role(day_index: int, total_days: int) -> dict[str, str]:
            if total_days == 1:
                return {"title": "核心体验日", "morning": "把最值得去的核心点放在上午", "afternoon": "下午延续主线游玩", "evening": "晚上轻松收尾"}
            if day_index == 0:
                return {"title": "到达适应日", "morning": "先安排进入状态较快的内容", "afternoon": "下午以同片区轻松游玩为主", "evening": "晚上用散步和晚餐慢慢收尾"}
            if day_index == total_days - 1:
                return {"title": "收尾返程日", "morning": "上午安排体验完整但强度别太高的点", "afternoon": "下午留一点机动和返程余量", "evening": "晚上以回酒店或返程前收尾为主"}
            return {"title": "深度主玩日", "morning": "上午安排当天最核心的主景点", "afternoon": "下午继续串联第二个重点区域", "evening": "晚上补一个夜游或慢逛点"}

        def choose_food_stop(region_hint: str, fallback_name: str) -> str:
            food_candidates = grouped.get("美食体验") or []
            if not food_candidates:
                return fallback_name
            for place in food_candidates:
                name = str(place.get("name") or "").strip()
                address = str(place.get("address") or "").strip()
                if region_hint and (region_hint in address or region_hint in name):
                    return name or fallback_name
            first = food_candidates[0] if food_candidates else {}
            return str(first.get("name") or fallback_name)

        def build_day_details(
            day_role: dict[str, str],
            morning_place: str,
            afternoon_place: str,
            evening_place: str,
            lunch_place: str,
            dinner_place: str,
            region_hint: str,
        ) -> list[str]:
            if weather_mode == "prefer_indoor":
                morning_text = f"{day_role['morning']}，优先在{morning_place}附近找室内或半室内内容，避开降雨影响。"
                afternoon_text = f"{day_role['afternoon']}，转到{afternoon_place}周边继续安排室内备选，尽量围绕{region_hint}活动。"
                evening_text = f"{day_role['evening']}，晚上安排{evening_place}一带收尾，兼顾晚餐、散步或商圈活动。"
            elif weather_mode == "avoid_midday_outdoor":
                morning_text = f"{day_role['morning']}，优先安排{morning_place}，把户外核心段放在上午完成。"
                afternoon_text = f"{day_role['afternoon']}，下午衔接{afternoon_place}，但尽量避开最晒时段，并把动线收在{region_hint}附近。"
                evening_text = f"{day_role['evening']}，傍晚后再去{evening_place}，体感会更舒服。"
            else:
                morning_text = f"{day_role['morning']}，优先安排{morning_place}，把当天最重要的体验前置。"
                afternoon_text = f"{day_role['afternoon']}，下午衔接{afternoon_place}，尽量保持在{region_hint}同片区，减少跨区折返。"
                evening_text = f"{day_role['evening']}，晚上安排{evening_place}一带收尾，兼顾散步、夜景或晚餐。"

            return [
                f"09:00-11:30：{morning_text}",
                f"12:00-13:30：午餐优先放在{lunch_place}附近，尽量沿上午到下午的动线解决吃饭，减少为了用餐单独折返。",
                f"14:00-17:00：{afternoon_text}",
                f"18:30-21:00：{evening_text}，晚餐可优先考虑{dinner_place}周边，结束后回{hotel_hint}休息。",
                f"区域节奏：当天优先围绕{region_hint}展开，把景点、用餐和回酒店的移动时间压缩到更低。",
            ]

        plan: list[dict[str, Any]] = []
        for idx in range(days):
            theme = ordered_themes[idx % len(ordered_themes)]
            day_places = pick_places(theme, count=2)
            if len(day_places) < 2:
                for fallback_theme in ordered_themes:
                    if fallback_theme == theme:
                        continue
                    day_places.extend(pick_places(fallback_theme, count=2 - len(day_places)))
                    if len(day_places) >= 2:
                        break

            morning_place = str((day_places[0] if day_places else {}).get("name") or f"{city}{theme}代表点")
            afternoon_place = str((day_places[1] if len(day_places) > 1 else day_places[0] if day_places else {}).get("name") or morning_place)
            region_hint = infer_region(day_places[0]) if day_places else city

            evening_theme = "夜游休闲" if grouped.get("夜游休闲") else theme
            evening_candidates = pick_places(evening_theme, count=1, region_hint=region_hint)
            evening_place = str((evening_candidates[0] if evening_candidates else {}).get("name") or afternoon_place)
            lunch_place = choose_food_stop(region_hint, morning_place)
            dinner_place = choose_food_stop(region_hint, evening_place)
            day_role = build_day_role(idx, days)
            details = build_day_details(day_role, morning_place, afternoon_place, evening_place, lunch_place, dinner_place, region_hint)
            if weather_note:
                details.append(f"天气提醒：{weather_note}")
            plan.append({
                "day": idx + 1,
                "theme": f"{theme} · {day_role['title']}",
                "hotel_hint": hotel_hint,
                "region_hint": region_hint,
                "lunch_place": lunch_place,
                "dinner_place": dinner_place,
                "details": details,
            })
        return plan

    def _render_daily_plan_cards(self, daily_plan: list[dict[str, Any]]) -> list[str]:
        cards: list[str] = []
        for item in daily_plan:
            if not isinstance(item, dict):
                continue
            day = item.get("day")
            theme = str(item.get("theme") or "当日安排")
            details = item.get("details") or []
            if not isinstance(details, list) or not details:
                continue
            lines = [f"第{day}天：{theme}"]
            lines.extend(f"- {detail}" for detail in details if detail)
            cards.append("\n".join(lines))
        return cards

    def _build_budget_section(self, profile: dict[str, Any], blueprint: dict[str, Any]) -> str:
        budget = profile.get("budget")
        days = int(profile.get("days") or settings.default_trip_days)
        nights = max(days - 1, 1)
        if not isinstance(budget, int) or budget <= 0:
            return ""
        hotel_cap = self._hotel_search_budget_cap(budget, days, profile.get("group_size")) or max(300, int(budget * 0.25 / nights))
        hotel_candidates = blueprint.get("hotel_candidates") or []
        hotel_hint = "住宿优先锁定后，再用剩余预算去分配门票、餐饮和市内交通。"
        if hotel_candidates:
            hotel_hint = f"住宿尽量按{hotel_cap}元/晚左右筛选，优先看已选出的交通便利候选。"
        return (
            "预算分配建议：\n"
            f"- 住宿建议按约{hotel_cap}元/晚上下控制，避免把预算过早压在单晚房费上；\n"
            "- 大交通优先锁定往返，再决定是否把更多预算给门票体验或特色餐饮；\n"
            f"- {hotel_hint}"
        )

    def _build_trip_blueprint(self, profile: dict[str, Any], tool_results: list[dict[str, Any]]) -> dict[str, Any]:
        city = str(profile.get("destination") or settings.default_city)
        days = int(profile.get("days") or settings.default_trip_days)
        preferences = profile.get("preferences") or []

        valid_places = self._collect_valid_places(tool_results)
        for place in valid_places:
            place["theme"] = self._classify_place_theme(place, preferences)
        valid_hotels = self._collect_valid_hotels(tool_results)
        hotel_strategy = self._select_hotel_strategy(valid_hotels, preferences, profile.get("budget"), days)
        weather_constraints = self._extract_weather_constraints(tool_results)
        daily_plan = self._build_daily_plan(city, days, valid_places, hotel_strategy, weather_constraints)

        place_summary_parts: list[str] = []
        theme_seen: dict[str, list[str]] = {}
        food_names: list[str] = []
        area_names: list[str] = []
        for place in valid_places:
            theme = str(place.get("theme") or "城市经典")
            theme_seen.setdefault(theme, [])
            name = str(place.get("name") or "").strip()
            address = str(place.get("address") or "").strip()
            if name and name not in theme_seen[theme]:
                theme_seen[theme].append(name)
            if theme == "美食体验" and name and name not in food_names:
                food_names.append(name)
            region = address or name
            if region and region not in area_names:
                area_names.append(region)
        for theme, names in theme_seen.items():
            if names:
                place_summary_parts.append(f"{theme}优先看：" + "、".join(names[:3]))

        hotel_summary = hotel_strategy.get("summary") or self._summarize_hotel_result(tool_results)
        selected_hotels = hotel_strategy.get("candidates") or []
        if selected_hotels:
            hotel_lines = []
            for hotel in selected_hotels[:3]:
                if not isinstance(hotel, dict):
                    continue
                name = str(hotel.get("name") or "酒店")
                area = str(hotel.get("area") or hotel.get("location") or "区域待确认")
                price = hotel.get("price")
                price_text = f"约{price}元/晚" if price is not None else "价格待确认"
                hotel_lines.append(f"{name}（{area}，{price_text}）")
            if hotel_lines:
                hotel_summary = hotel_summary + " 可优先看：" + "、".join(hotel_lines)

        weather_summary = self._summarize_weather_result(tool_results)
        route_summary = self._summarize_route_result(tool_results)
        xhs_summary = self._summarize_xiaohongshu_result(tool_results)
        food_summary = self._build_food_summary(valid_places, daily_plan)
        area_summary = self._build_area_summary(daily_plan, area_names)

        return {
            "city": city,
            "days": days,
            "weather_summary": weather_summary,
            "route_summary": route_summary,
            "xhs_summary": xhs_summary,
            "hotel_summary": hotel_summary,
            "food_summary": food_summary,
            "area_summary": area_summary,
            "place_summary": "；".join(place_summary_parts[:4]) or self._summarize_place_result(tool_results),
            "daily_plan": daily_plan,
            "place_groups": [str(place.get("name") or "").strip() for place in valid_places if place.get("name")][:12],
            "hotel_candidates": [str(hotel.get("name") or "").strip() for hotel in selected_hotels if hotel.get("name")][:3],
            "food_candidates": food_names[:6],
            "weather_constraints": weather_constraints,
            "regions": area_names[:8],
        }

    def _build_food_summary(self, valid_places: list[dict[str, Any]], daily_plan: list[dict[str, Any]]) -> str:
        food_names: list[str] = []
        for place in valid_places:
            if str(place.get("theme") or "") != "美食体验":
                continue
            name = str(place.get("name") or "").strip()
            if name and name not in food_names:
                food_names.append(name)
        meal_hints: list[str] = []
        for item in daily_plan[:3]:
            if not isinstance(item, dict):
                continue
            lunch_place = str(item.get("lunch_place") or "").strip()
            dinner_place = str(item.get("dinner_place") or "").strip()
            if lunch_place:
                meal_hints.append(f"白天可优先在{lunch_place}附近解决午餐")
            if dinner_place and dinner_place != lunch_place:
                meal_hints.append(f"晚上可把{dinner_place}作为收尾晚餐点")
        if food_names and meal_hints:
            return "餐饮候选可优先看：" + "、".join(food_names[:4]) + "；" + "；".join(meal_hints[:3])
        if food_names:
            return "餐饮候选可优先看：" + "、".join(food_names[:4])
        if meal_hints:
            return "；".join(meal_hints[:3])
        return ""

    def _build_area_summary(self, daily_plan: list[dict[str, Any]], regions: list[str]) -> str:
        summaries: list[str] = []
        for item in daily_plan[:3]:
            if not isinstance(item, dict):
                continue
            day = item.get("day")
            region_hint = str(item.get("region_hint") or "").strip()
            if day and region_hint:
                summaries.append(f"第{day}天优先围绕{region_hint}同片区展开")
        if summaries:
            return "；".join(summaries)
        if regions:
            return "优先把玩法集中在这些片区，减少跨区折返：" + "、".join(regions[:3])
        return ""

    def _build_daily_dayparts(
        self,
        city: str,
        days: int,
        place_groups: list[str],
        hotel_candidates: list[str],
        weather_summary: str,
        xhs_summary: str,
    ) -> list[str]:
        cards: list[str] = []
        if not place_groups:
            return cards
        pool = place_groups
        for idx in range(days):
            morning = pool[min(idx * 2, len(pool) - 1)]
            afternoon = pool[min(idx * 2 + 1, len(pool) - 1)]
            evening = pool[min(idx * 2 + 2, len(pool) - 1)] if len(pool) > 2 else pool[-1]
            hotel_hint = hotel_candidates[min(idx, len(hotel_candidates) - 1)] if hotel_candidates else f"{city}交通便利的住宿点"
            lines = [
                f"第{idx + 1}天：",
                f"- 09:00-11:30：优先安排{morning}，尽量把核心户外段放在上午，拍照和步行强度也更容易控制。",
                f"- 12:00-13:30：在上午景点附近吃午饭，避免跨区折返；如果天气热，午休或找商场/馆内项目过渡。",
                f"- 14:00-17:00：继续衔接{afternoon}，把同片区的自然景观或经典景点串在一起。",
                f"- 18:30-21:00：晚上安排{evening}周边夜景、散步或美食街，结束后回{hotel_hint}休息。",
                f"- 交通衔接：当天尽量保持同一区域游玩，景点之间优先地铁+短打车，减少高温和排队消耗。",
            ]
            if weather_summary:
                lines.append(f"- 天气提醒：{weather_summary}")
            if xhs_summary and idx == 0:
                lines.append(f"- 小红书补充：{xhs_summary}")
            cards.append("\n".join(lines))
        return cards

    def _summarize_hotel_result(self, tool_results: list[dict[str, Any]]) -> str:
        for item in tool_results:
            if item.get("tool") != "hotel_search":
                continue
            hotels = item.get("result") or []
            if not isinstance(hotels, list) or not hotels:
                continue
            top_lines = []
            for hotel in hotels[:3]:
                if not isinstance(hotel, dict):
                    continue
                name = str(hotel.get("name") or "酒店")
                area = str(hotel.get("area") or hotel.get("location") or "区域待确认")
                price = hotel.get("price")
                price_text = f"约{price}元/晚" if price is not None else "价格待确认"
                top_lines.append(f"{name}（{area}，{price_text}）")
            if top_lines:
                return "可以优先看：" + "、".join(top_lines) + "。"
        return ""

    def _summarize_food_result(self, tool_results: list[dict[str, Any]]) -> str:
        food_lines: list[str] = []
        for item in tool_results:
            if item.get("tool") != "place_search":
                continue
            keyword = str((item.get("args") or {}).get("keyword") or "")
            if keyword not in {"餐厅", "美食"}:
                continue
            places = item.get("result") or []
            if not isinstance(places, list) or not places:
                continue
            names = []
            for place in places[:4]:
                if not isinstance(place, dict) or not place.get("name"):
                    continue
                name = str(place.get("name") or "").strip()
                if not name or self._looks_like_fake_poi_name(name):
                    continue
                names.append(name)
            if names:
                food_lines.append(f"{keyword}方向可优先看：" + "、".join(names))
        return "；".join(food_lines[:2])

    def _summarize_place_result(self, tool_results: list[dict[str, Any]]) -> str:
        place_lines: list[str] = []
        for item in tool_results:
            if item.get("tool") != "place_search":
                continue
            keyword = str((item.get("args") or {}).get("keyword") or "景点")
            if keyword in {"餐厅", "美食"}:
                continue
            places = item.get("result") or []
            if not isinstance(places, list) or not places:
                continue
            names = []
            for place in places[:4]:
                if not isinstance(place, dict) or not place.get("name"):
                    continue
                name = str(place.get("name") or "").strip()
                if not name or self._looks_like_fake_poi_name(name):
                    continue
                names.append(name)
            if names:
                place_lines.append(f"{keyword}方向可以重点看：" + "、".join(names))
        return "；".join(place_lines[:2])

    def _answer_mentions_missing_fields(self, answer: str, missing: list[str]) -> bool:
        if not missing:
            return True
        text = answer or ""
        if "还缺" in text and any(field in text for field in missing):
            return True
        if "补充" in text and any(phrase in text for phrase in ("更准", "更贴合", "更准确", "更详细", "细化")):
            mentioned = sum(1 for field in missing if field in text)
            if mentioned >= min(2, len(missing)) or ("更多信息" in text and mentioned >= 1):
                return True
        if "更多信息" in text and any(field in text for field in missing):
            return True
        return False

    def _strip_detail_only_closing(self, answer: str) -> str:
        import re

        text = (answer or "").rstrip()
        patterns = [
            r"\n*如果你愿意，我下一步可以直接给你输出一版更细的[^。\n]*。?$",
            r"\n*如果你愿意，我可以继续直接展开成一版[^。\n]*详细行程表[^。\n]*。?$",
            r"\n*如果你愿意，我可以继续输出一版[^。\n]*详细[^。\n]*。?$",
        ]
        for pattern in patterns:
            text = re.sub(pattern, "", text, flags=re.DOTALL).rstrip()
        return text

    def _ensure_missing_info_closing(
        self,
        answer: str,
        profile: dict[str, Any],
        intent: str,
    ) -> tuple[str, bool]:
        if intent not in {"trip_plan", "detailed_trip_plan", "transport_advice"}:
            return answer, False

        missing = self._missing_trip_fields(profile)
        if not missing:
            return answer, False

        if self._answer_mentions_missing_fields(answer, missing):
            return answer, True

        closing = self._build_missing_info_closing(profile)
        trimmed = self._strip_detail_only_closing(answer)
        if trimmed:
            return f"{trimmed}\n\n{closing}", True
        return closing, True

    def _build_transport_advice_fallback(self, profile: dict[str, Any], message: str) -> str:
        origin = str(profile.get("origin") or "出发地")
        destination = str(profile.get("destination") or "目的地")
        days = profile.get("days")
        budget = profile.get("budget")
        group_size = profile.get("group_size")
        preferences = profile.get("preferences") or []

        route = self.tencent_map.route_plan(origin, destination, mode="driving")
        route_line = self._format_route_line(route) if not route.get("fallback") else f"{origin} → {destination}"
        origin_known = bool(origin and origin != "出发地")
        comparison = self._build_transport_comparison(profile, route, origin_known=origin_known)

        summary_parts: list[str] = []
        if isinstance(group_size, int) and group_size > 0:
            summary_parts.append(f"{group_size}人")
        if isinstance(days, int) and days > 0:
            summary_parts.append(f"{days}天")
        if isinstance(budget, int) and budget > 0:
            summary_parts.append(f"预算{budget}元")
        if isinstance(preferences, list) and preferences:
            summary_parts.append(f"偏好{'、'.join(str(item) for item in preferences if item)}")

        if origin_known:
            intro = f"如果只基于你当前这句问题，我更建议从{origin}去{destination}优先考虑{comparison['recommendation']}。"
            route_intro = f"先看基础路程：{route_line}。"
        else:
            intro = f"如果现在只知道目的地是{destination}，还不能直接判断高铁还是飞机一定更合适，关键要看你的出发地和两地之间的总耗时。"
            route_intro = "在没有出发地的前提下，只能先给你一个通用判断框架。"

        comparison_title = "高铁和飞机怎么选：" if self._is_highspeed_vs_flight_question(message) else "几种方式怎么选："
        answer = (
            intro
            + "\n\n"
            + route_intro
            + "\n\n"
            + "为什么这样建议：\n"
            + "\n".join(f"- {reason}" for reason in comparison["reasons"])
            + "\n\n"
            + comparison_title
            + "\n"
            + "\n".join(f"- {line}" for line in comparison["ranking_lines"])
        )

        if not origin_known:
            answer += "\n\n如果你告诉我出发城市，我可以直接按这趟行程帮你判断到底更适合高铁还是飞机。"
        elif summary_parts:
            answer += "\n\n补充说明：如果再结合你明确提供的" + "、".join(summary_parts) + "，我还可以把建议进一步细化。"
        else:
            answer += "\n\n如果你愿意，我下一步可以在你补充天数、预算、人数或偏好后，再把建议细化到更贴合你的行程。"

        return answer

    def _build_transport_comparison(self, profile: dict[str, Any], route: dict[str, Any], origin_known: bool = True) -> dict[str, Any]:
        destination = str(profile.get("destination") or "目的地")
        days = profile.get("days")
        budget = profile.get("budget")
        group_size = profile.get("group_size")
        preferences = profile.get("preferences") or []
        distance = int(route.get("distance_km") or 0)

        recommendation = "先看出发地再判断"
        reasons: list[str] = []
        if isinstance(days, int) and days > 0:
            reasons.append(f"{days}天行程里，通常应该优先压缩路上耗时，把白天留给{destination}游玩。")
        else:
            reasons.append(f"如果你的行程天数不长，通常应该优先压缩路上耗时，把白天留给{destination}游玩。")

        if isinstance(group_size, int) and group_size > 0:
            reasons.append(f"{group_size}人同行时，交通不仅要看票价，还要看稳定性、到站后的接驳效率和体力消耗。")
        else:
            reasons.append("旅游出行时，交通不仅要看票价，还要看稳定性、到站后的接驳效率和体力消耗。")

        ranking_lines = [
            "高铁：综合稳定性最好，适合多数城市间出行。",
            "自驾：适合景点分散、强依赖沿途停靠的路线。",
            "飞机：适合超远距离、时间特别紧、或者高铁耗时明显过长的情况。",
        ]

        if not origin_known:
            reasons.append("没有出发地时，不能直接替用户选高铁还是飞机，因为这本质上取决于门到门总耗时，而不是只看车程或飞行时长。")
            reasons.append("通用判断可以这样看：如果高铁4到5小时左右能到，通常高铁会比飞机更省心；如果高铁明显超过6小时，而机票和机场接驳都合适，飞机通常更有优势。")
            if destination == "上海":
                reasons.append("去上海要特别把落点算进去：高铁到虹桥站通常接地铁最方便；飞机落虹桥还不错，但如果落浦东，去市区的通勤时间要额外算。")
            else:
                reasons.append("比较时不要只看票价或飞行时间，还要把进站、安检、候机和到站后的接驳时间一起算进去。")
            reasons.append("所以更好的回答方式不是直接下结论，而是先给你判断框架，再说明补充出发城市后就能判断得更准。")
            ranking_lines = [
                "高铁更合适：中短距离、车站到市区接驳方便、想少折腾、看重准点性。",
                "飞机更合适：远距离、希望明显压缩跨城时间，而且机场往返市区也比较顺。",
            ]
            return {
                "recommendation": recommendation,
                "reasons": reasons[:5],
                "ranking_lines": ranking_lines,
            }

        if distance >= 1400:
            recommendation = "飞机"
            reasons.append("自驾路程过长，连续驾驶和中途停留会明显挤压游玩时间。")
            reasons.append("这种距离下，飞机通常是最节省时间的，落地后再转地铁或打车更合理。")
            ranking_lines = [
                "飞机：优先级最高，适合远距离快速到达。",
                "高铁：如果临近市中心、票价合适，也可以作为稳妥备选。",
                "自驾：除非就是想做长途公路旅行，否则不建议。",
            ]
        elif distance >= 700:
            recommendation = "高铁或飞机，重点看门到门总耗时"
            reasons.append("这是典型的中长距离跨城，通常优先比较高铁和飞机，而不是先考虑自驾。")
            reasons.append("如果高铁在4到5小时左右能直达核心站点，通常整体体验更稳；如果高铁明显超过6小时，且机票价格差不多，飞机会更划算。")
            if destination == "上海":
                reasons.append("去上海时要特别看落点：高铁到虹桥站一般接地铁进城更顺；飞机如果落浦东，要把后续进市区时间单独算进去。")
            else:
                reasons.append("比较时要把值机、安检、候机和机场往返市区时间一起算，不要只看飞行时间。")
            ranking_lines = [
                "高铁：更适合想少折腾、看重准点性、且车站进城方便的城市型旅行。",
                "飞机：更适合赶时间，或者高铁耗时已经明显偏长的情况。",
            ]
        elif distance >= 350:
            recommendation = "高铁"
            reasons.append("这个距离下，高铁通常比自驾更省心，也比飞机更少折腾。")
            reasons.append("对多数短途或中短途出行来说，高铁的上车下车效率和到市区的便利度通常更好。")
        else:
            recommendation = "高铁或自驾"
            reasons.append("距离不算特别远，高铁和自驾都能考虑，关键看你更想要省心还是灵活。")
            reasons.append("如果目的地市区集中，高铁更省事；如果周边分散景点多，自驾会更自由。")
            ranking_lines = [
                "高铁：适合市区主玩的轻松型路线。",
                "自驾：适合目的地周边还要串联多个分散景点。",
                "飞机：通常没必要，流程成本偏高。",
            ]

        if isinstance(budget, int) and budget > 0 and budget <= 3000:
            reasons.append("预算相对紧时，要避免把成本和精力都花在交通上，通常高铁会更稳。")
        if "自然景观" in preferences:
            reasons.append("如果你更想去分散的自然景观点，自驾的优势会更明显；纯城市线则不一定需要。")
        if "历史建筑" in preferences or "美食" in preferences:
            reasons.append("如果核心是老城区、人文馆和美食街，落地即进城通常比自己长时间开车更合适。")

        return {
            "recommendation": recommendation,
            "reasons": reasons[:5],
            "ranking_lines": ranking_lines,
        }

    def _build_trip_plan_fallback(self, profile: dict[str, Any]) -> str:
        city = str(profile.get("destination") or settings.default_city)
        days = int(profile.get("days") or settings.default_trip_days)
        budget = profile.get("budget")
        group_size = profile.get("group_size")
        origin = str(profile.get("origin") or "").strip()
        preferences = profile.get("preferences") or []
        travel_start_date = str(profile.get("travel_start_date") or "").strip()
        travel_end_date = str(profile.get("travel_end_date") or "").strip()

        intro: list[str] = []
        if origin:
            intro.append(f"从{origin}出发")
        intro.append(f"去{city}玩{days}天")
        if travel_start_date and travel_end_date:
            intro.append(f"日期是{travel_start_date}到{travel_end_date}")
        if isinstance(group_size, int) and group_size > 0:
            intro.append(f"人数{group_size}")
        if isinstance(budget, int) and budget > 0:
            intro.append(f"预算{budget}元左右")
        if isinstance(preferences, list) and preferences:
            intro.append("偏好=" + "、".join(str(item) for item in preferences if item))

        body = "我已经拿到一部分行程条件了：" + "，".join(intro) + "。"
        closing = self._build_missing_info_closing(profile)
        if closing:
            return f"{body}\n\n{closing}"
        return f"{body}\n\n如果你愿意，我可以继续根据交通、天气、住宿、景点和每日节奏，把这次旅行细化成更完整的动态行程。"

    def _build_detailed_trip_plan_fallback(self, profile: dict[str, Any]) -> str:
        city = str(profile.get("destination") or settings.default_city)
        days = int(profile.get("days") or settings.default_trip_days)
        budget = int(profile.get("budget") or 0)
        group_size = int(profile.get("group_size") or 0)
        origin = str(profile.get("origin") or "").strip()
        preferences = profile.get("preferences") or []
        travel_start_date = str(profile.get("travel_start_date") or "").strip()
        travel_end_date = str(profile.get("travel_end_date") or "").strip()

        summary_parts = [f"目的地{city}", f"天数{days}"]
        if travel_start_date and travel_end_date:
            summary_parts.append(f"日期{travel_start_date}到{travel_end_date}")
        if origin:
            summary_parts.append(f"出发地{origin}")
        if group_size > 0:
            summary_parts.append(f"人数{group_size}")
        if budget > 0:
            summary_parts.append(f"预算{budget}元")
        if isinstance(preferences, list) and preferences:
            summary_parts.append("偏好=" + "、".join(str(item) for item in preferences if item))

        return "我已经收到你的详细行程需求，当前已确认的信息有：" + "，".join(summary_parts) + "。接下来我会结合交通、天气、住宿、景点和用户画像，生成一版动态的详细行程安排。"

    def _looks_like_trip_plan_request(self, message: str) -> bool:
        keywords = ["行程", "规划", "安排", "几日游", "日游", "旅游", "旅行", "景点", "住宿"]
        return any(keyword in message for keyword in keywords)

    def _save_turn(self, session_id: str, user_message: str, assistant_message: str) -> None:
        self.memory.append_message(session_id, "user", user_message)
        self.memory.append_message(session_id, "assistant", assistant_message)

    def _sanitize_user_message(self, message: str) -> str:
        import re

        text = (message or "").strip()
        if not text:
            return text

        markers = [
            "第1天",
            "第2天",
            "第3天",
            "第4天",
            "第5天",
            "住宿建议",
            "路线说明",
            "预算建议",
            "预算控制",
            "出行方式建议",
            "住宿落点",
            "建议路线",
            "我给你做一版更具体的",
        ]
        for marker in markers:
            idx = text.find(marker)
            if idx > 0:
                text = text[:idx].strip()
                break

        text = re.sub(r"按\s*\d+\s*元预算来看，?", "", text)
        text = re.sub(r"预算建议：.*", "", text)
        text = re.sub(r"预算控制：.*", "", text)
        text = re.sub(r"预算是\s*\d+\s*元.*", "", text)
        text = re.sub(r"预算为\s*\d+\s*元.*", "", text)
        text = re.sub(r"(?:建议路线|我给你做一版更具体的).*$", "", text)
        text = re.sub(r"\n{2,}", "\n", text)
        return text.strip()

    def _serialize_history(self, history: list[tuple[str, str]]) -> str:
        if not history:
            return "无"
        lines: list[str] = []
        for role, content in history[-8:]:
            role_text = "用户" if role == "user" else "助手"
            lines.append(f"{role_text}：{self._sanitize_user_message(content)}")
        return "\n".join(lines)

    # ===== 修改开始 =====
    def _extract_route_points(self, message: str, history: list[tuple[str, str]]) -> tuple[str | None, str | None]:
        pair = self._extract_explicit_route_pair(message)
        if pair:
            return pair

        cleaned = message.replace("？", "").replace("?", "").replace("，", " ").replace(",", " ")
        origin = self._extract_origin(message, history)
        destination = self._extract_destination(message, history)

        if origin and destination and origin == destination:
            history_destination = self._resolve_destination({}, "", history)
            if history_destination and history_destination != origin:
                destination = history_destination

        if not origin or not destination:
            import re

            for pattern in [
                r"从\s*(?P<origin>[^到去]+)\s*(?:到|去|飞到|前往)\s*(?P<destination>[^，。！？\s]+)",
                r"我现在在\s*(?P<origin>[^，。！？\s]+).{0,20}?(?:去|到)\s*(?P<destination>[^，。！？\s]+)",
                r"(?P<origin>[^，。！？\s]+)\s*(?:到|去|飞到|前往)\s*(?P<destination>[^，。！？\s]+)",
            ]:
                match = re.search(pattern, cleaned)
                if match:
                    matched_origin = self._normalize_location_text(match.group("origin").strip())
                    matched_destination = self._normalize_location_text(match.group("destination").strip())
                    if matched_origin and not self._looks_like_date_text(matched_origin):
                        origin = origin or matched_origin
                    if matched_destination and not self._looks_like_date_text(matched_destination):
                        destination = destination or matched_destination
                    break

        if not destination:
            destination = self._resolve_destination({}, message, history)

        if not origin or not destination:
            return None, None
        return origin, destination
    # ===== 修改结束 =====

    def _session_id_from_history(self, history: list[tuple[str, str]]) -> str:
        return "default"

    def _last_user_location(self, history: list[tuple[str, str]]) -> str | None:
        for role, content in reversed(history):
            if role != "user":
                continue
            location = self._extract_city(content)
            if location:
                return location
        return None

    # ===== 修改开始 =====
    def _extract_origin(self, message: str, history: list[tuple[str, str]]) -> str | None:
        del history
        pair = self._extract_explicit_route_pair(message)
        if pair:
            return pair[0]

        cleaned = message.replace("？", "").replace("?", "")
        patterns = [
            r"(?:我现在在|我在|人在)\s*(?P<origin>[^，。！？\s]+)",
            r"从\s*(?P<origin>[^到去，。！？\s]+)",
            r"出发地[是为:]?\s*(?P<origin>[^，。！？\s]+)",
        ]
        for pattern in patterns:
            match = __import__("re").search(pattern, cleaned)
            if match:
                candidate = self._normalize_location_text(match.group("origin"))
                if candidate and not self._looks_like_date_text(candidate):
                    return candidate
        return None

    def _extract_destination(self, message: str, history: list[tuple[str, str]]) -> str | None:
        del history
        cleaned = message.replace("？", "").replace("?", "")
        if any(keyword in cleaned for keyword in ["怎么去", "怎么走", "推荐我怎么去", "建议我怎么去", "交通方式", "坐什么"]):
            return None

        pair = self._extract_explicit_route_pair(message)
        if pair:
            return pair[1]

        patterns = [
            r"(?:去|到|前往|飞到|目的地[是为:]?)\s*(?P<destination>[^，。！？\s]+)",
            r"(?:从\s*[^到去，。！？\s]+\s*(?:到|去|飞到|前往)\s*)(?P<destination>[^，。！？\s]+)",
            r"(?P<destination>[^，。！？\s]+)\s*(?:路线|行程|旅行|旅游)",
        ]
        for pattern in patterns:
            match = __import__("re").search(pattern, cleaned)
            if match:
                candidate = self._normalize_location_text(match.group("destination"))
                if candidate and not self._looks_like_date_text(candidate):
                    return candidate

        cities = self._extract_all_cities(cleaned)
        if cities:
            return cities[-1]
        return None

    def _extract_city(self, message: str) -> str | None:
        cities = self._extract_all_cities(message)
        if cities:
            return cities[0]
        return None

    def _extract_explicit_route_pair(self, message: str) -> tuple[str, str] | None:
        import re

        text = self._sanitize_user_message(message)
        if not text:
            return None
        text = re.sub(r"^(帮我|请帮我|麻烦你|给我|想让你)(规划一下|规划|安排一下|安排|做一版|看一下)?", "", text).strip()
        patterns = [
            r"(?:从)?(?P<origin>[\u4e00-\u9fa5]{2,12})\s*(?:到|去|飞到|前往)\s*(?P<destination>[\u4e00-\u9fa5]{2,12})(?:的)?(?:行程安排|旅游安排|出行安排|行程规划|旅游规划|出行规划|行程|安排|规划|旅游|旅行)?",
        ]
        stop_tokens = ["帮我规划一下", "帮我规划", "规划一下", "详细规划一下", "详细规划", "安排一下", "安排", "玩", "旅游", "旅行", "出发", "去玩"]
        for pattern in patterns:
            match = re.search(pattern, text)
            if not match:
                continue
            raw_origin = match.group("origin")
            raw_destination = match.group("destination")
            for token in stop_tokens:
                if raw_origin.startswith(token):
                    raw_origin = raw_origin[len(token):]
            for token in stop_tokens:
                if raw_destination.endswith(token):
                    raw_destination = raw_destination[: -len(token)]
            origin = self._sanitize_location_candidate(raw_origin, field_type="origin")
            destination = self._sanitize_location_candidate(raw_destination, field_type="destination")
            if origin and destination and origin != destination:
                return origin, destination
        return None

    def _extract_all_cities(self, message: str) -> list[str]:
        city_pool = [
            "北京", "上海", "杭州", "成都", "广州", "深圳", "南京", "西安", "重庆", "苏州", "济南", "青岛", "厦门", "武汉", "天津",
            "云南", "昆明", "大理", "丽江", "香格里拉", "西双版纳",
            "内蒙古", "呼和浩特", "包头", "鄂尔多斯", "赤峰", "呼伦贝尔", "满洲里", "乌兰察布",
            "长沙", "张家界", "桂林", "三亚", "哈尔滨", "长春", "沈阳"
        ]
        found: list[str] = []
        sorted_pool = sorted(city_pool, key=len, reverse=True)
        for city in sorted_pool:
            if city in message and city not in found:
                if any(city in existing and city != existing for existing in found):
                    continue
                found.append(city)
        return found

    def _extract_location_address(self, message: str) -> str | None:
        text = self._sanitize_user_message(message)
        if not text:
            return None
        import re

        patterns = [
            r"(?:我现在在|我在|人在|位置在|地址在)\s*(?P<address>[^，。！？]+)",
            r"(?:推荐一下|找一下|看看|搜一下)\s*(?P<address>[^，。！？\s]+)的(?:餐厅|饭店|酒店|景点|美食)",
            r"(?P<address>[^，。！？\s]+)的(?:餐厅|饭店|酒店|景点|美食)",
            r"(?P<address>[^，。！？\s]+)(?:附近|周边)",
        ]
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                address = match.group("address").strip()
                address = re.sub(r"^(的|这儿|这里|一下)", "", address)
                address = re.sub(r"(附近|周边|这里|的餐厅|的饭店|的酒店|的景点|的美食)$", "", address).strip()
                sanitized_address = self._sanitize_location_candidate(address, field_type="address")
                if sanitized_address:
                    return sanitized_address
        return self._sanitize_location_candidate(self._extract_city(text), field_type="address")
    # ===== 修改结束 =====

    def _extract_weather_time(self, message: str) -> str | None:
        text = self._sanitize_user_message(message)
        if not text:
            return None
        for label in ["今天", "明天", "后天", "大后天", "本周末", "周末"]:
            if label in text:
                return label
        return None

    # ===== 修改开始 =====
    def _extract_travel_date_range(self, message: str) -> dict[str, str] | None:
        text = self._sanitize_user_message(message)
        if not text:
            return None
        import re

        patterns = [
            r"(?P<start>\d{1,2}月\d{1,2}[日号]?)\s*(?:到|至|[-~—])\s*(?P<end>\d{1,2}月\d{1,2}[日号]?)",
            r"(?P<start>\d{1,2}[./]\d{1,2})\s*(?:到|至|[-~—])\s*(?P<end>\d{1,2}[./]\d{1,2})",
        ]
        for pattern in patterns:
            match = re.search(pattern, text)
            if not match:
                continue
            start_date = self._normalize_date_token(match.group("start"))
            end_date = self._normalize_date_token(match.group("end"))
            if start_date and end_date:
                return {"start_date": start_date, "end_date": end_date}
        return None
    # ===== 修改结束 =====

    def _looks_like_date_text(self, text: str) -> bool:
        if not text:
            return False
        import re

        normalized = text.strip()
        return bool(re.fullmatch(r"\d{1,2}月\d{1,2}号", normalized)) or bool(re.fullmatch(r"\d{1,2}[./]\d{1,2}", normalized)) or normalized in {"今天", "明天", "后天", "大后天"}

    # ===== 修改开始 =====
    def _extract_preference_tags(
        self,
        message: str,
        history: list[tuple[str, str]],
        profile: dict[str, Any],
        inherit_existing: bool = True,
    ) -> list[str]:
        text = " ".join([message] + [content for role, content in history if role == "user"])
        current_message = message or ""
        tag_map = {
            "自然景观": ["自然", "自然景观", "风景", "山水", "湿地", "徒步", "草原", "森林", "湖泊", "雪山", "沙漠"],
            "历史建筑": ["历史建筑", "古建", "古迹", "故宫", "寺庙", "胡同", "博物馆", "文物", "人文"],
            "美食": ["美食", "吃", "小吃", "餐厅"],
            "拍照": ["拍照", "出片", "机位"],
            "亲子": ["亲子", "带娃", "小朋友"],
            "情侣": ["情侣", "约会", "浪漫"],
            "休闲": ["轻松", "休闲", "慢游", "不赶", "悠闲"],
        }
        tags = []
        for tag, keywords in tag_map.items():
            source_text = text if inherit_existing else current_message
            if any(keyword in source_text for keyword in keywords):
                tags.append(tag)
        if inherit_existing:
            existing = profile.get("preferences")
            if isinstance(existing, list):
                for item in existing:
                    if item not in tags:
                        tags.append(item)
        return tags[:6]
    # ===== 修改结束 =====

    # ===== 修改开始 =====
    def _extract_days(self, message: str) -> int | None:
        import re

        normalized = self._sanitize_user_message(message)
        if not normalized:
            return None

        stay_match = re.search(r"([0-9]+|[一二两三四五六七八九十半]+)\s*天\s*([0-9]+|[一二两三四五六七八九十半]+)\s*晚", normalized)
        if stay_match:
            token = stay_match.group(1)
            if token.isdigit():
                return max(1, int(token))
            chinese_days = self._chinese_day_to_int(token)
            if chinese_days is not None:
                return chinese_days

        patterns = [
            r"(?:住|玩|游|行程|安排|规划|旅游|旅行)?\s*([0-9]+)\s*(?:天|日)\b",
            r"([0-9]+)\s*(?:天|日)\s*(?:游|行程|旅行|旅游)?",
            r"(?:住|玩|游|行程|安排|规划|旅游|旅行)?\s*([一二两三四五六七八九十半]+)\s*(?:天|日)(?:游)?",
            r"([一二两三四五六七八九十半]+)\s*(?:天|日)\s*(?:游|行程|旅行|旅游)?",
        ]
        for pattern in patterns:
            match = re.search(pattern, normalized)
            if not match:
                continue
            token = match.group(1)
            if token.isdigit():
                return max(1, int(token))
            chinese_days = self._chinese_day_to_int(token)
            if chinese_days is not None:
                return chinese_days
        return None
    # ===== 修改结束 =====

    def _extract_day_offset(self, message: str) -> int | None:
        text = self._sanitize_user_message(message)
        if not text:
            return None
        if "今天" in text:
            return 0
        if "明天" in text:
            return 1
        if "后天" in text:
            return 2
        if "大后天" in text:
            return 3
        return None

    def _extract_date_label(self, message: str) -> str | None:
        text = self._sanitize_user_message(message)
        if not text:
            return None
        for label in ["今天", "明天", "后天", "大后天"]:
            if label in text:
                return label
        return None

    def _chinese_day_to_int(self, token: str) -> int | None:
        if not token:
            return None
        if token == "半":
            return 1

        mapping = {
            "一": 1,
            "二": 2,
            "两": 2,
            "三": 3,
            "四": 4,
            "五": 5,
            "六": 6,
            "七": 7,
            "八": 8,
            "九": 9,
            "十": 10,
        }

        if token in mapping:
            return mapping[token]
        if token == "十一":
            return 11
        if token == "十二":
            return 12
        if token.startswith("十") and len(token) == 2:
            return 10 + mapping.get(token[1], 0)
        if token.endswith("十") and len(token) == 2:
            return mapping.get(token[0], 1) * 10
        if len(token) == 2 and token[0] in mapping and token[1] in mapping:
            tens = mapping[token[0]]
            ones = mapping[token[1]]
            if tens == 1:
                return 10 + ones
            return tens * 10 + ones
        return None

    def _format_route_line(self, route: dict[str, Any]) -> str:
        origin = route.get("origin", "起点")
        destination = route.get("destination", "终点")
        distance = route.get("distance_km")
        duration = route.get("duration_minutes")
        extras = []
        if distance is not None:
            extras.append(f"约{distance}公里")
        if duration is not None:
            extras.append(f"预计{duration}分钟")
        if not extras:
            extras.append("已尝试调用地图接口，但暂时未拿到距离和耗时")
        return f"{origin} → {destination}，{'，'.join(extras)}"

    def _safe_int(self, value: Any) -> int | None:
        try:
            if value is None or value == "":
                return None
            return int(float(value))
        except (TypeError, ValueError):
            return None

    # ===== 修改开始 =====
    def _normalize_location_text(self, text: str) -> str:
        import re

        cleaned = text.strip()
        cleaned = cleaned.replace("的路线", "").replace("的行程", "").replace("的旅行", "")
        cleaned = cleaned.replace("路线", "").replace("行程", "").replace("旅行", "")
        cleaned = cleaned.replace("目的地", "").replace("出发地", "")
        cleaned = re.sub(r"^(帮我|请帮我|麻烦你|给我|想让你|规划一下|规划|安排一下|安排)", "", cleaned).strip()
        suffixes = [
            "行程安排", "旅游安排", "出行安排", "行程规划", "旅游规划", "出行规划", "攻略推荐",
            "攻略", "玩法", "推荐", "安排", "规划", "行程", "路线", "旅程", "旅行", "旅游",
            "玩", "出行", "一下", "呢", "吧", "呀", "啊", "去", "到",
        ]
        changed = True
        while changed:
            changed = False
            for suffix in suffixes:
                if cleaned.endswith(suffix) and len(cleaned) > len(suffix):
                    cleaned = cleaned[: -len(suffix)]
                    changed = True
        return cleaned.strip(" 的，。！？\t\n\r")

    def _looks_like_invalid_location(self, text: str, field_type: str = "destination") -> bool:
        import re

        candidate = str(text or "").strip()
        if not candidate:
            return True
        if self._looks_like_date_text(candidate):
            return True
        if re.search(r"\d", candidate):
            return True
        invalid_tokens = [
            "行程", "安排", "规划", "攻略", "玩法", "推荐", "预算", "喜欢", "自然风景",
            "自然景观", "几天", "天", "晚", "个人", "人", "详细", "一点",
        ]
        if field_type != "address" and any(token == candidate for token in invalid_tokens):
            return True
        if field_type != "address" and any(candidate.endswith(token) for token in ["安排", "规划", "攻略", "玩法", "推荐"]):
            return True
        if len(candidate) < 2 or len(candidate) > 12:
            return True
        return False

    def _sanitize_location_candidate(self, text: str | None, field_type: str = "destination") -> str | None:
        if text is None:
            return None
        cleaned = self._normalize_location_text(str(text))
        if not cleaned:
            return None
        if self._looks_like_invalid_location(cleaned, field_type=field_type):
            cities = self._extract_all_cities(cleaned)
            if field_type == "destination" and cities:
                fallback = cities[-1]
                normalized_fallback = self._normalize_location_text(fallback)
                if normalized_fallback and not self._looks_like_invalid_location(normalized_fallback, field_type=field_type):
                    return normalized_fallback
            if field_type == "address" and cities:
                fallback = cities[0]
                normalized_fallback = self._normalize_location_text(fallback)
                if normalized_fallback and not self._looks_like_invalid_location(normalized_fallback, field_type=field_type):
                    return normalized_fallback
            return None
        return cleaned

    def _normalize_date_token(self, token: str) -> str:
        normalized = str(token or "").strip().replace("日", "号")
        if "/" in normalized or "." in normalized:
            normalized = normalized.replace("/", "月").replace(".", "月") + "号"
        elif normalized.endswith("月"):
            normalized += "1号"
        elif normalized.endswith("号"):
            return normalized
        elif "月" in normalized:
            normalized += "号"
        return normalized
    # ===== 修改结束 =====

    # ===== 修改开始 =====
    def _extract_budget(self, message: str) -> int | None:
        import re

        normalized = self._sanitize_user_message(message)
        if not normalized:
            return None

        explicit_patterns = [
            r"(?:预算|总预算|花费预算|旅行预算|旅费|费用预算)\s*[:：]?\s*([0-9]+(?:\.[0-9]+)?)\s*(?:元|块|人民币|rmb|RMB)?",
            r"([0-9]+(?:\.[0-9]+)?)\s*(?:元|块|人民币|rmb|RMB)\s*(?:预算|左右|以内|以下|上下)?",
            r"(?:预算|总预算|花费预算|旅行预算|旅费|费用预算).{0,8}?([0-9]+(?:\.[0-9]+)?)",
        ]
        for pattern in explicit_patterns:
            match = re.search(pattern, normalized)
            if match:
                value = float(match.group(1))
                if value >= 100:
                    return int(value)

        compact_match = re.search(r"预算\s*([0-9]{3,})", normalized.replace(" ", ""))
        if compact_match:
            return int(compact_match.group(1))

        compact = normalized.replace(" ", "")
        if compact in {"预算", "总预算", "费用预算", "旅行预算", "花费预算"}:
            return None
        return None
    # ===== 修改结束 =====

    # ===== 修改开始 =====
    def _extract_group_size(self, message: str) -> int | None:
        import re

        normalized = self._sanitize_user_message(message)
        if not normalized:
            return None

        digit_patterns = [
            r"(?:我们|咱们|一共|共|总共|同行|出行|去玩|旅游|旅行)?\s*([0-9]+)\s*(?:个人|人)",
            r"([0-9]+)\s*(?:个人|人)(?:同行|出行|一起)?",
        ]
        for pattern in digit_patterns:
            match = re.search(pattern, normalized)
            if match:
                return max(1, int(match.group(1)))

        chinese_patterns = [
            r"(?:我们|咱们|一共|共|总共|同行|出行|去玩|旅游|旅行)?\s*([一二两三四五六七八九十]+)\s*(?:个人|人)",
            r"([一二两三四五六七八九十]+)\s*(?:个人|人)(?:同行|出行|一起)?",
        ]
        for pattern in chinese_patterns:
            match = re.search(pattern, normalized)
            if match:
                count = self._chinese_day_to_int(match.group(1))
                if count is not None:
                    return max(1, count)

        if any(keyword in normalized for keyword in ["情侣", "两口子"]):
            return 2
        if "一家三口" in normalized:
            return 3
        return None
    # ===== 修改结束 =====

    def _extract_group_size_from_history(self, history: list[tuple[str, str]]) -> int | None:
        for role, content in reversed(history):
            if role != "user":
                continue
            group_size = self._extract_group_size(content)
            if group_size is not None:
                return group_size
        return None

    def _extract_budget_from_history(self, history: list[tuple[str, str]]) -> int | None:
        for role, content in reversed(history):
            if role != "user":
                continue
            budget = self._extract_budget(content)
            if budget is not None:
                return budget
        return None

    def _build_xiaohongshu_keyword(self, profile: dict[str, Any], fallback_message: str = "") -> str:
        city = str(profile.get("destination") or "").strip()
        days = profile.get("days")
        preferences = profile.get("preferences") or []

        normalized_preferences: list[str] = []
        for item in preferences[:2] if isinstance(preferences, list) else []:
            text = str(item).strip()
            if not text:
                continue
            if text == "自然景观":
                normalized_preferences.extend(["自然风景", "公园湿地"])
            elif text not in normalized_preferences:
                normalized_preferences.append(text)

        if city and normalized_preferences:
            return f"{city} {' '.join(normalized_preferences[:2])} 攻略"
        if city and isinstance(days, int) and days > 0:
            return f"{city} {days}日游 攻略"
        if city:
            return f"{city} 旅游攻略"
        sanitized = self._sanitize_user_message(fallback_message)
        return sanitized or "旅游攻略"

    # ===== 修改开始 =====
    def _build_workflow_plan(
        self,
        message: str,
        history: list[tuple[str, str]],
        profile: dict[str, Any],
        resolved_intent: dict[str, Any],
    ) -> dict[str, Any]:
        del history
        intent = str(resolved_intent.get("intent") or "")
        explicit_city = self._extract_city(message)
        trip_destination = str(profile.get("destination") or explicit_city or "").strip()
        address = str(profile.get("address") or self._extract_location_address(message) or "").strip()
        origin = str(profile.get("origin") or "").strip()
        days = profile.get("days")
        budget = profile.get("budget")
        preferences = profile.get("preferences") or []

        if intent in {"trip_plan", "detailed_trip_plan", "transport_advice"}:
            city = trip_destination
            tool_calls: list[dict[str, Any]] = []
            missing_info: list[str] = []
            departure_day_offset = profile.get("departure_day_offset") or 0
            departure_date_label = str(profile.get("departure_date_label") or "")
            effective_budget = budget if isinstance(budget, int) and budget > 0 else None
            effective_group_size = profile.get("group_size") if isinstance(profile.get("group_size"), int) and profile.get("group_size") > 0 else None
            effective_preferences = preferences if isinstance(preferences, list) and preferences else []
            is_transport_only = intent == "transport_advice"

            if not origin:
                missing_info.append("出发地")
            if not trip_destination:
                missing_info.append("目的地")
            if not (isinstance(days, int) and days > 0):
                missing_info.append("游玩天数")
            if not (isinstance(budget, int) and budget > 0):
                missing_info.append("预算")
            if not effective_group_size:
                missing_info.append("人数")
            if not departure_date_label and departure_day_offset == 0 and not profile.get("travel_date_range"):
                missing_info.append("出发日期")
            if not (isinstance(preferences, list) and preferences) and not is_transport_only:
                missing_info.append("偏好")

            if origin and trip_destination:
                tool_calls.append(
                    {"tool": "route_plan", "args": {"origin": origin, "destination": trip_destination, "mode": "driving"}}
                )
            if trip_destination and not is_transport_only:
                tool_calls.append(
                    {
                        "tool": "weather",
                        "args": {"city": trip_destination, "day_offset": departure_day_offset, "date_label": departure_date_label},
                    }
                )
                tool_calls.append({"tool": "hotel_search", "args": {"city": trip_destination, "budget": effective_budget}})
                tool_calls.append(
                    {
                        "tool": "xiaohongshu_search",
                        "args": {"keyword": self._build_xiaohongshu_keyword(profile, fallback_message=message), "limit": 5},
                    }
                )
                tool_calls.append({"tool": "place_search", "args": {"keyword": "餐厅", "city": trip_destination}})
                tool_calls.append({"tool": "place_search", "args": {"keyword": "美食", "city": trip_destination}})
                for keyword in self._derive_place_keywords(effective_preferences, message)[:3]:
                    tool_calls.append({"tool": "place_search", "args": {"keyword": keyword, "city": trip_destination}})

            task = intent if intent in {"trip_plan", "detailed_trip_plan", "transport_advice"} else "trip_plan"
            return {
                "task": task,
                "need_more_info": bool(missing_info),
                "missing_info": missing_info,
                "reasoning_brief": "先按旅游规划意图提取出发地、目的地、日期、预算、人数和偏好，再仅用通过校验的状态字段调用地图、天气、酒店和攻略工具。",
                "tool_calls": tool_calls,
                "response_goal": (
                    f"围绕{city or '目的地待补充'}给出怎么去、住哪里、去哪玩、怎么玩、注意什么的具体建议；"
                    f"若仍缺{('、'.join(missing_info) if missing_info else '无')}，回答末尾必须列出这些缺失项。"
                ),
            }

        if intent == "weather":
            weather_address = address or trip_destination or settings.default_city
            weather_day_offset = profile.get("departure_day_offset") or 0
            weather_date_label = str(profile.get("weather_time") or profile.get("departure_date_label") or "")
            return {
                "task": "weather",
                "need_more_info": False,
                "missing_info": [],
                "reasoning_brief": "天气类问题只读取地址或目的地字段，再调用天气工具。",
                "tool_calls": [{"tool": "weather", "args": {"city": weather_address, "day_offset": weather_day_offset, "date_label": weather_date_label}}],
                "response_goal": "直接回答天气情况，并补充适合出行的建议。",
            }

        if intent in {"food_search", "restaurant_recommendation"}:
            query_address = address or trip_destination or explicit_city or settings.default_city
            return {
                "task": "restaurant_recommendation",
                "need_more_info": False,
                "missing_info": [],
                "reasoning_brief": "餐厅推荐只读取地址字段；如果没有明确地址，再回退到目的地，再调用地图 POI 和小红书。",
                "tool_calls": [
                    {"tool": "place_search", "args": {"keyword": "餐厅", "city": query_address}},
                    {"tool": "place_search", "args": {"keyword": "美食", "city": query_address}},
                    {
                        "tool": "xiaohongshu_search",
                        "args": {"keyword": f"{query_address} 餐厅 美食 攻略", "limit": 3},
                    },
                ],
                "response_goal": "围绕用户当前地址或指定地址给出真实餐厅候选，并结合口味偏好继续细化。",
            }

        if intent == "route" and origin and trip_destination:
            return {
                "task": "route",
                "need_more_info": False,
                "missing_info": [],
                "reasoning_brief": "路线问题只读取出发地和目的地字段，再调用路线工具。",
                "tool_calls": [{"tool": "route_plan", "args": {"origin": origin, "destination": trip_destination, "mode": "driving"}}],
                "response_goal": "给出路线结论、基础距离耗时和如何选交通方式。",
            }

        if intent in {"hotel", "hotel_search"}:
            hotel_address = address or trip_destination or explicit_city or settings.default_city
            return {
                "task": "hotel_search",
                "need_more_info": False,
                "missing_info": [],
                "reasoning_brief": "酒店推荐优先读取地址字段，没有地址时再回退到目的地。",
                "tool_calls": [{"tool": "hotel_search", "args": {"city": hotel_address, "budget": budget if isinstance(budget, int) and budget > 0 else None}}],
                "response_goal": "给出可住区域、酒店候选和选择建议。",
            }

        if intent == "place_search":
            place_address = address or trip_destination or explicit_city or settings.default_city
            place_keywords = self._derive_place_keywords(preferences, message)[:2] or ["景点"]
            return {
                "task": "place_search",
                "need_more_info": False,
                "missing_info": [],
                "reasoning_brief": "景点推荐优先读取地址字段，没有地址时再回退到目的地。",
                "tool_calls": [{"tool": "place_search", "args": {"keyword": keyword, "city": place_address}} for keyword in place_keywords],
                "response_goal": "给出景点候选、适合的玩法方向和继续细化建议。",
            }

        return {
            "task": intent or "chat",
            "need_more_info": False,
            "missing_info": [],
            "reasoning_brief": "当前问题暂不需要工具，直接基于状态和历史回答。",
            "tool_calls": [],
            "response_goal": "结合当前状态直接回答用户问题。",
        }
    # ===== 修改结束 =====

    def _derive_place_keywords(self, preferences: list[Any], message: str) -> list[str]:
        keywords: list[str] = []
        if isinstance(preferences, list):
            preference_text = " ".join(str(item) for item in preferences)
        else:
            preference_text = ""
        combined = f"{preference_text} {message}"

        mapping = [
            (["美食", "吃", "夜宵"], ["美食", "小吃街", "夜市"]),
            (["博物馆", "历史建筑", "人文", "古建", "文化"], ["博物馆", "艺术馆", "古镇"]),
            (["自然", "自然景观", "风景", "山水", "徒步", "森林", "湖", "绿道"], ["自然景点", "山水风景", "湿地公园", "公园", "植物园"]),
            (["拍照", "出片", "夜景"], ["自然景点", "夜景", "街区"]),
            (["亲子", "带娃", "小朋友"], ["亲子乐园", "动物园", "海洋馆"]),
            (["休闲", "慢游", "逛逛"], ["公园", "街区", "商圈"]),
        ]
        for tokens, candidates in mapping:
            if any(token in combined for token in tokens):
                for candidate in candidates:
                    if candidate not in keywords:
                        keywords.append(candidate)
        if not keywords:
            keywords = ["景点", "公园"]
        if "自然" in combined or "风景" in combined:
            for candidate in ["自然景点", "山水风景", "湿地公园", "公园", "植物园"]:
                if candidate not in keywords:
                    keywords.append(candidate)
        return keywords[:5]

    def _build_weather_hints(self, result: dict[str, Any]) -> list[str]:
        city = result.get("city") or "目的地"
        forecast = result.get("forecast") or "天气未知"
        low = result.get("temperature_min")
        high = result.get("temperature_max")
        recommendation = result.get("recommendation") or ""
        indoor_bias = bool(result.get("indoor_bias"))
        mode_hint = str(result.get("travel_mode_hint") or "balanced")
        date_label = str(result.get("date_label") or "当天")
        temp_text = f"{low}~{high}°C" if low is not None and high is not None else "温度信息待确认"

        hints = [f"{city}{date_label}天气：{forecast}，{temp_text}。{recommendation}"]
        if indoor_bias:
            hints.append("当天更适合室内优先：博物馆、美术馆、商圈、室内展馆可作为主行程，户外景点放到天气较稳定时段或作为机动项。")
        elif mode_hint == "avoid_midday_outdoor":
            hints.append("白天偏热，核心户外行程应尽量前置到上午或后移到傍晚，中午安排室内休息或用餐。")
        else:
            hints.append("天气相对稳定，可按常规安排户外主景点，同时保留1个室内备选点。")

        daily_forecasts = result.get("daily_forecasts") or []
        if isinstance(daily_forecasts, list) and len(daily_forecasts) >= 2:
            summary_parts: list[str] = []
            for idx, item in enumerate(daily_forecasts[:3]):
                if not isinstance(item, dict):
                    continue
                day_info = item.get("day") or {}
                weather = day_info.get("weather") or "天气待确认"
                day_temp = self._safe_int(day_info.get("temperature"))
                summary_parts.append(
                    f"第{idx + 1}天{weather}{f' {day_temp}°C' if day_temp is not None else ''}"
                )
            if summary_parts:
                hints.append("未来几天节奏参考：" + "；".join(summary_parts))
        return hints

    def _hotel_search_budget_cap(self, total_budget: Any, days: Any, group_size: Any) -> int | None:
        if not isinstance(total_budget, int) or total_budget <= 0:
            return None
        normalized_days = days if isinstance(days, int) and days > 1 else settings.default_trip_days
        nights = max(normalized_days - 1, 1)
        traveler_count = group_size if isinstance(group_size, int) and group_size > 0 else 2
        per_person_daily = total_budget / traveler_count / normalized_days
        if per_person_daily <= 350:
            ratio = 0.22
        elif per_person_daily <= 600:
            ratio = 0.26
        else:
            ratio = 0.32
        return max(220, int(total_budget * ratio / nights))

    def _build_place_hints(self, city: str, keyword: str, places: list[dict[str, Any]]) -> list[str]:
        if not places:
            return [f"{city}未检索到可用的{keyword}候选地点。"]
        names = []
        for place in places[:4]:
            name = str(place.get("name") or "").strip()
            address = str(place.get("address") or "").strip()
            if name:
                names.append(f"{name}（{address or '地址待确认'}）")
        return [f"{city}{keyword}候选：" + "、".join(names)] if names else []

    def _build_hotel_hints(self, city: str, hotels: list[dict[str, Any]], budget: Any) -> list[str]:
        if not hotels:
            return [f"{city}暂无合适酒店候选。"]
        hints = []
        top_hotels = []
        for hotel in hotels[:3]:
            name = hotel.get("name") or "酒店"
            price = hotel.get("price")
            area = hotel.get("area") or hotel.get("location") or "区域待确认"
            rating = hotel.get("rating")
            distance_hint = hotel.get("distance_hint") or ""
            price_text = f"约{price}元" if price is not None else "价格待确认"
            rating_text = f"评分{rating}" if rating is not None else "评分待确认"
            extra_text = f"，{distance_hint}" if distance_hint else ""
            top_hotels.append(f"{name}（{area}，{price_text}，{rating_text}{extra_text}）")
        hints.append("住宿候选：" + "、".join(top_hotels))
        if isinstance(budget, int) and budget > 0:
            hints.append(f"当前住宿筛选参考预算：{budget}元/晚以内优先。")
        return hints

    def _build_xiaohongshu_hints(self, result: dict[str, Any]) -> list[str]:
        hints = []
        insights = result.get("insights") or []
        if isinstance(insights, list):
            hints.extend(str(item) for item in insights[:3] if item)
        notes = result.get("notes") or []
        if isinstance(notes, list) and notes:
            top_titles = [str(note.get("title") or "").strip() for note in notes[:3] if note.get("title")]
            if top_titles:
                hints.append("小红书高参考笔记：" + "、".join(top_titles))
        return hints

    def _build_route_hints(self, route: dict[str, Any], profile: dict[str, Any]) -> list[str]:
        origin = route.get("origin") or profile.get("origin") or "出发地"
        destination = route.get("destination") or profile.get("destination") or "目的地"
        distance = route.get("distance_km")
        duration = route.get("duration_minutes")
        group_size = profile.get("group_size")
        budget = profile.get("budget")
        days = profile.get("days")

        route_desc = self._format_route_line(route) if route else f"{origin} → {destination}"
        hints = [f"路线基础信息：{route_desc}。"]
        if distance is not None and duration is not None:
            if distance >= 1200:
                hints.append("距离较远，优先比较飞机和高铁；只有特别强调沿途自由度时才考虑自驾。")
            elif distance >= 500:
                hints.append("中长距离行程通常优先高铁，自驾适合多人分摊和沿途停靠需求更强的情况。")
            else:
                hints.append("中短距离可重点比较高铁和自驾：高铁更省心，自驾更灵活。")
        if isinstance(group_size, int) and group_size >= 4:
            hints.append("多人同行时，自驾的人均交通成本可能更有优势，但要综合疲劳和停车成本。")
        if isinstance(budget, int) and budget > 0 and isinstance(days, int) and days <= 3:
            hints.append("短天数行程更应优先节省路上时间，通常高铁或飞机比长时间自驾更合适。")
        return hints

    def _infer_weather_mode(self, tool_results: list[dict[str, Any]]) -> str:
        for item in tool_results:
            if item.get("source") != "weather_tool":
                continue
            result = item.get("result") or {}
            if not isinstance(result, dict):
                continue
            mode = str(result.get("travel_mode_hint") or "").strip()
            if mode:
                return mode
        return "balanced"

    def _build_orchestration_hints(self, tool_results: list[dict[str, Any]], profile: dict[str, Any], trip_blueprint: dict[str, Any] | None = None) -> str:
        city = str(profile.get("destination") or settings.default_city)
        days = profile.get("days") or settings.default_trip_days
        hint_lines = [f"目的地：{city}；优先按 {days} 天节奏组织行程。"]

        missing = self._missing_trip_fields(profile)
        if missing:
            hint_lines.append(
                f"当前仍缺关键信息：{'、'.join(missing)}。"
                "回答末尾必须明确提示用户补充这些项，并说明补充后能把交通、天气、住宿和行程排得更准；"
                "不要只引导‘更详细时间表’而不提缺失项。"
            )

        route_hints: list[str] = []
        weather_hints: list[str] = []
        place_hints: list[str] = []
        hotel_hints: list[str] = []
        xhs_hints: list[str] = []

        for item in tool_results:
            tool_name = item.get("tool")
            source = item.get("source")
            hints = item.get("planning_hints") or []
            if not isinstance(hints, list):
                continue
            if tool_name == "route_plan":
                route_hints.extend(str(h) for h in hints if h)
            elif source == "weather_tool":
                weather_hints.extend(str(h) for h in hints if h)
            elif tool_name == "place_search":
                place_hints.extend(str(h) for h in hints if h)
            elif source == "hotel_search_tool":
                hotel_hints.extend(str(h) for h in hints if h)
            elif source == "xiaohongshu_tool":
                xhs_hints.extend(str(h) for h in hints if h)

        weather_mode = self._infer_weather_mode(tool_results)
        blueprint = trip_blueprint or {}
        blueprint_daily = blueprint.get("daily_plan") or []
        blueprint_place_summary = str(blueprint.get("place_summary") or "").strip()
        blueprint_hotel_summary = str(blueprint.get("hotel_summary") or "").strip()
        if blueprint_place_summary:
            hint_lines.append("筛选后的玩法池：" + blueprint_place_summary)
        if blueprint_hotel_summary:
            hint_lines.append("筛选后的住宿策略：" + blueprint_hotel_summary)
        if isinstance(blueprint_daily, list) and blueprint_daily:
            first_day = blueprint_daily[0] if isinstance(blueprint_daily[0], dict) else {}
            first_theme = str(first_day.get("theme") or "").strip()
            if first_theme:
                hint_lines.append(f"按天规划要求：先给整体结论，再解释为什么第1天从“{first_theme}”主题切入，后续每天保持节奏递进。")

        if route_hints:
            hint_lines.append("路线建议：" + " | ".join(route_hints[:3]))
        if weather_hints:
            hint_lines.append("天气约束：" + " | ".join(weather_hints[:3]))
        if place_hints:
            hint_lines.append("景点/美食候选：" + " | ".join(place_hints[:3]))
        if hotel_hints:
            hint_lines.append("住宿落点参考：" + " | ".join(hotel_hints[:2]))
        if xhs_hints:
            hint_lines.append("小红书经验参考：" + " | ".join(xhs_hints[:3]))

        hint_lines.append("回答组织优先顺序：先给整体结论，再分别整合路线、天气、酒店、小红书经验，最后补预算建议、避坑提醒和可继续细化的信息。")
        hint_lines.append("最终答案尽量写得像真实攻略：每一天有主题、有时间段、有商圈/馆名/街区名，有为什么这样排。")

        if weather_mode == "prefer_indoor":
            hint_lines.append("行程编排强约束：下雨/强对流天气时，优先把博物馆、美术馆、室内展馆、大型商圈、特色餐饮街区安排为主行程；户外景点降级为机动项或缩短停留时间。")
            hint_lines.append("如果必须保留户外景点，应放在降雨较弱时段，并同步给出雨天替代方案。")
        elif weather_mode == "avoid_midday_outdoor":
            hint_lines.append("行程编排强约束：中午高温时段减少暴晒型户外活动，把室内体验、午餐和休整放在正午前后。")
        else:
            hint_lines.append("行程编排强约束：天气正常时，以核心户外景点为主，补充1个同片区室内备选点。")

        hint_lines.append("如果信息有限，也要先给可执行方案，再明确说明补充哪些信息后可以把路线、天气、酒店和攻略进一步细化。")
        return "\n".join(hint_lines)
