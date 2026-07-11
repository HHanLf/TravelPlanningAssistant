from __future__ import annotations

from app.agent.state import AgentState, ReflectionResult
from app.agent.travel_semantics import get_poi_theme_profile, is_polluted_poi, score_poi_for_theme


class ReflectionAgent:
    TRIP_REQUIRED_SECTIONS = (
        "行程总览",
        "每日安排",
        "住宿",
        "美食",
        "预算",
        "注意事项",
    )
    TRIP_FORBIDDEN_TERMS = (
        "travel_theme",
        "semantic_theme",
        "tool_result",
        "tool_results",
        "payload",
        "success",
        "failed",
        "JSON",
        "filtered_out",
    )
    TIME_SLOT_KEYWORDS = ("上午", "下午", "晚上")

    def evaluate(self, state: AgentState, round_index: int = 0) -> ReflectionResult:
        answer = state.draft_answer or state.final_answer or ""
        intent_type = state.intent.type if state.intent else "chat"
        missing_info = state.problem.missing_info if state.problem else []
        constraints = state.problem.constraints if state.problem else {}
        tool_results = state.tool_results
        knowledge_summary = state.knowledge_summary or {}

        score = 40
        issues: list[str] = []
        suggestions: list[str] = []

        critical_errors = self._critical_tool_errors(state)
        if critical_errors:
            transparent = "当前无法生成可靠" in answer or "当前无法完成可靠" in answer
            names_covered = all(self._tool_label(item.name) in answer or (item.error and item.error in answer) for item in critical_errors)
            if transparent and names_covered:
                return ReflectionResult(
                    passed=True,
                    score=90,
                    issues=[],
                    suggestions=["关键工具失败已明确告知用户，已停止兜底生成。"],
                    round=round_index,
                )
            return ReflectionResult(
                passed=False,
                score=45,
                issues=["关键工具失败时没有明确返回错误原因。"],
                suggestions=["不要继续生成兜底行程；请列出失败工具、参数、错误信息和排查建议。"],
                round=round_index,
            )

        if len(answer.strip()) >= 80:
            score += 15
        else:
            issues.append("回答过短，可能没有形成可执行建议。")
            suggestions.append("补充行程节奏、关键判断和下一步问题。")

        if missing_info:
            if any(self._missing_label(item) in answer or item in answer for item in missing_info):
                score += 10
            else:
                issues.append("缺失信息没有明确告知用户。")
                suggestions.append("说明缺少哪些信息，同时给出默认假设下的初版。")
        else:
            score += 10

        if state.plan and state.plan.tool_calls:
            if tool_results:
                score += 10
                if any(item.success and (item.summary or item.payload) for item in tool_results):
                    score += 10
                elif any(item.error for item in tool_results):
                    score += 3
                    suggestions.append("工具失败时要透明说明，并给出不依赖实时数据的保守建议。")
            else:
                issues.append("计划需要工具，但回答前没有工具结果。")
                suggestions.append("重新执行工具或解释无法获取实时结果。")
        else:
            score += 8

        research_reflection = knowledge_summary.get("research_reflection") if isinstance(knowledge_summary, dict) else None
        if isinstance(research_reflection, dict):
            research_issues = [str(item) for item in research_reflection.get("issues", []) if item]
            research_suggestions = [str(item) for item in research_reflection.get("suggestions", []) if item]
            if research_issues:
                issues.extend(research_issues[:3])
                suggestions.extend(research_suggestions[:3])
            else:
                score += 8
        elif knowledge_summary:
            score += 4

        if intent_type == "trip_plan":
            template_issues = self._trip_template_issues(answer, state)
            if template_issues:
                issues.extend(template_issues)
                suggestions.append(self._trip_rewrite_instruction(state))
            else:
                score += 15

            expected = ["住宿", "美食", "预算"]
            if (state.problem and state.problem.origin) or any(item.name == "route_planning" for item in tool_results):
                expected.append("交通")
            missed = [item for item in expected if item not in answer]
            if missed:
                issues.append("综合行程遗漏关键模块：" + "、".join(missed))
                suggestions.append("补齐交通、住宿、美食、预算和注意事项。")
            else:
                score += 12
            if "天气" in answer or any(item.name == "weather_lookup" for item in tool_results):
                score += 5
            if state.problem and state.problem.budget and "预算" not in answer:
                issues.append("用户提供了预算，但回答没有说明预算控制。")
                suggestions.append("补充预算拆分，并避免明显超过用户总预算。")
        elif intent_type in {"weather", "hotel_search", "restaurant_recommendation", "place_search", "transport_advice"}:
            if self._intent_keyword(intent_type) in answer:
                score += 8

        constraint_issues = self._constraint_issues(answer, constraints)
        if constraint_issues:
            issues.extend(constraint_issues)
            suggestions.append("把用户约束前置到路线、住宿、景点和餐饮选择中。")
        elif constraints:
            score += 8

        semantic_issues = self._semantic_place_issues(answer, constraints, tool_results)
        if semantic_issues:
            issues.extend(semantic_issues)
            suggestions.append("景点结果需要先按用户偏好主题做语义过滤和重排，再用保留下来的真实旅行 POI 重新规划。")

        if state.reflection_notes:
            if any(note[:8] in answer for note in state.reflection_notes):
                score += 3

        score = min(score, 100)
        passed = score >= 85 and not issues
        if not issues and passed:
            suggestions.append("回答已覆盖核心诉求，可返回用户。")
        return ReflectionResult(
            passed=passed,
            score=score,
            issues=issues,
            suggestions=suggestions,
            round=round_index,
        )

    @classmethod
    def _trip_template_issues(cls, answer: str, state: AgentState) -> list[str]:
        issues: list[str] = []
        normalized = answer.strip()
        if not normalized:
            return ["回答为空，无法作为旅行方案返回。"]

        missing_sections = [section for section in cls.TRIP_REQUIRED_SECTIONS if section not in normalized]
        if missing_sections:
            issues.append("回答缺少模板化模块：" + "、".join(missing_sections))

        forbidden = [term for term in cls.TRIP_FORBIDDEN_TERMS if term in normalized]
        if forbidden:
            issues.append("回答暴露了内部字段或调试信息：" + "、".join(forbidden))

        problem = state.problem
        days = max(problem.days or 3, 1) if problem else 3
        for day in range(1, days + 1):
            if f"Day {day}" not in normalized and f"第 {day} 天" not in normalized and f"第{day}天" not in normalized:
                issues.append(f"回答缺少第 {day} 天的明确安排。")
                break

        missing_slots = [slot for slot in cls.TIME_SLOT_KEYWORDS if slot not in normalized]
        if missing_slots:
            issues.append("每日安排没有覆盖上午/下午/晚上三个时段。")

        if "|" not in normalized and "行程总览" in normalized:
            issues.append("行程总览缺少表格或清晰的三段式概览。")

        if problem:
            if problem.destination and problem.destination not in normalized:
                issues.append(f"回答没有明确围绕目的地 {problem.destination} 展开。")
            if problem.budget and "预算" in normalized and str(problem.budget) not in normalized:
                issues.append("用户提供了预算，但回答没有明确引用总预算。")
            for preference in problem.preferences[:3]:
                if preference and preference not in normalized and not cls._preference_covered(preference, normalized):
                    issues.append(f"回答没有明显体现用户偏好：{preference}。")
                    break

        duplicate_places = cls._duplicate_place_issue(normalized)
        if duplicate_places:
            issues.append(duplicate_places)
        return issues

    @staticmethod
    def _preference_covered(preference: str, answer: str) -> bool:
        equivalents = {
            "自然": ("自然", "山林", "湿地", "湖泊", "森林", "风景"),
            "风景": ("风景", "山林", "湿地", "湖泊", "夜景"),
            "历史": ("历史", "人文", "古迹", "文物"),
            "历史建筑": ("历史建筑", "古建筑", "古迹", "文物"),
            "美食": ("美食", "餐饮", "餐厅", "小吃"),
        }
        return any(keyword in answer for keyword in equivalents.get(preference, (preference,)))

    @staticmethod
    def _duplicate_place_issue(answer: str) -> str:
        candidates: list[str] = []
        for line in answer.splitlines():
            stripped = line.strip(" -|")
            if not stripped or len(stripped) > 40:
                continue
            if any(slot in stripped for slot in ("上午", "下午", "晚上", "Day", "第")):
                cleaned = stripped
                for token in ("上午：", "下午：", "晚上：", "上午", "下午", "晚上", "："):
                    cleaned = cleaned.replace(token, "")
                cleaned = cleaned.strip()
                if 2 <= len(cleaned) <= 20:
                    candidates.append(cleaned)
        repeated = {item for item in candidates if candidates.count(item) >= 3}
        if repeated:
            return "行程中存在明显重复点位：" + "、".join(sorted(repeated))
        return ""

    @staticmethod
    def _trip_rewrite_instruction(state: AgentState) -> str:
        problem = state.problem
        destination = problem.destination if problem else "目的地"
        days = problem.days or 3 if problem else 3
        budget = problem.budget if problem else None
        budget_text = f"总预算约 {budget} 元" if budget else "预算未明确"
        return (
            f"请按高质量旅行方案模板重写：标题写明「{destination} {days} 天」和{budget_text}；"
            "先给一句结论；必须包含行程总览表；每日按 Day 拆分，并覆盖上午、下午、晚上；"
            "每个核心景点说明亮点、适合原因和建议游玩时间；补充住宿建议、美食建议、预算参考表、注意事项和路线适合理由；"
            "不要暴露 travel_theme、semantic_theme、payload、success、JSON 等内部字段；不要只罗列工具候选。"
        )

    @staticmethod
    def _missing_label(name: str) -> str:
        return {
            "destination": "目的地",
            "days": "天数",
            "budget": "预算",
            "group_size": "人数",
        }.get(name, name)

    @staticmethod
    def _intent_keyword(intent_type: str) -> str:
        return {
            "weather": "天气",
            "hotel_search": "住宿",
            "restaurant_recommendation": "美食",
            "place_search": "景点",
            "transport_advice": "交通",
        }.get(intent_type, "")

    @staticmethod
    def _critical_tool_errors(state: AgentState) -> list:
        intent_type = state.intent.type if state.intent else ""
        if intent_type == "chat":
            return []
        critical_names = {
            "weather_lookup",
            "place_search",
            "route_planning",
            "hotel_search",
            "restaurant_recommendation",
        }
        if intent_type in {"weather", "transport_advice", "hotel_search", "restaurant_recommendation", "place_search"}:
            mapped = {
                "weather": "weather_lookup",
                "transport_advice": "route_planning",
                "hotel_search": "hotel_search",
                "restaurant_recommendation": "restaurant_recommendation",
                "place_search": "place_search",
            }.get(intent_type)
            critical_names = {mapped} if mapped else critical_names
        return [
            item
            for item in state.tool_results
            if item.name in critical_names and not item.success and item.error
        ]

    @staticmethod
    def _tool_label(name: str) -> str:
        return {
            "weather_lookup": "天气查询",
            "place_search": "地点搜索",
            "route_planning": "路线规划",
            "hotel_search": "酒店搜索",
            "restaurant_recommendation": "餐厅推荐",
        }.get(name, name)

    @staticmethod
    def _constraint_issues(answer: str, constraints: dict) -> list[str]:
        issues: list[str] = []
        if not constraints:
            return issues
        checks = {
            "transport_mode": ("公共交通", "地铁", "公交") if constraints.get("transport_mode") == "public" else ("自驾", "停车", "打车"),
            "family_friendly": ("亲子", "孩子", "儿童"),
            "low_walking": ("少走", "步行", "轻松"),
            "pace": ("慢节奏", "轻松") if constraints.get("pace") == "relaxed" else ("紧凑", "高效"),
            "hotel_near_metro": ("地铁", "交通方便"),
            "avoid_paid_attractions": ("免费", "门票", "收费"),
            "dietary_preference": (str(constraints.get("dietary_preference")),),
        }
        for key, keywords in checks.items():
            if key in {"travel_theme", "requested_region", "tool_destination", "region_resolution_note"}:
                continue
            if key not in constraints or constraints.get(key) in (False, None, ""):
                continue
            if not any(keyword and keyword in answer for keyword in keywords):
                issues.append(f"回答没有体现用户约束：{key}。")
        return issues

    @staticmethod
    def _semantic_place_issues(answer: str, constraints: dict, tool_results: list) -> list[str]:
        travel_theme = str(constraints.get("travel_theme") or "")
        profile = get_poi_theme_profile(travel_theme)
        if not profile:
            return []

        issues: list[str] = []
        if ReflectionAgent._answer_has_polluted_poi(answer, travel_theme):
            issues.append(f"{profile.label}需求中混入了语义不相关或非旅行类 POI。")

        for result in tool_results:
            if getattr(result, "name", "") != "place_search":
                continue
            payload = getattr(result, "payload", {}) or {}
            result_theme = str(payload.get("semantic_theme") or travel_theme) if isinstance(payload, dict) else travel_theme
            if result_theme != travel_theme:
                continue
            places = payload.get("places") if isinstance(payload, dict) else []
            if not isinstance(places, list):
                continue
            for place in places:
                if not isinstance(place, dict):
                    continue
                if is_polluted_poi(place, travel_theme):
                    issues.append(f"工具结果仍包含不符合{profile.label}主题的 POI，需要重新过滤。")
                    return issues
        return issues

    @staticmethod
    def _answer_has_polluted_poi(answer: str, travel_theme: str) -> bool:
        candidates: list[str] = []
        for line in answer.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith(("上午：", "下午：", "晚上：")):
                name = stripped.split("：", 1)[1].strip()
                name = name.split("，", 1)[0].split("；", 1)[0].strip()
                if name:
                    candidates.append(name)
            elif stripped.startswith("| Day"):
                cells = [cell.strip() for cell in stripped.strip("|").split("|")]
                candidates.extend(cell for cell in cells[1:] if cell and not cell.startswith("Day"))
            elif stripped.startswith(("- 第", "- Day")) and "：" in stripped:
                tail = stripped.split("：", 1)[1]
                candidates.extend(item.strip() for item in tail.split("+") if item.strip())
        return any(score_poi_for_theme({"name": candidate}, travel_theme)[0] < 0 for candidate in candidates)
