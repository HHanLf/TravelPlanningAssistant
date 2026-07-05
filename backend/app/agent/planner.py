from __future__ import annotations

from backend.app.domain.models import AgentContext, ExecutionPlan, ExecutionStage, ExecutionStep, ToolCall


class Planner:
    def create_plan(self, context: AgentContext, intent: dict) -> ExecutionPlan:
        destination = intent.get("destination") or context.user_profile.destination
        preferences = intent.get("preferences") or context.user_profile.preferences
        preferred_tools = intent.get("preferred_tools", [])
        missing_information = list(intent.get("missing_information", []))

        tool_calls: list[ToolCall] = []
        if destination:
            for tool_name in preferred_tools:
                arguments = self._build_arguments(tool_name, destination, preferences)
                tool_calls.append(
                    ToolCall(
                        name=tool_name,
                        arguments=arguments,
                        reason=self._tool_reason(tool_name),
                    )
                )

        return ExecutionPlan(
            intent=intent,
            missing_information=missing_information,
            tool_calls=tool_calls,
            steps=self._build_steps(intent, destination, bool(tool_calls), missing_information),
        )

    @staticmethod
    def _build_arguments(tool_name: str, destination: str, preferences: list[str]) -> dict:
        if tool_name == "poi_recommendation":
            return {"destination": destination, "preferences": preferences}
        return {"destination": destination}

    @staticmethod
    def _tool_reason(tool_name: str) -> str:
        reasons = {
            "weather_lookup": "天气会直接影响出行安排与穿衣建议",
            "route_planning": "需要基于城市动线生成可执行路线",
            "poi_recommendation": "需要根据用户偏好筛选景点与美食",
            "hotel_search": "需要补充住宿候选项供用户决策",
        }
        return reasons.get(tool_name, "需要调用专业工具补充信息")

    @staticmethod
    def _build_steps(
        intent: dict,
        destination: str | None,
        has_tool_calls: bool,
        missing_information: list[str],
    ) -> list[ExecutionStep]:
        details = []
        details.append(
            ExecutionStep(
                stage=ExecutionStage.UNDERSTAND,
                title="理解用户需求",
                detail=f"识别当前任务为 {intent.get('type', 'travel_planning')}。",
            )
        )

        if missing_information:
            details.append(
                ExecutionStep(
                    stage=ExecutionStage.PLAN,
                    title="识别缺失信息",
                    detail=f"当前仍缺少: {', '.join(missing_information)}。",
                )
            )
        else:
            details.append(
                ExecutionStep(
                    stage=ExecutionStage.PLAN,
                    title="生成执行计划",
                    detail=f"已锁定目的地 {destination}，进入工具编排阶段。",
                )
            )

        if has_tool_calls:
            details.append(
                ExecutionStep(
                    stage=ExecutionStage.EXECUTE,
                    title="调用外部工具",
                    detail="按意图选择工具，避免让 LLM 直接依赖具体实现。",
                )
            )
        else:
            details.append(
                ExecutionStep(
                    stage=ExecutionStage.EXECUTE,
                    title="跳过工具调用",
                    detail="当前信息不足或无需工具，先返回澄清或结构化建议。",
                )
            )

        details.append(
            ExecutionStep(
                stage=ExecutionStage.RESPOND,
                title="构建最终回复",
                detail="汇总上下文、工具结果与后续行动建议。",
            )
        )
        return details
