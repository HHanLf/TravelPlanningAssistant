from __future__ import annotations

from app.agent.state import AgentState, ToolCall, ToolResult
from app.tools.registry import ToolRegistry


class ToolExecutor:
    def __init__(self, registry: ToolRegistry) -> None:
        self._registry = registry

    async def execute_many(self, state: AgentState, tool_calls: list[ToolCall]) -> list[ToolResult]:
        results: list[ToolResult] = []
        for call in tool_calls:
            if not self._registry.has(call.name):
                results.append(
                    ToolResult(
                        name=call.name,
                        arguments=call.arguments,
                        success=False,
                        error=f"工具未注册：{call.name}",
                    )
                )
                continue

            tool = self._registry.get(call.name)
            missing_fields = tool.validate(call.arguments)
            if missing_fields:
                results.append(
                    ToolResult(
                        name=call.name,
                        arguments=call.arguments,
                        success=False,
                        error=f"缺少必要参数：{', '.join(missing_fields)}",
                    )
                )
                continue

            try:
                result = await tool.execute(state, call.arguments)
            except Exception as exc:  # noqa: BLE001
                result = ToolResult(
                    name=call.name,
                    arguments=call.arguments,
                    success=False,
                    error=str(exc),
                )
            if not result.arguments:
                result.arguments = call.arguments
            results.append(result)
        return results
