from __future__ import annotations

from backend.app.domain.models import AgentContext, ToolCall, ToolResult
from backend.app.tools.registry import ToolRegistry


class ToolExecutor:
    def __init__(self, registry: ToolRegistry) -> None:
        self._registry = registry

    async def execute_many(self, context: AgentContext, tool_calls: list[ToolCall]) -> list[ToolResult]:
        results: list[ToolResult] = []
        for call in tool_calls:
            if not self._registry.has(call.name):
                results.append(
                    ToolResult(
                        name=call.name,
                        success=False,
                        payload={},
                        error=f"Tool not found: {call.name}",
                    )
                )
                continue

            tool = self._registry.get(call.name)
            missing_fields = tool.validate(call.arguments)
            if missing_fields:
                results.append(
                    ToolResult(
                        name=call.name,
                        success=False,
                        payload={},
                        error=f"Missing required arguments: {', '.join(missing_fields)}",
                    )
                )
                continue

            try:
                results.append(await tool.execute(context, call.arguments))
            except Exception as exc:  # noqa: BLE001
                results.append(
                    ToolResult(
                        name=call.name,
                        success=False,
                        payload={},
                        error=str(exc),
                    )
                )
        return results
