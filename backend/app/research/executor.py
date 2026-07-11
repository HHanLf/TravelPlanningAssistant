from __future__ import annotations

import asyncio
from typing import Any

from app.agent.state import AgentState, ToolResult
from app.research.models import ResearchResult, ResearchTask
from app.tools.registry import ToolRegistry


class ResearchExecutor:
    """Executes research tasks concurrently while preserving ToolResult compatibility."""

    def __init__(self, registry: ToolRegistry, timeout_seconds: float = 45.0) -> None:
        self._registry = registry
        self._timeout_seconds = timeout_seconds

    async def execute(self, state: AgentState, tasks: list[ResearchTask]) -> list[ResearchResult]:
        if not tasks:
            return []
        ordered = sorted(tasks, key=lambda item: (-item.priority, item.id))
        results = await asyncio.gather(*(self._execute_one(state, task) for task in ordered))
        return list(results)

    async def _execute_one(self, state: AgentState, task: ResearchTask) -> ResearchResult:
        if not self._registry.has(task.tool_name):
            return ResearchResult(task=task, tool_result=self._error(task, f"tool is not registered: {task.tool_name}"))

        tool = self._registry.get(task.tool_name)
        missing_fields = tool.validate(task.arguments)
        if missing_fields:
            return ResearchResult(task=task, tool_result=self._error(task, f"missing required fields: {', '.join(missing_fields)}"))

        try:
            result = await asyncio.wait_for(tool.execute(state, task.arguments), timeout=self._timeout_seconds)
        except TimeoutError:
            result = self._error(task, f"research task timed out after {self._timeout_seconds:.0f}s")
        except Exception as exc:  # noqa: BLE001
            result = self._error(task, str(exc))

        if not result.arguments:
            result.arguments = task.arguments
        return ResearchResult(task=task, tool_result=result)

    @staticmethod
    def _error(task: ResearchTask, error: str) -> ToolResult:
        return ToolResult(
            name=task.tool_name,
            arguments=task.arguments,
            success=False,
            payload={"research_task_id": task.id, "category": task.category},
            summary=error,
            error=error,
        )

