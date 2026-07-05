from __future__ import annotations

from abc import ABC
from typing import Any

from backend.app.domain.models import AgentContext, ToolResult, ToolSpec


class BaseTool(ABC):
    spec: ToolSpec

    async def execute(self, context: AgentContext, arguments: dict[str, Any]) -> ToolResult:
        del context
        runner = getattr(self, "run", None)
        if not callable(runner):
            return ToolResult(
                name=getattr(self, "name", self.__class__.__name__),
                success=False,
                payload={},
                error="tool does not implement run() or execute()",
            )

        try:
            result = runner(**arguments)
            if isinstance(result, ToolResult):
                return result
            payload = result if isinstance(result, dict) else {"result": result}
            return ToolResult(
                name=getattr(self, "name", self.__class__.__name__),
                success=True,
                payload=payload,
                error=None,
            )
        except Exception as exc:  # noqa: BLE001
            return ToolResult(
                name=getattr(self, "name", self.__class__.__name__),
                success=False,
                payload={},
                error=str(exc),
            )

    def validate(self, arguments: dict[str, Any]) -> list[str]:
        spec = getattr(self, "spec", None)
        required_fields = getattr(spec, "required_fields", []) if spec is not None else []
        missing: list[str] = []
        for field_name in required_fields:
            if not arguments.get(field_name):
                missing.append(field_name)
        return missing
