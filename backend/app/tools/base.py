from __future__ import annotations

from abc import ABC
from typing import Any

from app.agent.state import AgentState, ToolResult
from app.domain.models import ToolSpec


class BaseTool(ABC):
    spec: ToolSpec

    async def execute(self, state: AgentState, arguments: dict[str, Any]) -> ToolResult:
        del state
        runner = getattr(self, "run", None)
        if not callable(runner):
            return ToolResult(
                name=self.spec.name,
                arguments=arguments,
                success=False,
                error="tool does not implement run()",
            )

        try:
            result = runner(**arguments)
            payload = result if isinstance(result, dict) else {"result": result}
            return ToolResult(
                name=self.spec.name,
                arguments=arguments,
                success=True,
                payload=payload,
                summary=self.summarize(payload),
            )
        except Exception as exc:  # noqa: BLE001
            return ToolResult(
                name=self.spec.name,
                arguments=arguments,
                success=False,
                error=str(exc),
            )

    def validate(self, arguments: dict[str, Any]) -> list[str]:
        missing: list[str] = []
        for field_name in getattr(self.spec, "required_fields", []):
            if arguments.get(field_name) in (None, "", []):
                missing.append(field_name)
        return missing

    def summarize(self, payload: dict[str, Any]) -> str:
        if payload.get("summary"):
            return str(payload["summary"])
        if payload.get("error"):
            return str(payload["error"])
        compact = {key: value for key, value in payload.items() if key != "raw"}
        return str(compact)[:240]
