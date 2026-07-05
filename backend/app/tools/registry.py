from __future__ import annotations

from collections.abc import Iterable

from backend.app.tools.base import BaseTool


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, BaseTool] = {}

    def register(self, tool: BaseTool) -> None:
        self._tools[tool.spec.name] = tool

    def register_many(self, tools: Iterable[BaseTool]) -> None:
        for tool in tools:
            self.register(tool)

    def get(self, name: str) -> BaseTool:
        return self._tools[name]

    def has(self, name: str) -> bool:
        return name in self._tools

    def list_specs(self) -> list[dict]:
        return [
            {
                "name": tool.spec.name,
                "description": tool.spec.description,
                "category": tool.spec.category.value,
                "required_fields": tool.spec.required_fields,
                "tags": tool.spec.tags,
            }
            for tool in self._tools.values()
        ]
