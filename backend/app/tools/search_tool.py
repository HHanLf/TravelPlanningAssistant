from __future__ import annotations

from typing import Any

from app.tools.base import BaseTool


class SearchTool(BaseTool):
    name = "search"

    def __init__(self, api_key: str = "", base_url: str = "") -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")

    def search(self, query: str) -> list[dict[str, str]]:
        return self.run(query=query)

    def run(self, **kwargs: Any) -> list[dict[str, str]]:
        query = str(kwargs.get("query", "")).strip()
        if not query:
            return []

        return [
            {
                "title": f"{query} 结果 1",
                "url": "https://example.com/1",
                "snippet": "示例搜索结果。",
            },
            {
                "title": f"{query} 结果 2",
                "url": "https://example.com/2",
                "snippet": "示例搜索结果。",
            },
        ]
