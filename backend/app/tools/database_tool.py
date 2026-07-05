from __future__ import annotations

from typing import Any

from backend.app.tools.base import BaseTool


class DatabaseTool(BaseTool):
    name = "database"

    def run(self, **kwargs: Any) -> dict[str, Any]:
        query = kwargs.get("query", "")
        return {"query": query, "rows": []}
