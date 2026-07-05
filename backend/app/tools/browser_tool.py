from __future__ import annotations

from typing import Any

from backend.app.tools.base import BaseTool


class BrowserTool(BaseTool):
    name = "browser"

    def run(self, **kwargs: Any) -> dict[str, Any]:
        url = kwargs.get("url", "")
        return {"url": url, "content": "浏览器工具示例结果"}
