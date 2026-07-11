from __future__ import annotations

from typing import Any

from app.tools.base import BaseTool


class PythonTool(BaseTool):
    name = "python"

    def run(self, **kwargs: Any) -> dict[str, Any]:
        code = kwargs.get("code", "")
        return {"code": code, "result": "Python 执行工具示例结果"}
