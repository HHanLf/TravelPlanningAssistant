from __future__ import annotations


class SearchTool:
    def __init__(self, api_key: str) -> None:
        self.api_key = api_key

    def search(self, query: str) -> list[dict]:
        return [
            {"title": f"关于 {query} 的最新信息", "snippet": "这是模拟联网搜索结果。"},
            {"title": f"{query} 参考攻略", "snippet": "可根据实时搜索接入真实服务。"},
        ]
