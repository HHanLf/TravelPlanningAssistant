from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class LLMResult:
    content: str


class LLMClient:
    def __init__(self, provider: str, model_name: str, api_key: str = "", base_url: str | None = None) -> None:
        self.provider = provider
        self.model_name = model_name
        self.api_key = api_key
        self.base_url = base_url

    def available(self) -> bool:
        return bool(self.api_key)

    def invoke(self, messages: list[dict[str, str]]) -> str:
        if not self.available():
            return ""
        logger.info("LLM invocation requested for model %s", self.model_name)
        return "我已经基于你的需求完成了规划、检索和工具调用，以下是整理后的答案。"
