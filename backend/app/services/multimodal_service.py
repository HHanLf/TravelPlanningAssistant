from __future__ import annotations

import base64
import mimetypes
from pathlib import Path
from typing import Any

from app.services.llm_service import DashScopeLLMService


class MultimodalService:
    def __init__(self) -> None:
        self.llm = DashScopeLLMService()

    def describe_image(self, image_path: str) -> str:
        if not self.llm.available():
            return "当前未配置大模型 API Key，无法进行真实图片识别。"

        path = Path(image_path)
        if not path.exists():
            return "图片文件不存在。"

        mime_type, _ = mimetypes.guess_type(path.name)
        if not mime_type:
            mime_type = "image/jpeg"

        encoded = base64.b64encode(path.read_bytes()).decode("utf-8")
        data_url = f"data:{mime_type};base64,{encoded}"
        messages = [
            {
                "role": "system",
                "content": "你是一个多模态旅行助手，需要识别图片中的地点、文字、景物并总结可用于旅行规划的信息。",
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "请识别这张图片中的旅行相关信息，并给出简短总结。"},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            },
        ]
        return self.llm.chat(messages)

    def transcribe_audio(self, audio_path: str) -> dict[str, Any]:
        result = self.llm.transcribe_audio(
            audio_path,
            prompt="这是一段与旅行规划相关的中文语音，请尽量准确识别目的地、出发地、预算、人数、日期和偏好。",
        )
        text = str(result.get("text") or "").strip()
        error = str(result.get("error") or "").strip()
        summary = f"语音识别结果：{text}" if text else error
        return {
            "text": text,
            "error": error,
            "summary": summary,
        }
