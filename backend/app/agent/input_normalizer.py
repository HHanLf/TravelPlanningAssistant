from __future__ import annotations

from pathlib import Path

from app.agent.state import AgentState
from app.services.multimodal_service import MultimodalService


class InputNormalizer:
    def __init__(self, multimodal_service: MultimodalService | None = None) -> None:
        self._multimodal = multimodal_service or MultimodalService()

    def normalize(
        self,
        state: AgentState,
        image_path: str | None = None,
        audio_path: str | None = None,
    ) -> AgentState:
        message_parts = [state.raw_message.strip()]
        summaries: list[str] = []

        if audio_path:
            audio_result = self._multimodal.transcribe_audio(audio_path)
            transcript = str(audio_result.get("text") or "").strip()
            state.audio_transcript = transcript or None
            if transcript:
                message_parts.append(transcript)
                summaries.append(f"语音：{transcript}")
            elif audio_result.get("error"):
                summaries.append(f"语音识别失败：{audio_result['error']}")

        if image_path:
            if Path(image_path).exists():
                image_summary = self._multimodal.describe_image(image_path)
            else:
                image_summary = "图片文件不存在，无法识别。"
            state.image_summary = image_summary
            summaries.append(f"图片：{image_summary}")
            if image_summary:
                message_parts.append(image_summary)

        state.multimodal_summary = "\n".join(item for item in summaries if item) or None
        state.effective_message = "\n".join(part for part in message_parts if part).strip()
        if not state.effective_message:
            state.effective_message = "用户没有提供可识别的文本内容。"
        return state
