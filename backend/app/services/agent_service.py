from __future__ import annotations

import asyncio
from typing import Any

from app.dependencies import get_memory_service, get_travel_agent
from app.schemas.chat import ChatResponse


class AgentService:
    """Compatibility adapter for code that still imports the old service name."""

    def __init__(self) -> None:
        self._agent = get_travel_agent()
        self.memory = get_memory_service()

    async def chat_async(
        self,
        session_id: str,
        message: str,
        image_path: str | None = None,
        audio_path: str | None = None,
        multimodal: bool = False,
    ) -> dict[str, Any]:
        data = await self._agent.handle(
            session_id=session_id,
            message=message,
            image_path=image_path,
            audio_path=audio_path,
            multimodal=multimodal,
        )
        ChatResponse(**data)
        return data

    def chat(
        self,
        session_id: str,
        message: str,
        image_path: str | None = None,
        audio_path: str | None = None,
        multimodal: bool = False,
    ) -> dict[str, Any]:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(
                self.chat_async(
                    session_id=session_id,
                    message=message,
                    image_path=image_path,
                    audio_path=audio_path,
                    multimodal=multimodal,
                )
            )
        raise RuntimeError("AgentService.chat() cannot run inside an active event loop; use chat_async().")
