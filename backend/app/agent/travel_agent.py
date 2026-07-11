from __future__ import annotations

from typing import Any

from app.agent.orchestrator import TravelAgentOrchestrator


class TravelAgent:
    """Thin public facade kept for dependency and route compatibility."""

    def __init__(self, orchestrator: TravelAgentOrchestrator) -> None:
        self._orchestrator = orchestrator

    async def handle(
        self,
        session_id: str,
        message: str,
        image_path: str | None = None,
        audio_path: str | None = None,
        multimodal: bool = False,
    ) -> dict[str, Any]:
        return await self._orchestrator.handle(
            session_id=session_id,
            message=message,
            image_path=image_path,
            audio_path=audio_path,
            multimodal=multimodal,
        )
