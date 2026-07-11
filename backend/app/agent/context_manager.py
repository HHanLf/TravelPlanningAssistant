from __future__ import annotations

from app.domain.models import AgentContext
from app.services.memory import MemoryService


class ContextManager:
    def __init__(self, memory_service: MemoryService) -> None:
        self._memory_service = memory_service

    def build(self, session_id: str, latest_message: str) -> AgentContext:
        return AgentContext(
            session_id=session_id,
            latest_message=latest_message,
            history=self._memory_service.get_history(session_id),
            user_profile=self._memory_service.get_or_create_profile(session_id),
            memory=self._memory_service.snapshot(session_id),
        )
