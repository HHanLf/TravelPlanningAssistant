from __future__ import annotations

import json
from collections import defaultdict, deque

from app.core.config import get_settings


settings = get_settings()


class ConversationMemoryService:
    def __init__(self) -> None:
        self._history: dict[str, deque[tuple[str, str]]] = defaultdict(
            lambda: deque(maxlen=settings.conversation_history_limit)
        )
        self._profiles: dict[str, dict] = {}

    def get_history(self, session_id: str) -> list[tuple[str, str]]:
        return list(self._history.get(session_id, []))

    def append_message(self, session_id: str, role: str, content: str) -> None:
        self._history[session_id].append((role, content))

    def get_profile(self, session_id: str) -> dict:
        return self._profiles.get(session_id, {})

    def update_profile(self, session_id: str, **kwargs) -> dict:
        profile = self._profiles.setdefault(session_id, {})
        profile.update(kwargs)
        return profile

    def export_state(self) -> str:
        payload = {
            "history": {sid: list(items) for sid, items in self._history.items()},
            "profiles": self._profiles,
        }
        return json.dumps(payload, ensure_ascii=False, indent=2)
