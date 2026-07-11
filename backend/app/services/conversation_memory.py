from collections import deque
from dataclasses import dataclass, asdict

from app.core.config import get_settings


settings = get_settings()


@dataclass
class MemoryItem:
    role: str
    content: str


class ConversationMemoryService:
    def __init__(self) -> None:
        self._history: dict[str, deque[MemoryItem]] = {}
        self._profile: dict[str, dict] = {}

    def get_history(self, session_id: str) -> list[tuple[str, str]]:
        items = self._history.get(session_id, deque())
        return [(item.role, item.content) for item in items]

    def get_history_items(self, session_id: str) -> list[dict]:
        items = self._history.get(session_id, deque())
        return [asdict(item) for item in items]

    def append_message(self, session_id: str, role: str, content: str) -> None:
        if session_id not in self._history:
            self._history[session_id] = deque(maxlen=settings.conversation_history_limit)
        self._history[session_id].append(MemoryItem(role=role, content=content))

    def get_profile(self, session_id: str) -> dict:
        return self._profile.get(session_id, {})

    def update_profile(self, session_id: str, **kwargs) -> dict:
        profile = self._profile.setdefault(session_id, {})
        profile.update(kwargs)
        return profile

    def clear_profile_fields(self, session_id: str, *fields: str) -> dict:
        profile = self._profile.setdefault(session_id, {})
        for field in fields:
            profile.pop(field, None)
        return profile
