from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Any

from backend.app.domain.models import ConversationTurn, UserProfile


class MemoryRepository(ABC):
    @abstractmethod
    def append_turn(self, session_id: str, turn: ConversationTurn) -> None:
        raise NotImplementedError

    @abstractmethod
    def get_history(self, session_id: str) -> list[ConversationTurn]:
        raise NotImplementedError

    @abstractmethod
    def get_or_create_profile(self, session_id: str) -> UserProfile:
        raise NotImplementedError

    @abstractmethod
    def snapshot(self, session_id: str) -> dict[str, Any]:
        raise NotImplementedError


class InMemoryRepository(MemoryRepository):
    def __init__(self) -> None:
        self._turns: dict[str, list[ConversationTurn]] = defaultdict(list)
        self._profiles: dict[str, UserProfile] = {}

    def append_turn(self, session_id: str, turn: ConversationTurn) -> None:
        self._turns[session_id].append(turn)

    def get_history(self, session_id: str) -> list[ConversationTurn]:
        return list(self._turns.get(session_id, []))

    def get_or_create_profile(self, session_id: str) -> UserProfile:
        profile = self._profiles.get(session_id)
        if profile is None:
            profile = UserProfile()
            self._profiles[session_id] = profile
        return profile

    def snapshot(self, session_id: str) -> dict[str, Any]:
        profile = self.get_or_create_profile(session_id)
        history = self._turns.get(session_id, [])
        return {
            "history_count": len(history),
            "history": [
                {"role": turn.role, "content": turn.content, "created_at": turn.created_at.isoformat()}
                for turn in history
            ],
            "user_profile": profile.to_dict(),
        }


class MemoryService:
    def __init__(self, repository: MemoryRepository | None = None) -> None:
        self._repository = repository or InMemoryRepository()

    def append_turn(self, session_id: str, turn: ConversationTurn) -> None:
        self._repository.append_turn(session_id, turn)

    def get_history(self, session_id: str) -> list[ConversationTurn]:
        return self._repository.get_history(session_id)

    def get_or_create_profile(self, session_id: str) -> UserProfile:
        return self._repository.get_or_create_profile(session_id)

    def snapshot(self, session_id: str) -> dict[str, Any]:
        return self._repository.snapshot(session_id)
