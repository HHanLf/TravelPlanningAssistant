from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any

from app.core.config import Settings, get_settings
from app.domain.models import ConversationTurn, UserProfile

logger = logging.getLogger(__name__)


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
    def save_profile(self, session_id: str, profile: UserProfile) -> None:
        raise NotImplementedError

    @abstractmethod
    def update_long_memory(self, session_id: str, updates: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def clear_destination_context(self, session_id: str, keep_keys: set[str] | None = None) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def snapshot(self, session_id: str) -> dict[str, Any]:
        raise NotImplementedError


class InMemoryRepository(MemoryRepository):
    def __init__(self, history_limit: int = 20) -> None:
        self._history_limit = history_limit
        self._turns: dict[str, list[ConversationTurn]] = defaultdict(list)
        self._profiles: dict[str, UserProfile] = {}
        self._long_memory: dict[str, dict[str, Any]] = defaultdict(dict)

    def append_turn(self, session_id: str, turn: ConversationTurn) -> None:
        history = self._turns[session_id]
        history.append(turn)
        self._turns[session_id] = history[-self._history_limit :]

    def get_history(self, session_id: str) -> list[ConversationTurn]:
        return list(self._turns.get(session_id, []))

    def get_or_create_profile(self, session_id: str) -> UserProfile:
        profile = self._profiles.get(session_id)
        if profile is None:
            profile = UserProfile()
            self._profiles[session_id] = profile
        return profile

    def save_profile(self, session_id: str, profile: UserProfile) -> None:
        self._profiles[session_id] = profile

    def update_long_memory(self, session_id: str, updates: dict[str, Any]) -> dict[str, Any]:
        current = self._long_memory[session_id]
        current.update({key: value for key, value in updates.items() if value not in (None, "", [])})
        return dict(current)

    def clear_destination_context(self, session_id: str, keep_keys: set[str] | None = None) -> dict[str, Any]:
        keep = keep_keys or set()
        current = self._long_memory.get(session_id, {})
        self._long_memory[session_id] = {key: value for key, value in current.items() if key in keep}
        return dict(self._long_memory[session_id])

    def snapshot(self, session_id: str) -> dict[str, Any]:
        history = self.get_history(session_id)
        profile = self.get_or_create_profile(session_id)
        long_memory = dict(self._long_memory.get(session_id, {}))
        return {
            "history_count": len(history),
            "chat_history": [self._turn_to_dict(turn) for turn in history],
            "recent_history": [self._turn_to_dict(turn) for turn in history[-8:]],
            "short_memory": {"recent_history": [self._turn_to_dict(turn) for turn in history[-8:]]},
            "long_memory": long_memory,
            "user_profile": profile.to_dict(),
            "backend": "memory",
        }

    @staticmethod
    def _turn_to_dict(turn: ConversationTurn) -> dict[str, str]:
        return {
            "role": turn.role,
            "content": turn.content,
            "created_at": turn.created_at.isoformat(),
        }


class RedisBackedRepository(MemoryRepository):
    def __init__(self, settings: Settings, fallback: InMemoryRepository | None = None) -> None:
        self.settings = settings
        self._fallback = fallback or InMemoryRepository(settings.conversation_history_limit)
        self._use_fallback = False
        self._redis = None
        try:
            import redis

            self._redis = redis.Redis.from_url(settings.redis_url, decode_responses=True)
            self._redis.ping()
        except Exception:  # noqa: BLE001
            logger.warning("Redis memory unavailable; falling back to in-memory store.", exc_info=True)
            self._use_fallback = True

    def append_turn(self, session_id: str, turn: ConversationTurn) -> None:
        if self._use_fallback:
            self._fallback.append_turn(session_id, turn)
            return
        try:
            payload = json.dumps(self._turn_to_dict(turn), ensure_ascii=False)
            client = self._client()
            client.rpush(self._history_key(session_id), payload)
            client.ltrim(self._history_key(session_id), -self.settings.conversation_history_limit, -1)
            client.expire(self._history_key(session_id), self.settings.conversation_memory_ttl_seconds)
        except Exception:  # noqa: BLE001
            self._degrade()
            self.append_turn(session_id, turn)

    def get_history(self, session_id: str) -> list[ConversationTurn]:
        if self._use_fallback:
            return self._fallback.get_history(session_id)
        try:
            items = self._client().lrange(self._history_key(session_id), 0, -1)
            return [self._turn_from_dict(json.loads(item)) for item in items]
        except Exception:  # noqa: BLE001
            self._degrade()
            return self._fallback.get_history(session_id)

    def get_or_create_profile(self, session_id: str) -> UserProfile:
        if self._use_fallback:
            return self._fallback.get_or_create_profile(session_id)
        try:
            raw = self._client().get(self._profile_key(session_id))
            return self._profile_from_dict(json.loads(raw or "{}"))
        except Exception:  # noqa: BLE001
            self._degrade()
            return self._fallback.get_or_create_profile(session_id)

    def save_profile(self, session_id: str, profile: UserProfile) -> None:
        if self._use_fallback:
            self._fallback.save_profile(session_id, profile)
            return
        try:
            self._client().set(
                self._profile_key(session_id),
                json.dumps(profile.to_dict(), ensure_ascii=False),
                ex=self.settings.user_profile_ttl_seconds,
            )
        except Exception:  # noqa: BLE001
            self._degrade()
            self._fallback.save_profile(session_id, profile)

    def update_long_memory(self, session_id: str, updates: dict[str, Any]) -> dict[str, Any]:
        if self._use_fallback:
            return self._fallback.update_long_memory(session_id, updates)
        try:
            client = self._client()
            current = json.loads(client.get(self._long_key(session_id)) or "{}")
            current.update({key: value for key, value in updates.items() if value not in (None, "", [])})
            client.set(self._long_key(session_id), json.dumps(current, ensure_ascii=False))
            return current
        except Exception:  # noqa: BLE001
            self._degrade()
            return self._fallback.update_long_memory(session_id, updates)

    def clear_destination_context(self, session_id: str, keep_keys: set[str] | None = None) -> dict[str, Any]:
        if self._use_fallback:
            return self._fallback.clear_destination_context(session_id, keep_keys)
        keep = keep_keys or set()
        try:
            client = self._client()
            current = json.loads(client.get(self._long_key(session_id)) or "{}")
            cleaned = {key: value for key, value in current.items() if key in keep}
            client.set(self._long_key(session_id), json.dumps(cleaned, ensure_ascii=False))
            return cleaned
        except Exception:  # noqa: BLE001
            self._degrade()
            return self._fallback.clear_destination_context(session_id, keep_keys)

    def snapshot(self, session_id: str) -> dict[str, Any]:
        if self._use_fallback:
            snapshot = self._fallback.snapshot(session_id)
            snapshot["backend"] = "memory_fallback"
            return snapshot
        history = self.get_history(session_id)
        profile = self.get_or_create_profile(session_id)
        try:
            long_memory = json.loads(self._client().get(self._long_key(session_id)) or "{}")
        except Exception:  # noqa: BLE001
            long_memory = {}
        recent = [self._turn_to_dict(turn) for turn in history[-8:]]
        return {
            "history_count": len(history),
            "chat_history": [self._turn_to_dict(turn) for turn in history],
            "recent_history": recent,
            "short_memory": {"recent_history": recent},
            "long_memory": long_memory,
            "user_profile": profile.to_dict(),
            "backend": "redis",
        }

    def _client(self):
        if self._redis is None:
            raise RuntimeError("Redis client is not initialized")
        return self._redis

    def _degrade(self) -> None:
        logger.warning("Redis memory operation failed; switching to in-memory fallback.", exc_info=True)
        self._use_fallback = True

    def _history_key(self, session_id: str) -> str:
        return f"travel_agent:history:{session_id}"

    def _profile_key(self, session_id: str) -> str:
        return f"travel_agent:profile:{session_id}"

    def _long_key(self, session_id: str) -> str:
        return f"travel_agent:long:{session_id}"

    @staticmethod
    def _turn_to_dict(turn: ConversationTurn) -> dict[str, str]:
        return {
            "role": turn.role,
            "content": turn.content,
            "created_at": turn.created_at.isoformat(),
        }

    @staticmethod
    def _turn_from_dict(item: dict[str, Any]) -> ConversationTurn:
        created_at = item.get("created_at")
        try:
            parsed_at = datetime.fromisoformat(created_at) if created_at else datetime.now(timezone.utc)
        except ValueError:
            parsed_at = datetime.now(timezone.utc)
        return ConversationTurn(
            role=str(item.get("role") or ""),
            content=str(item.get("content") or ""),
            created_at=parsed_at,
        )

    @staticmethod
    def _profile_from_dict(data: dict[str, Any]) -> UserProfile:
        allowed = {"departure", "destination", "budget", "days", "companions", "preferences", "date_range", "constraints"}
        return UserProfile(**{key: value for key, value in data.items() if key in allowed})


class MemoryService:
    def __init__(self, repository: MemoryRepository | None = None) -> None:
        settings = get_settings()
        self._repository = repository or InMemoryRepository(settings.conversation_history_limit)

    def append_turn(self, session_id: str, turn: ConversationTurn) -> None:
        self._repository.append_turn(session_id, turn)

    def get_history(self, session_id: str) -> list[ConversationTurn]:
        return self._repository.get_history(session_id)

    def get_or_create_profile(self, session_id: str) -> UserProfile:
        return self._repository.get_or_create_profile(session_id)

    def save_profile(self, session_id: str, profile: UserProfile) -> None:
        self._repository.save_profile(session_id, profile)

    def update_profile(self, session_id: str, **updates: Any) -> dict[str, Any]:
        context_scope = str(updates.pop("_context_scope", "") or "")
        explicit_fields = set(updates.pop("_explicit_fields", []) or [])
        profile = self.get_or_create_profile(session_id)
        incoming_destination = updates.get("destination")
        if incoming_destination and profile.destination and incoming_destination != profile.destination:
            self.clear_destination_context(session_id)
            if context_scope == "local_lookup":
                profile.budget = None
                profile.days = None
                profile.companions = None
                profile.preferences = []
                profile.date_range = None
                profile.constraints = {}
        for key, value in updates.items():
            if value in (None, "", []):
                continue
            if context_scope == "local_lookup" and key not in explicit_fields and key in {
                "budget",
                "days",
                "companions",
                "preferences",
                "date_range",
                "constraints",
            }:
                continue
            if hasattr(profile, key):
                setattr(profile, key, value)
        self.save_profile(session_id, profile)
        return profile.to_dict()

    def update_long_memory(self, session_id: str, **updates: Any) -> dict[str, Any]:
        return self._repository.update_long_memory(session_id, updates)

    def clear_destination_context(self, session_id: str) -> dict[str, Any]:
        keep_keys = {
            "budget",
            "companions",
            "group_size",
            "preferences",
            "travel_style",
            "hotel_preference",
            "food_preference",
            "transport_preference",
            "constraints",
            "last_origin",
            "last_preferences",
        }
        return self._repository.clear_destination_context(session_id, keep_keys)

    def snapshot(self, session_id: str) -> dict[str, Any]:
        return self._repository.snapshot(session_id)
