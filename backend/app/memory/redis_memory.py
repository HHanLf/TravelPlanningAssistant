from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

import redis

from app.config.settings import Settings

logger = logging.getLogger(__name__)


@dataclass
class MemorySnapshot:
    chat_history: list[dict[str, str]]
    user_profile: dict[str, Any]
    short_memory: dict[str, Any]
    long_memory: dict[str, Any]


class RedisMemoryService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.client = redis.Redis.from_url(settings.redis_url, decode_responses=True)

    def _history_key(self, session_id: str) -> str:
        return f"travel_agent:history:{session_id}"

    def _profile_key(self, session_id: str) -> str:
        return f"travel_agent:profile:{session_id}"

    def _long_key(self, session_id: str) -> str:
        return f"travel_agent:long:{session_id}"

    def read(self, session_id: str) -> MemorySnapshot:
        try:
            history = [json.loads(item) for item in self.client.lrange(self._history_key(session_id), 0, -1)]
            profile = json.loads(self.client.get(self._profile_key(session_id)) or "{}")
            long_memory = json.loads(self.client.get(self._long_key(session_id)) or "{}")
            short_memory = {"recent_history": history[-8:]}
            return MemorySnapshot(chat_history=history, user_profile=profile, short_memory=short_memory, long_memory=long_memory)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Failed to read memory")
            return MemorySnapshot(chat_history=[], user_profile={"error": str(exc)}, short_memory={}, long_memory={})

    def append_message(self, session_id: str, role: str, content: str) -> None:
        payload = json.dumps({"role": role, "content": content}, ensure_ascii=False)
        self.client.rpush(self._history_key(session_id), payload)
        self.client.ltrim(self._history_key(session_id), -20, -1)

    def update_profile(self, session_id: str, updates: dict[str, Any]) -> dict[str, Any]:
        profile = json.loads(self.client.get(self._profile_key(session_id)) or "{}")
        profile.update({k: v for k, v in updates.items() if v is not None})
        self.client.set(self._profile_key(session_id), json.dumps(profile, ensure_ascii=False))
        return profile

    def update_long_memory(self, session_id: str, updates: dict[str, Any]) -> dict[str, Any]:
        long_memory = json.loads(self.client.get(self._long_key(session_id)) or "{}")
        long_memory.update({k: v for k, v in updates.items() if v is not None})
        self.client.set(self._long_key(session_id), json.dumps(long_memory, ensure_ascii=False))
        return long_memory
