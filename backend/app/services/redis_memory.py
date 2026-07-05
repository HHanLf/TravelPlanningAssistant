from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

import redis

from backend.app.config.settings import Settings, get_settings

logger = logging.getLogger(__name__)


@dataclass
class MemorySnapshot:
    chat_history: list[dict[str, str]]
    user_profile: dict[str, Any]
    short_memory: dict[str, Any]
    long_memory: dict[str, Any]


class RedisMemoryService:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.client = redis.Redis.from_url(self.settings.redis_url, decode_responses=True)
        self._local_history: dict[str, list[dict[str, str]]] = {}
        self._local_profile: dict[str, dict[str, Any]] = {}
        self._local_long: dict[str, dict[str, Any]] = {}
        self._use_local_store = False
        try:
            self.client.ping()
        except Exception:  # noqa: BLE001
            self._use_local_store = True

    def _history_key(self, session_id: str) -> str:
        return f"travel_agent:history:{session_id}"

    def _profile_key(self, session_id: str) -> str:
        return f"travel_agent:profile:{session_id}"

    def _long_key(self, session_id: str) -> str:
        return f"travel_agent:long:{session_id}"

    def read(self, session_id: str) -> MemorySnapshot:
        if self._use_local_store:
            history = self._local_history.get(session_id, [])
            profile = self._local_profile.get(session_id, {})
            long_memory = self._local_long.get(session_id, {})
            return MemorySnapshot(
                chat_history=list(history),
                user_profile=dict(profile),
                short_memory={"recent_history": history[-8:]},
                long_memory=dict(long_memory),
            )

        try:
            history = [json.loads(item) for item in self.client.lrange(self._history_key(session_id), 0, -1)]
            profile = json.loads(self.client.get(self._profile_key(session_id)) or "{}")
            long_memory = json.loads(self.client.get(self._long_key(session_id)) or "{}")
            short_memory = {"recent_history": history[-8:]}
            return MemorySnapshot(chat_history=history, user_profile=profile, short_memory=short_memory, long_memory=long_memory)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Failed to read memory")
            return MemorySnapshot(chat_history=[], user_profile={"error": str(exc)}, short_memory={}, long_memory={})

    def get_history(self, session_id: str) -> list[tuple[str, str]]:
        snapshot = self.read(session_id)
        history: list[tuple[str, str]] = []
        for item in snapshot.chat_history:
            role = str(item.get("role", ""))
            content = str(item.get("content", ""))
            if role and content:
                history.append((role, content))
        return history

    def append_message(self, session_id: str, role: str, content: str) -> None:
        payload = {"role": role, "content": content}
        if self._use_local_store:
            history = self._local_history.setdefault(session_id, [])
            history.append(payload)
            self._local_history[session_id] = history[-20:]
            return

        try:
            encoded = json.dumps(payload, ensure_ascii=False)
            self.client.rpush(self._history_key(session_id), encoded)
            self.client.ltrim(self._history_key(session_id), -20, -1)
            self.client.expire(self._history_key(session_id), self.settings.conversation_memory_ttl_seconds)
        except Exception:  # noqa: BLE001
            logger.exception("Failed to append message to redis memory")
            self._use_local_store = True
            self.append_message(session_id, role, content)

    def update_profile(self, session_id: str, **updates: Any) -> dict[str, Any]:
        if self._use_local_store:
            profile = self._local_profile.setdefault(session_id, {})
            profile.update({k: v for k, v in updates.items() if v is not None})
            return dict(profile)

        try:
            profile = json.loads(self.client.get(self._profile_key(session_id)) or "{}")
            profile.update({k: v for k, v in updates.items() if v is not None})
            self.client.set(
                self._profile_key(session_id),
                json.dumps(profile, ensure_ascii=False),
                ex=self.settings.user_profile_ttl_seconds,
            )
            return profile
        except Exception:  # noqa: BLE001
            logger.exception("Failed to update profile in redis memory")
            self._use_local_store = True
            return self.update_profile(session_id, **updates)

    def clear_profile_fields(self, session_id: str, *fields: str) -> dict[str, Any]:
        if not fields:
            return self.get_profile(session_id)

        if self._use_local_store:
            profile = self._local_profile.setdefault(session_id, {})
            for field in fields:
                profile.pop(field, None)
            return dict(profile)

        try:
            profile = json.loads(self.client.get(self._profile_key(session_id)) or "{}")
            changed = False
            for field in fields:
                if field in profile:
                    profile.pop(field, None)
                    changed = True
            if changed:
                self.client.set(
                    self._profile_key(session_id),
                    json.dumps(profile, ensure_ascii=False),
                    ex=self.settings.user_profile_ttl_seconds,
                )
            return profile
        except Exception:  # noqa: BLE001
            logger.exception("Failed to clear profile fields in redis memory")
            self._use_local_store = True
            return self.clear_profile_fields(session_id, *fields)

    def get_profile(self, session_id: str) -> dict[str, Any]:
        if self._use_local_store:
            return dict(self._local_profile.get(session_id, {}))

        try:
            return json.loads(self.client.get(self._profile_key(session_id)) or "{}")
        except Exception:  # noqa: BLE001
            logger.exception("Failed to read profile from redis memory")
            self._use_local_store = True
            return dict(self._local_profile.get(session_id, {}))

    def update_long_memory(self, session_id: str, **updates: Any) -> dict[str, Any]:
        if self._use_local_store:
            long_memory = self._local_long.setdefault(session_id, {})
            long_memory.update({k: v for k, v in updates.items() if v is not None})
            return dict(long_memory)

        try:
            long_memory = json.loads(self.client.get(self._long_key(session_id)) or "{}")
            long_memory.update({k: v for k, v in updates.items() if v is not None})
            self.client.set(self._long_key(session_id), json.dumps(long_memory, ensure_ascii=False))
            return long_memory
        except Exception:  # noqa: BLE001
            logger.exception("Failed to update long memory in redis")
            self._use_local_store = True
            return self.update_long_memory(session_id, **updates)

    def clear_profile_fields(self, session_id: str, *fields: str) -> dict[str, Any]:
        if self._use_local_store:
            profile = self._local_profile.setdefault(session_id, {})
            for field in fields:
                profile.pop(field, None)
            return dict(profile)

        try:
            profile = json.loads(self.client.get(self._profile_key(session_id)) or "{}")
            for field in fields:
                profile.pop(field, None)
            self.client.set(
                self._profile_key(session_id),
                json.dumps(profile, ensure_ascii=False),
                ex=self.settings.user_profile_ttl_seconds,
            )
            return profile
        except Exception:  # noqa: BLE001
            logger.exception("Failed to clear profile fields in redis memory")
            self._use_local_store = True
            return self.clear_profile_fields(session_id, *fields)
