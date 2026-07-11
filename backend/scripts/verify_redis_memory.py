from __future__ import annotations

import json
import sys
from pathlib import Path


BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.core.config import get_settings  # noqa: E402
from app.domain.models import ConversationTurn  # noqa: E402
from app.services.memory import InMemoryRepository, MemoryService, RedisBackedRepository  # noqa: E402


def main() -> int:
    settings = get_settings()
    repository = RedisBackedRepository(
        settings=settings,
        fallback=InMemoryRepository(settings.conversation_history_limit),
    )
    service = MemoryService(repository=repository)
    session_id = "redis-verify-session"

    service.append_turn(session_id, ConversationTurn(role="user", content="去杭州旅游"))
    service.append_turn(session_id, ConversationTurn(role="assistant", content="已记录杭州旅行需求"))
    profile = service.update_profile(
        session_id,
        destination="杭州",
        budget=5000,
        days=3,
        companions=2,
        preferences=["自然", "风景"],
        constraints={"travel_theme": "natural_scenery"},
    )
    long_memory = service.update_long_memory(
        session_id,
        last_destination="杭州",
        budget=5000,
        preferences=["自然", "风景"],
    )
    snapshot = service.snapshot(session_id)

    result = {
        "backend": snapshot.get("backend"),
        "history_count": snapshot.get("history_count"),
        "profile": profile,
        "long_memory": long_memory,
        "redis_url": settings.redis_url,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))

    if snapshot.get("backend") != "redis":
        print("Redis is not active; check USE_REDIS_MEMORY, REDIS_URL, and whether Redis is running.", file=sys.stderr)
        return 1
    if snapshot.get("history_count", 0) < 2:
        print("Redis history write verification failed.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
