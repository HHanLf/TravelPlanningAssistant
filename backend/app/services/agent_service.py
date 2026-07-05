from __future__ import annotations

from typing import Any

from backend.app.config.settings import get_settings
from backend.app.schemas.chat import ChatResponse
from backend.app.services.travel_agent_service import TravelAgentService


class AgentService:
    def __init__(self) -> None:
        self.settings = get_settings()
        self._service = TravelAgentService()

    @property
    def memory(self):
        return self._service.memory

    def chat(
        self,
        session_id: str,
        message: str,
        image_path: str | None = None,
        audio_path: str | None = None,
        multimodal: bool = False,
    ) -> dict[str, Any]:
        history = self.memory.get_history(session_id)
        result = self._service.respond(
            session_id=session_id,
            message=message,
            history=history,
            image_path=image_path,
            audio_path=audio_path,
            multimodal=multimodal,
        )

        snapshot = self.memory.read(session_id)
        reflection_result = result.get("reflection_result", {})
        normalized: dict[str, Any] = {
            "session_id": session_id,
            "answer": result.get("answer", ""),
            "intent": {"type": result.get("intent", "chat")},
            "plan": {
                "steps": [result.get("action_taken", "chat")],
                "needs_rag": result.get("action_taken") == "trip_plan_generated",
                "needs_tool": bool(result.get("sources")),
                "action_taken": result.get("action_taken"),
                "need_more_info": result.get("need_more_info", False),
                "cards": result.get("plan_cards", []),
                "sources": result.get("sources", []),
            },
            "retrieved_docs": [],
            "tool_results": {
                "sources": result.get("sources", []),
                "action_taken": result.get("action_taken"),
                "need_more_info": result.get("need_more_info", False),
                "trip_cost_estimate": reflection_result.get("trip_cost_estimate", {}),
                "budget_note": reflection_result.get("budget_note", ""),
            },
            "memory_context": {
                "chat_history": snapshot.chat_history,
                "user_profile": snapshot.user_profile,
                "short_memory": snapshot.short_memory,
                "long_memory": snapshot.long_memory,
            },
            "reflection_result": reflection_result,
            "profile": result.get("profile", {}),
            "multimodal_summary": result.get("multimodal_summary"),
            "audio_transcript": result.get("audio_transcript"),
        }

        ChatResponse(**normalized)
        return normalized
