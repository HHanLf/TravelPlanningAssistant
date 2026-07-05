from __future__ import annotations

from backend.app.agent.context_manager import ContextManager
from backend.app.agent.planner import Planner
from backend.app.agent.profile_updater import ProfileUpdater
from backend.app.agent.response_builder import ResponseBuilder
from backend.app.domain.models import ConversationTurn
from backend.app.services.intent import IntentAnalyzer
from backend.app.services.memory import MemoryService
from backend.app.tools.executor import ToolExecutor


class TravelAgent:
    def __init__(
        self,
        memory_service: MemoryService,
        intent_analyzer: IntentAnalyzer,
        context_manager: ContextManager,
        profile_updater: ProfileUpdater,
        planner: Planner,
        tool_executor: ToolExecutor,
        response_builder: ResponseBuilder,
    ) -> None:
        self._memory_service = memory_service
        self._intent_analyzer = intent_analyzer
        self._context_manager = context_manager
        self._profile_updater = profile_updater
        self._planner = planner
        self._tool_executor = tool_executor
        self._response_builder = response_builder

    async def handle(self, session_id: str, message: str, audio_transcript: str | None = None) -> dict:
        normalized_message = audio_transcript or message
        self._memory_service.append_turn(
            session_id,
            ConversationTurn(role="user", content=normalized_message),
        )

        context = self._context_manager.build(session_id=session_id, latest_message=normalized_message)
        intent = self._intent_analyzer.analyze(normalized_message)
        self._profile_updater.apply_intent(context.user_profile, intent)

        execution_plan = self._planner.create_plan(context, intent)
        tool_results = await self._tool_executor.execute_many(context, execution_plan.tool_calls)
        response = self._response_builder.build(
            context=context,
            intent=intent,
            tool_results=tool_results,
            plan=execution_plan.to_dict(),
        )
        response.audio_transcript = audio_transcript

        self._memory_service.append_turn(session_id, ConversationTurn(role="assistant", content=response.answer))
        context.memory = self._memory_service.snapshot(session_id)
        response.memory_context = context.memory

        return {
            "answer": response.answer,
            "intent": response.intent,
            "plan": response.plan,
            "tool_results": response.tool_results,
            "memory_context": response.memory_context,
            "reflection_result": response.reflection_result,
            "retrieved_docs": response.retrieved_docs,
            "audio_transcript": response.audio_transcript,
        }
