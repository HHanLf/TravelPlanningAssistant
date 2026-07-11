from __future__ import annotations

from typing import Any

from app.agent.answer_generator import AnswerGenerator
from app.agent.input_normalizer import InputNormalizer
from app.agent.planner import Planner
from app.agent.problem_analyzer import ProblemAnalyzer
from app.agent.reflection import ReflectionAgent
from app.agent.state import AgentResponse, AgentState, ToolResult
from app.core.config import get_settings
from app.domain.models import ConversationTurn
from app.research.executor import ResearchExecutor
from app.research.fusion import InformationFusion
from app.research.normalizer import EvidenceNormalizer
from app.research.planner import ResearchPlanner
from app.research.reflection import ResearchReflection
from app.research.summarizer import KnowledgeSummarizer
from app.services.intent import IntentAnalyzer
from app.services.memory import MemoryService
from app.tools.executor import ToolExecutor
from app.tools.registry import ToolRegistry


class TravelAgentOrchestrator:
    def __init__(
        self,
        memory_service: MemoryService,
        tool_registry: ToolRegistry,
        input_normalizer: InputNormalizer,
        intent_analyzer: IntentAnalyzer,
        problem_analyzer: ProblemAnalyzer,
        planner: Planner,
        tool_executor: ToolExecutor,
        answer_generator: AnswerGenerator,
        reflection_agent: ReflectionAgent,
        research_planner: ResearchPlanner | None = None,
        research_executor: ResearchExecutor | None = None,
        evidence_normalizer: EvidenceNormalizer | None = None,
        information_fusion: InformationFusion | None = None,
        knowledge_summarizer: KnowledgeSummarizer | None = None,
        research_reflection: ResearchReflection | None = None,
    ) -> None:
        self._memory = memory_service
        self._tool_registry = tool_registry
        self._input_normalizer = input_normalizer
        self._intent_analyzer = intent_analyzer
        self._problem_analyzer = problem_analyzer
        self._planner = planner
        self._tool_executor = tool_executor
        self._answer_generator = answer_generator
        self._reflection = reflection_agent
        self._research_planner = research_planner or ResearchPlanner()
        self._research_executor = research_executor or ResearchExecutor(tool_registry)
        self._evidence_normalizer = evidence_normalizer or EvidenceNormalizer()
        self._information_fusion = information_fusion or InformationFusion()
        self._knowledge_summarizer = knowledge_summarizer or KnowledgeSummarizer()
        self._research_reflection = research_reflection or ResearchReflection()
        self._settings = get_settings()

    async def handle(
        self,
        session_id: str,
        message: str,
        image_path: str | None = None,
        audio_path: str | None = None,
        multimodal: bool = False,
    ) -> dict[str, Any]:
        state = AgentState(session_id=session_id, raw_message=message or "")
        state = self._input_normalizer.normalize(
            state,
            image_path=image_path if multimodal or image_path else None,
            audio_path=audio_path if multimodal or audio_path else None,
        )

        state.memory_context = self._memory.snapshot(session_id)
        state.problem = self._problem_analyzer.analyze(state.effective_message, state.memory_context)
        state.intent = self._intent_analyzer.analyze(state.effective_message, state.problem)
        state.profile = self._memory.update_profile(
            session_id,
            **self._problem_analyzer.profile_updates(state.problem),
        )
        state.memory_context = self._memory.snapshot(session_id)

        max_rounds = max(1, int(self._settings.reflection_max_rounds or 1))
        for round_index in range(max_rounds):
            state.loop_count = round_index
            state.plan = self._planner.create_plan(
                message=state.effective_message,
                intent=state.intent,
                problem=state.problem,
                memory_context=state.memory_context,
                available_tools=self._tool_registry.list_specs(),
                reflection_notes=state.reflection_notes,
            )
            research_tasks = self._research_planner.create_tasks(
                message=state.effective_message,
                intent=state.intent,
                problem=state.problem,
                plan=state.plan,
                available_tools=self._tool_registry.list_specs(),
                reflection_notes=state.reflection_notes,
            )
            state.research_tasks = [item.to_dict() for item in research_tasks]
            if state.research_tasks:
                research_results = await self._research_executor.execute(state, research_tasks)
                state.tool_results = [item.tool_result for item in research_results]
                evidence_items = self._evidence_normalizer.normalize_many(research_results)
                state.evidence_items = [item.to_dict() for item in evidence_items]
                fused = self._information_fusion.fuse(
                    evidence_items,
                    destination=state.problem.destination if state.problem and state.problem.destination else "",
                )
                state.knowledge_summary = self._knowledge_summarizer.summarize(fused)
                research_quality = self._research_reflection.evaluate(
                    state.knowledge_summary,
                    intent_type=state.intent.type if state.intent else "",
                )
                state.knowledge_summary["research_reflection"] = research_quality
                if not research_quality.get("passed"):
                    state.reflection_notes = [
                        *state.reflection_notes,
                        *[f"Research issue: {item}" for item in research_quality.get("issues", [])],
                        *[f"Research suggestion: {item}" for item in research_quality.get("suggestions", [])],
                    ]
            else:
                state.tool_results = await self._tool_executor.execute_many(state, state.plan.tool_calls)
            state.draft_answer = self._answer_generator.generate(state)
            state.reflection_result = self._reflection.evaluate(state, round_index=round_index)
            if state.reflection_result.passed or round_index >= max_rounds - 1:
                break
            state.reflection_notes = [*state.reflection_result.issues, *state.reflection_result.suggestions]

        if (
            state.intent
            and state.intent.type == "trip_plan"
            and state.reflection_result
            and not state.reflection_result.passed
        ):
            state.reflection_notes = [*state.reflection_result.issues, *state.reflection_result.suggestions]
            state.draft_answer = self._answer_generator.generate(state, force_fallback=True)
            state.reflection_result = self._reflection.evaluate(state, round_index=max_rounds)

        state.final_answer = state.draft_answer
        self._save_memory(state)
        state.memory_context = self._memory.snapshot(session_id)
        state.profile = state.memory_context.get("user_profile", state.profile)
        return self._build_response(state).to_dict()

    def _save_memory(self, state: AgentState) -> None:
        self._memory.append_turn(
            state.session_id,
            ConversationTurn(role="user", content=state.effective_message),
        )
        self._memory.append_turn(
            state.session_id,
            ConversationTurn(role="assistant", content=state.final_answer),
        )
        if state.problem:
            self._memory.update_long_memory(
                state.session_id,
                last_destination=state.problem.destination,
                last_origin=state.problem.origin,
                last_preferences=state.problem.preferences,
                last_intent=state.intent.type if state.intent else None,
                budget=state.problem.budget,
                companions=state.problem.group_size,
                preferences=state.problem.preferences,
                constraints=state.problem.constraints,
            )

    def _build_response(self, state: AgentState) -> AgentResponse:
        return AgentResponse(
            session_id=state.session_id,
            answer=state.final_answer,
            intent=state.intent.to_dict() if state.intent else {},
            plan=state.plan.to_dict() if state.plan else {},
            tool_results=self._tool_results_payload(state.tool_results),
            memory_context=state.memory_context,
            reflection_result=state.reflection_result.to_dict() if state.reflection_result else {},
            knowledge_summary=state.knowledge_summary,
            research_tasks=state.research_tasks,
            evidence_items=state.evidence_items,
            retrieved_docs=state.retrieved_docs,
            profile=state.profile,
            multimodal_summary=state.multimodal_summary,
            audio_transcript=state.audio_transcript,
        )

    @staticmethod
    def _tool_results_payload(results: list[ToolResult]) -> dict[str, Any]:
        items = [item.to_dict() for item in results]
        by_name: dict[str, list[dict[str, Any]]] = {}
        for item in items:
            by_name.setdefault(str(item.get("name")), []).append(item)
        payload: dict[str, Any] = {"items": items, "by_name": by_name}
        for name, values in by_name.items():
            payload[name] = values[0] if len(values) == 1 else values
        return payload
