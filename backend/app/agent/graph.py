from __future__ import annotations

from dataclasses import fields
from typing import Any

from langgraph.graph import END, StateGraph

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
from app.research.models import ResearchTask
from app.research.normalizer import EvidenceNormalizer
from app.research.planner import ResearchPlanner
from app.research.reflection import ResearchReflection
from app.research.summarizer import KnowledgeSummarizer
from app.services.intent import IntentAnalyzer
from app.services.memory import MemoryService
from app.tools.executor import ToolExecutor
from app.tools.registry import ToolRegistry


class TravelAgentGraph:
    """LangGraph production workflow for the travel agent backend."""

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
        self.graph = self._build_graph().compile()

    async def handle(
        self,
        session_id: str,
        message: str,
        image_path: str | None = None,
        audio_path: str | None = None,
        multimodal: bool = False,
    ) -> dict[str, Any]:
        initial_state = {
            "session_id": session_id,
            "raw_message": message or "",
            "response": {
                "_image_path": image_path if multimodal or image_path else None,
                "_audio_path": audio_path if multimodal or audio_path else None,
            },
        }
        result = await self.graph.ainvoke(initial_state)
        response = result.get("response") if isinstance(result, dict) else None
        if isinstance(response, dict) and "answer" in response:
            return response
        return self._build_response(self._state(result)).to_dict()

    def _build_graph(self) -> StateGraph:
        graph = StateGraph(AgentState)
        graph.add_node("normalize_input", self.normalize_input)
        graph.add_node("load_memory", self.load_memory)
        graph.add_node("analyze_problem", self.analyze_problem)
        graph.add_node("analyze_intent", self.analyze_intent)
        graph.add_node("update_profile", self.update_profile)
        graph.add_node("create_plan", self.create_plan)
        graph.add_node("create_research_tasks", self.create_research_tasks)
        graph.add_node("execute_research_or_tools", self.execute_research_or_tools)
        graph.add_node("generate_answer", self.generate_answer)
        graph.add_node("reflect", self.reflect)
        graph.add_node("prepare_reflection_retry", self.prepare_reflection_retry)
        graph.add_node("fallback_if_needed", self.fallback_if_needed)
        graph.add_node("save_memory", self.save_memory)
        graph.add_node("build_response", self.build_response)

        graph.set_entry_point("normalize_input")
        graph.add_edge("normalize_input", "load_memory")
        graph.add_edge("load_memory", "analyze_problem")
        graph.add_edge("analyze_problem", "analyze_intent")
        graph.add_edge("analyze_intent", "update_profile")
        graph.add_edge("update_profile", "create_plan")
        graph.add_edge("create_plan", "create_research_tasks")
        graph.add_conditional_edges(
            "create_research_tasks",
            self.execution_route,
            {
                "research": "execute_research_or_tools",
                "tools": "execute_research_or_tools",
            },
        )
        graph.add_edge("execute_research_or_tools", "generate_answer")
        graph.add_edge("generate_answer", "reflect")
        graph.add_conditional_edges(
            "reflect",
            self.reflection_route,
            {
                "retry": "prepare_reflection_retry",
                "finalize": "fallback_if_needed",
            },
        )
        graph.add_edge("prepare_reflection_retry", "create_plan")
        graph.add_edge("fallback_if_needed", "save_memory")
        graph.add_edge("save_memory", "build_response")
        graph.add_edge("build_response", END)
        return graph

    def normalize_input(self, state: AgentState | dict[str, Any]) -> dict[str, Any]:
        current = self._state(state)
        metadata = current.response or {}
        normalized = self._input_normalizer.normalize(
            current,
            image_path=metadata.get("_image_path"),
            audio_path=metadata.get("_audio_path"),
        )
        return self._updates(normalized)

    def load_memory(self, state: AgentState | dict[str, Any]) -> dict[str, Any]:
        current = self._state(state)
        return {"memory_context": self._memory.snapshot(current.session_id)}

    def analyze_problem(self, state: AgentState | dict[str, Any]) -> dict[str, Any]:
        current = self._state(state)
        return {
            "problem": self._problem_analyzer.analyze(
                current.effective_message,
                current.memory_context,
            )
        }

    def analyze_intent(self, state: AgentState | dict[str, Any]) -> dict[str, Any]:
        current = self._state(state)
        return {"intent": self._intent_analyzer.analyze(current.effective_message, current.problem)}

    def update_profile(self, state: AgentState | dict[str, Any]) -> dict[str, Any]:
        current = self._state(state)
        if current.problem is None:
            return {"profile": current.profile}
        profile = self._memory.update_profile(
            current.session_id,
            **self._problem_analyzer.profile_updates(current.problem),
        )
        return {
            "profile": profile,
            "memory_context": self._memory.snapshot(current.session_id),
        }

    def create_plan(self, state: AgentState | dict[str, Any]) -> dict[str, Any]:
        current = self._state(state)
        if current.intent is None or current.problem is None:
            return {"plan": None, "research_tasks": [], "tool_results": []}
        return {
            "plan": self._planner.create_plan(
                message=current.effective_message,
                intent=current.intent,
                problem=current.problem,
                memory_context=current.memory_context,
                available_tools=self._tool_registry.list_specs(),
                reflection_notes=current.reflection_notes,
            ),
            "research_tasks": [],
            "evidence_items": [],
            "knowledge_summary": {},
            "tool_results": [],
        }

    def create_research_tasks(self, state: AgentState | dict[str, Any]) -> dict[str, Any]:
        current = self._state(state)
        tasks = self._research_planner.create_tasks(
            message=current.effective_message,
            intent=current.intent,
            problem=current.problem,
            plan=current.plan,
            available_tools=self._tool_registry.list_specs(),
            reflection_notes=current.reflection_notes,
        )
        return {"research_tasks": [item.to_dict() for item in tasks]}

    async def execute_research_or_tools(self, state: AgentState | dict[str, Any]) -> dict[str, Any]:
        current = self._state(state)
        if current.research_tasks:
            tasks = [self._research_task(item) for item in current.research_tasks]
            research_results = await self._research_executor.execute(current, tasks)
            evidence_items = self._evidence_normalizer.normalize_many(research_results)
            fused = self._information_fusion.fuse(
                evidence_items,
                destination=current.problem.destination
                if current.problem and current.problem.destination
                else "",
            )
            knowledge_summary = self._knowledge_summarizer.summarize(fused)
            research_quality = self._research_reflection.evaluate(
                knowledge_summary,
                intent_type=current.intent.type if current.intent else "",
            )
            knowledge_summary["research_reflection"] = research_quality
            reflection_notes = list(current.reflection_notes)
            if not research_quality.get("passed"):
                reflection_notes.extend(
                    [
                        *[
                            f"Research issue: {item}"
                            for item in research_quality.get("issues", [])
                        ],
                        *[
                            f"Research suggestion: {item}"
                            for item in research_quality.get("suggestions", [])
                        ],
                    ]
                )
            return {
                "tool_results": [item.tool_result for item in research_results],
                "evidence_items": [item.to_dict() for item in evidence_items],
                "knowledge_summary": knowledge_summary,
                "reflection_notes": reflection_notes,
            }

        tool_calls = current.plan.tool_calls if current.plan else []
        return {"tool_results": await self._tool_executor.execute_many(current, tool_calls)}

    def generate_answer(self, state: AgentState | dict[str, Any]) -> dict[str, Any]:
        current = self._state(state)
        return {"draft_answer": self._answer_generator.generate(current)}

    def reflect(self, state: AgentState | dict[str, Any]) -> dict[str, Any]:
        current = self._state(state)
        return {"reflection_result": self._reflection.evaluate(current, round_index=current.loop_count)}

    def prepare_reflection_retry(self, state: AgentState | dict[str, Any]) -> dict[str, Any]:
        current = self._state(state)
        if current.reflection_result is None:
            return {"loop_count": current.loop_count + 1}
        return {
            "loop_count": current.loop_count + 1,
            "reflection_notes": [
                *current.reflection_result.issues,
                *current.reflection_result.suggestions,
            ],
        }

    def fallback_if_needed(self, state: AgentState | dict[str, Any]) -> dict[str, Any]:
        current = self._state(state)
        if (
            current.intent
            and current.intent.type == "trip_plan"
            and current.reflection_result
            and not current.reflection_result.passed
        ):
            reflection_notes = [
                *current.reflection_result.issues,
                *current.reflection_result.suggestions,
            ]
            fallback_state = self._state({**self._updates(current), "reflection_notes": reflection_notes})
            draft_answer = self._answer_generator.generate(fallback_state, force_fallback=True)
            fallback_state.draft_answer = draft_answer
            return {
                "reflection_notes": reflection_notes,
                "draft_answer": draft_answer,
                "final_answer": draft_answer,
                "reflection_result": self._reflection.evaluate(
                    fallback_state,
                    round_index=self._reflection_max_rounds(),
                ),
            }
        return {"final_answer": current.draft_answer}

    def save_memory(self, state: AgentState | dict[str, Any]) -> dict[str, Any]:
        current = self._state(state)
        self._save_memory(current)
        memory_context = self._memory.snapshot(current.session_id)
        return {
            "memory_context": memory_context,
            "profile": memory_context.get("user_profile", current.profile),
        }

    def build_response(self, state: AgentState | dict[str, Any]) -> dict[str, Any]:
        current = self._state(state)
        return {"response": self._build_response(current).to_dict()}

    def execution_route(self, state: AgentState | dict[str, Any]) -> str:
        current = self._state(state)
        return "research" if current.research_tasks else "tools"

    def reflection_route(self, state: AgentState | dict[str, Any]) -> str:
        current = self._state(state)
        if (
            current.reflection_result
            and not current.reflection_result.passed
            and current.loop_count < self._reflection_max_rounds() - 1
        ):
            return "retry"
        return "finalize"

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

    def _reflection_max_rounds(self) -> int:
        return max(1, int(self._settings.reflection_max_rounds or 1))

    @staticmethod
    def _research_task(payload: dict[str, Any] | ResearchTask) -> ResearchTask:
        if isinstance(payload, ResearchTask):
            return payload
        return ResearchTask(**payload)

    @staticmethod
    def _state(state: AgentState | dict[str, Any]) -> AgentState:
        if isinstance(state, AgentState):
            return state
        field_names = {item.name for item in fields(AgentState)}
        values = {key: value for key, value in dict(state or {}).items() if key in field_names}
        return AgentState(**values)

    @staticmethod
    def _updates(state: AgentState) -> dict[str, Any]:
        return {item.name: getattr(state, item.name) for item in fields(AgentState)}
