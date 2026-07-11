from __future__ import annotations

from typing import Any

from app.agent.answer_generator import AnswerGenerator
from app.agent.graph import TravelAgentGraph
from app.agent.input_normalizer import InputNormalizer
from app.agent.planner import Planner
from app.agent.problem_analyzer import ProblemAnalyzer
from app.agent.reflection import ReflectionAgent
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
    """Compatibility wrapper around the LangGraph production workflow."""

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
        self._graph = TravelAgentGraph(
            memory_service=memory_service,
            tool_registry=tool_registry,
            input_normalizer=input_normalizer,
            intent_analyzer=intent_analyzer,
            problem_analyzer=problem_analyzer,
            planner=planner,
            tool_executor=tool_executor,
            answer_generator=answer_generator,
            reflection_agent=reflection_agent,
            research_planner=research_planner,
            research_executor=research_executor,
            evidence_normalizer=evidence_normalizer,
            information_fusion=information_fusion,
            knowledge_summarizer=knowledge_summarizer,
            research_reflection=research_reflection,
        )

    async def handle(
        self,
        session_id: str,
        message: str,
        image_path: str | None = None,
        audio_path: str | None = None,
        multimodal: bool = False,
    ) -> dict[str, Any]:
        return await self._graph.handle(
            session_id=session_id,
            message=message,
            image_path=image_path,
            audio_path=audio_path,
            multimodal=multimodal,
        )
