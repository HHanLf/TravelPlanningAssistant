from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class IntentResult:
    type: str
    confidence: float = 0.7
    requires_tools: bool = False
    preferred_tools: list[str] = field(default_factory=list)
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class PlanningProblem:
    origin: str | None = None
    destination: str | None = None
    days: int | None = None
    budget: int | None = None
    group_size: int | None = None
    date_range: dict[str, str] | None = None
    preferences: list[str] = field(default_factory=list)
    constraints: dict[str, Any] = field(default_factory=dict)
    missing_info: list[str] = field(default_factory=list)
    assumptions: list[str] = field(default_factory=list)
    explicit_fields: list[str] = field(default_factory=list)
    context_scope: str = "trip"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ToolCall:
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ToolResult:
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    success: bool = False
    payload: dict[str, Any] = field(default_factory=dict)
    summary: str = ""
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ExecutionPlan:
    response_goal: str
    missing_info: list[str] = field(default_factory=list)
    tool_calls: list[ToolCall] = field(default_factory=list)
    assumptions: list[str] = field(default_factory=list)
    steps: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "response_goal": self.response_goal,
            "missing_info": list(self.missing_info),
            "tool_calls": [item.to_dict() for item in self.tool_calls],
            "assumptions": list(self.assumptions),
            "steps": list(self.steps),
            "need_tool": bool(self.tool_calls),
        }


@dataclass(slots=True)
class ReflectionResult:
    passed: bool
    score: int
    issues: list[str] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)
    round: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class AgentResponse:
    session_id: str
    answer: str
    intent: dict[str, Any]
    plan: dict[str, Any]
    tool_results: dict[str, Any]
    memory_context: dict[str, Any]
    reflection_result: dict[str, Any]
    problem: dict[str, Any] = field(default_factory=dict)
    knowledge_summary: dict[str, Any] = field(default_factory=dict)
    research_tasks: list[dict[str, Any]] = field(default_factory=list)
    evidence_items: list[dict[str, Any]] = field(default_factory=list)
    retrieved_docs: list[dict[str, Any]] = field(default_factory=list)
    profile: dict[str, Any] = field(default_factory=dict)
    multimodal_summary: str | None = None
    audio_transcript: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class AgentState:
    session_id: str
    raw_message: str
    effective_message: str = ""
    audio_transcript: str | None = None
    image_summary: str | None = None
    multimodal_summary: str | None = None
    memory_context: dict[str, Any] = field(default_factory=dict)
    profile: dict[str, Any] = field(default_factory=dict)
    intent: IntentResult | None = None
    problem: PlanningProblem | None = None
    plan: ExecutionPlan | None = None
    research_tasks: list[dict[str, Any]] = field(default_factory=list)
    evidence_items: list[dict[str, Any]] = field(default_factory=list)
    knowledge_summary: dict[str, Any] = field(default_factory=dict)
    tool_results: list[ToolResult] = field(default_factory=list)
    retrieved_docs: list[dict[str, Any]] = field(default_factory=list)
    draft_answer: str = ""
    final_answer: str = ""
    reflection_result: ReflectionResult | None = None
    reflection_notes: list[str] = field(default_factory=list)
    loop_count: int = 0
    response: dict[str, Any] = field(default_factory=dict)
