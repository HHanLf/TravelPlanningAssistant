from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class ToolCategory(str, Enum):
    INFORMATION = "information"
    PLANNING = "planning"
    TRANSPORT = "transport"
    WEATHER = "weather"
    LODGING = "lodging"
    FOOD = "food"
    SOCIAL = "social"
    MEMORY = "memory"


class ExecutionStage(str, Enum):
    UNDERSTAND = "understand"
    PLAN = "plan"
    EXECUTE = "execute"
    RESPOND = "respond"
    REFLECT = "reflect"


@dataclass(slots=True)
class ConversationTurn:
    role: str
    content: str
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass(slots=True)
class UserProfile:
    departure: str | None = None
    destination: str | None = None
    budget: int | None = None
    days: int | None = None
    companions: int | None = None
    preferences: list[str] = field(default_factory=list)
    date_range: dict[str, str] | None = None
    constraints: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class AgentContext:
    session_id: str
    latest_message: str
    history: list[ConversationTurn] = field(default_factory=list)
    user_profile: UserProfile = field(default_factory=UserProfile)
    memory: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ToolSpec:
    name: str
    description: str
    category: ToolCategory
    required_fields: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)


@dataclass(slots=True)
class ToolCall:
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    reason: str = ""


@dataclass(slots=True)
class ToolResult:
    name: str
    success: bool
    payload: dict[str, Any]
    error: str | None = None


@dataclass(slots=True)
class ExecutionStep:
    stage: ExecutionStage
    title: str
    detail: str


@dataclass(slots=True)
class ExecutionPlan:
    intent: dict[str, Any]
    tool_calls: list[ToolCall] = field(default_factory=list)
    missing_information: list[str] = field(default_factory=list)
    steps: list[ExecutionStep] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "intent": self.intent,
            "tool_calls": [asdict(item) for item in self.tool_calls],
            "missing_information": list(self.missing_information),
            "steps": [
                {"stage": item.stage.value, "title": item.title, "detail": item.detail}
                for item in self.steps
            ],
        }


@dataclass(slots=True)
class AgentResponse:
    answer: str
    intent: dict[str, Any]
    plan: dict[str, Any]
    tool_results: dict[str, Any]
    memory_context: dict[str, Any]
    reflection_result: dict[str, Any]
    retrieved_docs: list[dict[str, Any]] = field(default_factory=list)
    audio_transcript: str | None = None
