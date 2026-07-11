from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class ResearchTask:
    id: str
    category: str
    tool_name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    query: str = ""
    priority: int = 50
    required: bool = False
    expected_output: str = ""
    source_policy: str = ""
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class EvidenceItem:
    id: str
    category: str
    source: str
    source_type: str
    title: str
    content: str = ""
    url: str | None = None
    location: dict[str, Any] | None = None
    timestamp: str | None = None
    confidence: float = 0.5
    task_id: str = ""
    raw: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ResearchResult:
    task: ResearchTask
    tool_result: Any

    def to_dict(self) -> dict[str, Any]:
        task_payload = self.task.to_dict()
        if hasattr(self.tool_result, "to_dict"):
            result_payload = self.tool_result.to_dict()
        else:
            result_payload = self.tool_result
        return {"task": task_payload, "tool_result": result_payload}


@dataclass(slots=True)
class KnowledgeSummary:
    destination: str = ""
    attractions: list[dict[str, Any]] = field(default_factory=list)
    restaurants: list[dict[str, Any]] = field(default_factory=list)
    hotels: list[dict[str, Any]] = field(default_factory=list)
    transport: list[dict[str, Any]] = field(default_factory=list)
    weather: dict[str, Any] = field(default_factory=dict)
    xiaohongshu_insights: list[str] = field(default_factory=list)
    web_findings: list[dict[str, Any]] = field(default_factory=list)
    conflicts: list[str] = field(default_factory=list)
    source_coverage: dict[str, int] = field(default_factory=dict)
    confidence_score: float = 0.0
    research_notes: list[str] = field(default_factory=list)
    evidence_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

