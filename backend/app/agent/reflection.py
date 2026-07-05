from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class ReflectionOutcome:
    score: int
    reason: str
    passed: bool


class ReflectionAgent:
    def evaluate(self, state: dict[str, Any]) -> ReflectionOutcome:
        answer = state.get("draft_answer") or state.get("final_answer", "")
        docs = state.get("retrieved_docs", [])
        tool_results = state.get("tool_results", {})
        score = 70
        if answer:
            score += 15
        if docs:
            score += 5
        if tool_results or not state.get("need_tool"):
            score += 5
        passed = score >= 80
        reason = "结果通过反思检查" if passed else "结果存在遗漏，需要重新规划"
        return ReflectionOutcome(score=score, reason=reason, passed=passed)
