from __future__ import annotations

from typing import TypedDict


class AgentState(TypedDict, total=False):
    session_id: str
    question: str
    chat_history: list[tuple[str, str]]
    user_profile: dict
    intent: dict
    plan: dict
    need_rag: bool
    need_tool: bool
    need_memory: bool
    retrieved_docs: list[dict]
    tool_results: dict
    memory_context: dict
    draft_answer: str
    reflection_result: dict
    final_answer: str
    loop_count: int
