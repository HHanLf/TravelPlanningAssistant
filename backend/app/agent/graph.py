from __future__ import annotations

from typing import Any

from langgraph.graph import END, StateGraph

from backend.app.agent.executor import Executor
from backend.app.agent.planner import Planner
from backend.app.agent.reflection import ReflectionAgent
from backend.app.agent.state import AgentState


class TravelAgentGraph:
    def __init__(self, services: dict[str, Any]) -> None:
        self.services = services
        self.planner = Planner()
        self.executor = Executor()
        self.reflection = ReflectionAgent()
        self.graph = self._build_graph().compile()

    def _build_graph(self) -> StateGraph:
        graph = StateGraph(AgentState)
        graph.add_node("intent", self.intent_node)
        graph.add_node("memory", self.memory_node)
        graph.add_node("plan", self.planner_node)
        graph.add_node("rag", self.rag_node)
        graph.add_node("tool", self.tool_node)
        graph.add_node("execute", self.executor_node)
        graph.add_node("reflect", self.reflection_node)
        graph.add_node("answer", self.answer_node)
        graph.set_entry_point("intent")
        graph.add_edge("intent", "memory")
        graph.add_edge("memory", "plan")
        graph.add_conditional_edges("plan", self.decide_next, {"rag": "rag", "tool": "tool", "execute": "execute"})
        graph.add_edge("rag", "execute")
        graph.add_edge("tool", "execute")
        graph.add_edge("execute", "reflect")
        graph.add_conditional_edges("reflect", self.reflect_next, {"plan": "plan", "answer": "answer"})
        graph.add_edge("answer", END)
        return graph

    def intent_node(self, state: AgentState) -> AgentState:
        return {**state, "intent": self.planner.build_intent(state["question"], state.get("memory_context", {}))}

    def memory_node(self, state: AgentState) -> AgentState:
        snapshot = self.services["memory"].read(state["session_id"])
        memory_context = {"short_memory": snapshot.short_memory, "long_memory": snapshot.long_memory, "profile": snapshot.user_profile}
        return {**state, "chat_history": snapshot.chat_history, "user_profile": snapshot.user_profile, "memory_context": memory_context}

    def planner_node(self, state: AgentState) -> AgentState:
        plan = self.planner.make_plan(state.get("intent", {}), state.get("memory_context", {}))
        return {**state, "plan": plan, "need_rag": plan["needs_rag"], "need_tool": plan["needs_tool"], "need_memory": plan["needs_memory"]}

    def decide_next(self, state: AgentState) -> str:
        if state.get("need_rag"):
            return "rag"
        if state.get("need_tool"):
            return "tool"
        return "execute"

    def rag_node(self, state: AgentState) -> AgentState:
        return {**state, "retrieved_docs": self.services["retriever"].retrieve(state["question"])}

    def tool_node(self, state: AgentState) -> AgentState:
        return {**state, "tool_results": self.services["tool_router"].execute(state)}

    def executor_node(self, state: AgentState) -> AgentState:
        return {**state, "draft_answer": self.executor.synthesize(state)}

    def reflection_node(self, state: AgentState) -> AgentState:
        result = self.reflection.evaluate(state)
        return {**state, "reflection_result": result.__dict__}

    def reflect_next(self, state: AgentState) -> str:
        loop_count = state.get("loop_count", 0)
        if not state.get("reflection_result", {}).get("passed") and loop_count < self.services["settings"].reflection_max_rounds - 1:
            return "plan"
        return "answer"

    def answer_node(self, state: AgentState) -> AgentState:
        final = state.get("draft_answer", "")
        memory = self.services["memory"]
        memory.append_message(state["session_id"], "user", state["question"])
        memory.append_message(state["session_id"], "assistant", final)
        return {**state, "final_answer": final}
