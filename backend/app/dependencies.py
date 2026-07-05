from __future__ import annotations

from functools import lru_cache

from backend.app.agent.context_manager import ContextManager
from backend.app.agent.planner import Planner
from backend.app.agent.profile_updater import ProfileUpdater
from backend.app.agent.response_builder import ResponseBuilder
from backend.app.agent.travel_agent import TravelAgent
from backend.app.core.config import get_settings
from backend.app.services.intent import IntentAnalyzer
from backend.app.services.memory import InMemoryRepository, MemoryService
from backend.app.tools.executor import ToolExecutor
from backend.app.tools.implementations import MapTool, PoiTool, WeatherTool
from backend.app.tools.registry import ToolRegistry


@lru_cache
def get_memory_service() -> MemoryService:
    settings = get_settings()
    repository = InMemoryRepository()
    if settings.use_redis_memory:
        # 当前保留扩展点，后续可替换为 RedisRepository。
        repository = InMemoryRepository()
    return MemoryService(repository=repository)


@lru_cache
def get_tool_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register_many([WeatherTool(), MapTool(), PoiTool()])
    return registry


@lru_cache
def get_travel_agent() -> TravelAgent:
    memory_service = get_memory_service()
    registry = get_tool_registry()
    return TravelAgent(
        memory_service=memory_service,
        intent_analyzer=IntentAnalyzer(),
        context_manager=ContextManager(memory_service),
        profile_updater=ProfileUpdater(),
        planner=Planner(),
        tool_executor=ToolExecutor(registry),
        response_builder=ResponseBuilder(),
    )
