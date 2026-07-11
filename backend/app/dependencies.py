from __future__ import annotations

from functools import lru_cache

from app.agent.answer_generator import AnswerGenerator
from app.agent.input_normalizer import InputNormalizer
from app.agent.orchestrator import TravelAgentOrchestrator
from app.agent.planner import Planner
from app.agent.problem_analyzer import ProblemAnalyzer
from app.agent.reflection import ReflectionAgent
from app.agent.travel_agent import TravelAgent
from app.core.config import get_settings
from app.services.intent import IntentAnalyzer
from app.services.memory import InMemoryRepository, MemoryService, RedisBackedRepository
from app.tools.amap_tool import TencentMapTool
from app.tools.ctrip_tool import CtripTool
from app.tools.executor import ToolExecutor
from app.tools.registry import ToolRegistry
from app.tools.search_tool import SearchTool
from app.tools.travel_tools import (
    HotelSearchTool,
    PlaceSearchTool,
    RestaurantRecommendationTool,
    RoutePlanningTool,
    WeatherLookupTool,
    WebSearchTool,
    XiaohongshuSearchTool,
)
from app.tools.weather_tool import WeatherTool
from app.tools.xiaohongshu_tool import XiaohongshuTool


@lru_cache
def get_memory_service() -> MemoryService:
    settings = get_settings()
    fallback = InMemoryRepository(settings.conversation_history_limit)
    if settings.use_redis_memory:
        return MemoryService(repository=RedisBackedRepository(settings, fallback=fallback))
    return MemoryService(repository=fallback)


@lru_cache
def get_tool_registry() -> ToolRegistry:
    settings = get_settings()
    registry = ToolRegistry()

    map_tool = TencentMapTool(
        api_key=settings.tencent_map_api_key,
        base_url=settings.tencent_map_base_url,
        place_search_path=settings.tencent_map_place_search_path,
        geocoder_path=settings.tencent_map_geocoder_path,
        direction_path_prefix=settings.tencent_map_direction_path_prefix,
    )
    weather_client = WeatherTool(
        api_key=settings.weather_api_key or settings.tencent_map_api_key,
        base_url=settings.weather_base_url,
        geocoder_path=settings.tencent_map_geocoder_path,
        weather_path=settings.tencent_map_weather_path,
    )
    ctrip_client = CtripTool(api_key=settings.ctrip_api_key, base_url=settings.ctrip_base_url, map_tool=map_tool)
    xiaohongshu_client = XiaohongshuTool(
        token=settings.justoneapi_token,
        base_url=settings.justoneapi_base_url,
        search_note_path=settings.justoneapi_xiaohongshu_search_note_path,
    )

    registry.register_many(
        [
            WeatherLookupTool(weather_client),
            RoutePlanningTool(map_tool),
            PlaceSearchTool(map_tool),
            RestaurantRecommendationTool(map_tool),
            HotelSearchTool(ctrip_client),
            XiaohongshuSearchTool(xiaohongshu_client),
            WebSearchTool(SearchTool(api_key=settings.search_api_key, base_url=settings.search_base_url)),
        ]
    )
    return registry


@lru_cache
def get_travel_agent() -> TravelAgent:
    memory_service = get_memory_service()
    registry = get_tool_registry()
    orchestrator = TravelAgentOrchestrator(
        memory_service=memory_service,
        tool_registry=registry,
        input_normalizer=InputNormalizer(),
        intent_analyzer=IntentAnalyzer(),
        problem_analyzer=ProblemAnalyzer(),
        planner=Planner(),
        tool_executor=ToolExecutor(registry),
        answer_generator=AnswerGenerator(),
        reflection_agent=ReflectionAgent(),
    )
    return TravelAgent(orchestrator)
