from __future__ import annotations

from app.domain.models import AgentContext, ToolCategory, ToolResult, ToolSpec
from app.tools.base import BaseTool


class WeatherTool(BaseTool):
    spec = ToolSpec(
        name="weather_lookup",
        description="查询目的地天气并输出出行建议",
        category=ToolCategory.WEATHER,
        required_fields=["destination"],
        tags=["weather", "forecast"],
    )

    async def execute(self, context: AgentContext, arguments: dict) -> ToolResult:
        destination = arguments.get("destination") or context.user_profile.destination or "目的地"
        return ToolResult(
            name=self.spec.name,
            success=True,
            payload={
                "destination": destination,
                "forecast": "未来 3 天以多云到晴为主，适合安排城市步行和轻户外活动。",
                "tips": ["建议携带轻便外套", "中午紫外线较强，注意补水和防晒"],
            },
        )


class MapTool(BaseTool):
    spec = ToolSpec(
        name="route_planning",
        description="生成城市内景点串联与交通建议",
        category=ToolCategory.TRANSPORT,
        required_fields=["destination"],
        tags=["route", "map", "transport"],
    )

    async def execute(self, context: AgentContext, arguments: dict) -> ToolResult:
        destination = arguments.get("destination") or context.user_profile.destination or "目的地"
        return ToolResult(
            name=self.spec.name,
            success=True,
            payload={
                "destination": destination,
                "route_summary": "建议按区域分天游玩，优先串联核心景区，减少往返折返。",
                "segments": [
                    "第一天聚焦核心城区与地标景点，晚上安排周边餐饮。",
                    "第二天围绕特色街区、美食与夜景展开，兼顾休闲体验。",
                ],
            },
        )


class PoiTool(BaseTool):
    spec = ToolSpec(
        name="poi_recommendation",
        description="根据偏好推荐景点、美食与城市体验",
        category=ToolCategory.INFORMATION,
        required_fields=["destination"],
        tags=["poi", "food", "attraction"],
    )

    async def execute(self, context: AgentContext, arguments: dict) -> ToolResult:
        destination = arguments.get("destination") or context.user_profile.destination or "目的地"
        preferences = arguments.get("preferences") or context.user_profile.preferences
        recommendations = [
            f"{destination}地标景点",
            f"{destination}本地美食街区",
            f"{destination}文化体验区域",
        ]
        if preferences:
            recommendations.append(f"结合偏好 {', '.join(preferences)} 的主题路线")

        return ToolResult(
            name=self.spec.name,
            success=True,
            payload={
                "destination": destination,
                "recommendations": recommendations,
            },
        )
