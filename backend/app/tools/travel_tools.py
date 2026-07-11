from __future__ import annotations

from dataclasses import asdict
from typing import Any

from app.agent.state import AgentState, ToolResult
from app.agent.travel_semantics import get_poi_theme_profile, infer_poi_theme, score_poi_for_theme
from app.domain.models import ToolCategory, ToolSpec
from app.tools.amap_tool import TencentMapTool
from app.tools.base import BaseTool
from app.tools.ctrip_tool import CtripTool
from app.tools.search_tool import SearchTool
from app.tools.weather_tool import WeatherTool
from app.tools.xiaohongshu_tool import XiaohongshuTool


class WeatherLookupTool(BaseTool):
    spec = ToolSpec(
        name="weather_lookup",
        description="查询目的地天气并输出出行建议",
        category=ToolCategory.WEATHER,
        required_fields=["destination"],
        tags=["weather", "forecast"],
    )

    def __init__(self, client: WeatherTool) -> None:
        self._client = client

    async def execute(self, state: AgentState, arguments: dict[str, Any]) -> ToolResult:
        del state
        city = str(arguments.get("destination") or "")
        payload = self._client.get_weather(
            city=city,
            day_offset=int(arguments.get("day_offset") or 0),
            date_label=str(arguments.get("date_label") or ""),
        )
        return self._result(arguments, payload, self._summary(payload))

    def _result(self, arguments: dict[str, Any], payload: dict[str, Any], summary: str) -> ToolResult:
        error = str(payload.get("error") or "") or None
        return ToolResult(
            name=self.spec.name,
            arguments=arguments,
            success=error is None,
            payload=payload,
            summary=summary,
            error=error,
        )

    @staticmethod
    def _summary(payload: dict[str, Any]) -> str:
        if payload.get("error"):
            return str(payload["error"])
        return f"{payload.get('resolved_city') or payload.get('city')} {payload.get('date_label')}: {payload.get('forecast')}；{payload.get('recommendation')}"


class RoutePlanningTool(BaseTool):
    spec = ToolSpec(
        name="route_planning",
        description="调用地图能力规划起点到目的地的交通路线",
        category=ToolCategory.TRANSPORT,
        required_fields=["origin", "destination"],
        tags=["route", "map", "transport"],
    )

    def __init__(self, client: TencentMapTool) -> None:
        self._client = client

    async def execute(self, state: AgentState, arguments: dict[str, Any]) -> ToolResult:
        del state
        if not self._client.api_key:
            return ToolResult(
                name=self.spec.name,
                arguments=arguments,
                success=False,
                payload={"provider": "tencent_map", "fallback": False},
                summary="tencent map api key is not configured",
                error="tencent map api key is not configured",
            )
        payload = self._client.route_plan(
            origin=str(arguments.get("origin") or ""),
            destination=str(arguments.get("destination") or ""),
            mode=str(arguments.get("mode") or "driving"),
        )
        error = str(payload.get("error") or "") or None
        summary = error or self._summary(payload)
        return ToolResult(
            name=self.spec.name,
            arguments=arguments,
            success=error is None,
            payload=payload,
            summary=summary,
            error=error,
        )

    @staticmethod
    def _summary(payload: dict[str, Any]) -> str:
        distance = payload.get("distance_km")
        duration = payload.get("duration_minutes")
        mode = payload.get("mode") or "route"
        pieces = [f"{payload.get('origin')} 到 {payload.get('destination')}"]
        if distance:
            pieces.append(f"约 {distance} 公里")
        if duration:
            pieces.append(f"约 {duration} 分钟")
        pieces.append(f"方式：{mode}")
        return "，".join(pieces)


class PlaceSearchTool(BaseTool):
    spec = ToolSpec(
        name="place_search",
        description="按城市和关键词检索景点、街区、博物馆等地点",
        category=ToolCategory.INFORMATION,
        required_fields=["destination", "keyword"],
        tags=["poi", "map", "attraction"],
    )

    def __init__(self, client: TencentMapTool) -> None:
        self._client = client

    async def execute(self, state: AgentState, arguments: dict[str, Any]) -> ToolResult:
        city = str(arguments.get("destination") or "")
        keyword = str(arguments.get("keyword") or "景点")
        if not self._client.api_key:
            return ToolResult(
                name=self.spec.name,
                arguments=arguments,
                success=False,
                payload={"provider": "tencent_map", "city": city, "keyword": keyword, "places": []},
                summary="tencent map api key is not configured",
                error="tencent map api key is not configured",
            )
        semantic_theme = self._semantic_theme(state, keyword)
        query_keywords = self._query_keywords(keyword, semantic_theme)
        try:
            raw_places = self._search_many(city, query_keywords)
        except Exception as exc:  # noqa: BLE001
            error = str(exc)
            return ToolResult(
                name=self.spec.name,
                arguments=arguments,
                success=False,
                payload={
                    "provider": "tencent_map",
                    "city": city,
                    "keyword": keyword,
                    "query_keywords": query_keywords,
                    "semantic_theme": semantic_theme,
                    "places": [],
                    "error": error,
                },
                summary=error,
                error=error,
            )
        places, filtered_out = self._filter_and_rank(raw_places, semantic_theme)
        payload = {
            "provider": "tencent_map",
            "city": city,
            "keyword": keyword,
            "query_keywords": query_keywords,
            "semantic_theme": semantic_theme,
            "places": places,
            "filtered_out": filtered_out,
        }
        summary = self._summary(city, keyword, places)
        return ToolResult(
            name=self.spec.name,
            arguments=arguments,
            success=bool(places),
            payload=payload,
            summary=summary,
            error=None if places else "未检索到地点结果或地图 API 不可用",
        )

    def _semantic_theme(self, state: AgentState, keyword: str) -> str:
        if self.spec.name != "place_search":
            return ""
        problem = state.problem
        if problem is None:
            return infer_poi_theme(text=keyword)
        return infer_poi_theme(
            text=keyword,
            preferences=problem.preferences,
            constraints=problem.constraints,
        )

    def _query_keywords(self, keyword: str, semantic_theme: str) -> list[str]:
        profile = get_poi_theme_profile(semantic_theme)
        if profile:
            return self._dedupe_strings([keyword, *profile.query_keywords])
        return [keyword]

    def _search_many(self, city: str, keywords: list[str]) -> list[dict[str, Any]]:
        seen: set[str] = set()
        results: list[dict[str, Any]] = []
        for keyword in keywords:
            for item in self._client.search_place(keyword, city):
                key = str(item.get("id") or item.get("name") or "") + "|" + str(item.get("address") or "")
                if not key.strip("|") or key in seen:
                    continue
                seen.add(key)
                normalized = dict(item)
                normalized.setdefault("matched_keyword", keyword)
                results.append(normalized)
        return results

    def _filter_and_rank(self, places: list[dict[str, Any]], semantic_theme: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        if not get_poi_theme_profile(semantic_theme):
            return places[:8], []

        accepted: list[tuple[int, dict[str, Any]]] = []
        rejected: list[dict[str, Any]] = []
        for place in places:
            score, reason = score_poi_for_theme(place, semantic_theme)
            normalized = {**place, "semantic_score": score}
            if score <= 0:
                rejected.append({**normalized, "filter_reason": reason})
                continue
            accepted.append((score, normalized))

        accepted.sort(key=lambda item: (-item[0], str(item[1].get("name") or "")))
        return [item for _, item in accepted[:8]], rejected[:12]

    @staticmethod
    def _dedupe_strings(values: list[str]) -> list[str]:
        seen: set[str] = set()
        result: list[str] = []
        for value in values:
            normalized = str(value or "").strip()
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            result.append(normalized)
        return result
    @staticmethod
    def _summary(city: str, keyword: str, places: list[dict[str, Any]]) -> str:
        if not places:
            return f"{city} 暂未检索到 {keyword} 结果。"
        names = "、".join(str(item.get("name") or "") for item in places[:5] if item.get("name"))
        return f"{city} {keyword} 候选：{names}"


class RestaurantRecommendationTool(PlaceSearchTool):
    spec = ToolSpec(
        name="restaurant_recommendation",
        description="按城市检索餐厅、小吃和美食街区",
        category=ToolCategory.FOOD,
        required_fields=["destination"],
        tags=["food", "restaurant", "map"],
    )

    async def execute(self, state: AgentState, arguments: dict[str, Any]) -> ToolResult:
        city = str(arguments.get("destination") or "")
        keyword = str(arguments.get("keyword") or "美食")
        if not self._client.api_key:
            return ToolResult(
                name=self.spec.name,
                arguments=arguments,
                success=False,
                payload={"provider": "tencent_map", "city": city, "keyword": keyword, "places": []},
                summary="tencent map api key is not configured",
                error="tencent map api key is not configured",
            )

        query_keywords = self._restaurant_keywords(city, keyword, state)
        try:
            raw_places = self._search_many(city, query_keywords)
        except Exception as exc:  # noqa: BLE001
            error = str(exc)
            return ToolResult(
                name=self.spec.name,
                arguments=arguments,
                success=False,
                payload={
                    "provider": "tencent_map",
                    "city": city,
                    "keyword": keyword,
                    "query_keywords": query_keywords,
                    "places": [],
                    "error": error,
                },
                summary=error,
                error=error,
            )

        places, filtered_out = self._filter_and_rank_restaurants(raw_places, keyword)
        payload = {
            "provider": "tencent_map",
            "city": city,
            "keyword": keyword,
            "query_keywords": query_keywords,
            "places": places,
            "filtered_out": filtered_out,
        }
        summary = self._summary(city, keyword, places)
        return ToolResult(
            name=self.spec.name,
            arguments=arguments,
            success=bool(places),
            payload=payload,
            summary=summary,
            error=None if places else "未检索到餐厅结果或地图 API 不可用",
        )

    def _restaurant_keywords(self, city: str, keyword: str, state: AgentState) -> list[str]:
        text = " ".join(
            [
                keyword,
                state.effective_message or "",
                " ".join(state.problem.preferences if state.problem else []),
            ]
        )
        explicit = [keyword] if keyword and keyword != "美食" else []
        if any(token in text for token in ("咖啡", "下午茶")):
            return self._dedupe_strings([*explicit, "咖啡", "下午茶", "甜品"])
        if any(token in text for token in ("早餐", "早饭", "早点")):
            return self._dedupe_strings([*explicit, "早餐", "小吃", "面馆"])
        if any(token in text for token in ("夜宵", "夜市")):
            return self._dedupe_strings([*explicit, "夜宵", "夜市", "小吃"])
        if any(token in text for token in ("火锅", "羊肉", "羊锅", "锅")):
            return self._dedupe_strings([*explicit, keyword, "火锅", "羊肉", "当地特色餐厅"])

        city_specials = {
            "杭州": ["杭帮菜", "杭州小吃", "面馆", "早餐", "西湖附近餐厅", "游客友好餐厅"],
            "广州": ["早茶", "粤菜", "烧腊", "肠粉", "老字号餐厅", "茶餐厅"],
            "成都": ["川菜", "火锅", "串串", "小吃", "面馆", "苍蝇馆子"],
            "北京": ["北京菜", "烤鸭", "炸酱面", "胡同小吃", "老字号餐厅"],
            "上海": ["本帮菜", "生煎", "小笼", "面馆", "老字号餐厅"],
            "厦门": ["沙茶面", "海鲜", "闽南小吃", "姜母鸭", "中山路美食"],
            "西安": ["肉夹馍", "羊肉泡馍", "面馆", "小吃", "回民街周边"],
            "重庆": ["火锅", "小面", "江湖菜", "串串", "夜宵"],
        }
        defaults = city_specials.get(city, ["当地特色餐厅", "小吃", "老字号餐厅", "游客友好餐厅", "面馆"])
        return self._dedupe_strings([*explicit, *defaults])

    def _filter_and_rank_restaurants(
        self,
        places: list[dict[str, Any]],
        keyword: str,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        accepted: list[tuple[int, dict[str, Any]]] = []
        filtered_out: list[dict[str, Any]] = []
        cuisine_counts: dict[str, int] = {}

        for index, place in enumerate(places):
            score, reason, cuisine = self._restaurant_score(place, keyword, index)
            normalized = {**place, "restaurant_score": score, "cuisine_hint": cuisine}
            if score < 0:
                filtered_out.append({**normalized, "filter_reason": reason})
                continue
            accepted.append((score, normalized))

        accepted.sort(key=lambda item: (-item[0], str(item[1].get("name") or "")))
        ranked: list[dict[str, Any]] = []
        overflow: list[dict[str, Any]] = []
        for score, place in accepted:
            cuisine = str(place.get("cuisine_hint") or "综合")
            limit = 2 if cuisine in {"羊肉锅物", "火锅"} else 3
            if cuisine_counts.get(cuisine, 0) >= limit:
                overflow.append({**place, "filter_reason": f"{cuisine} 类型结果过多，已为多样性降权"})
                continue
            cuisine_counts[cuisine] = cuisine_counts.get(cuisine, 0) + 1
            ranked.append(place)
            if len(ranked) >= 8:
                break

        filtered_out.extend(overflow[:12])
        return ranked, filtered_out[:16]

    @staticmethod
    def _restaurant_score(place: dict[str, Any], keyword: str, index: int) -> tuple[int, str, str]:
        name = str(place.get("name") or "")
        category = str(place.get("category") or "")
        address = str(place.get("address") or "")
        matched_keyword = str(place.get("matched_keyword") or "")
        combined = f"{name} {category} {address} {matched_keyword}"

        cuisine = RestaurantRecommendationTool._cuisine_hint(combined)
        score = 80 - min(index, 12)
        if any(token in category for token in ("美食", "餐饮", "中餐", "小吃", "餐厅")):
            score += 12
        if matched_keyword and matched_keyword != "美食":
            score += 10
        if any(token in combined for token in ("老字号", "特色", "本帮", "杭帮", "小吃", "面馆", "早餐")):
            score += 8
        if keyword and keyword != "美食" and keyword in combined:
            score += 12

        if "羊锅" in combined or "羊肉锅" in combined:
            score -= 18
            if keyword == "美食":
                score -= 12
        if any(token in combined for token in ("公司", "批发", "市场", "培训", "学校", "商行")):
            return -1, "非餐饮消费地点", cuisine

        return score, "", cuisine

    @staticmethod
    def _cuisine_hint(text: str) -> str:
        mapping = (
            ("羊肉锅", ("羊锅", "羊肉锅", "羊蝎子")),
            ("火锅", ("火锅", "串串")),
            ("小吃", ("小吃", "生煎", "肉夹馍", "泡馍", "肠粉", "沙茶面")),
            ("面馆", ("面馆", "面", "粉")),
            ("早餐", ("早餐", "早茶", "早点")),
            ("咖啡甜品", ("咖啡", "甜品", "下午茶")),
            ("地方菜", ("杭帮", "本帮", "粤菜", "川菜", "北京菜", "闽南")),
        )
        for label, tokens in mapping:
            if any(token in text for token in tokens):
                return label
        return "综合餐饮"


class HotelSearchTool(BaseTool):
    spec = ToolSpec(
        name="hotel_search",
        description="检索目的地酒店候选并按预算做粗筛",
        category=ToolCategory.LODGING,
        required_fields=["destination"],
        tags=["hotel", "lodging"],
    )

    def __init__(self, client: CtripTool) -> None:
        self._client = client

    async def execute(self, state: AgentState, arguments: dict[str, Any]) -> ToolResult:
        del state
        city = str(arguments.get("destination") or "")
        budget = arguments.get("budget_per_night") or arguments.get("budget")
        if not self._client.map_tool.api_key:
            return ToolResult(
                name=self.spec.name,
                arguments=arguments,
                success=False,
                payload={"provider": "ctrip_or_map", "city": city, "hotels": []},
                summary="tencent map api key is not configured",
                error="tencent map api key is not configured",
            )
        try:
            hotels = self._client.search_hotels(city=city, budget=self._safe_int(budget))
        except Exception as exc:  # noqa: BLE001
            error = str(exc)
            return ToolResult(
                name=self.spec.name,
                arguments=arguments,
                success=False,
                payload={"provider": "ctrip_or_map", "city": city, "hotels": [], "error": error},
                summary=error,
                error=error,
            )
        hotel_payload = [asdict(item) for item in hotels]
        payload = {"provider": "ctrip_or_map", "city": city, "hotels": hotel_payload}
        summary = self._summary(city, hotel_payload)
        return ToolResult(
            name=self.spec.name,
            arguments=arguments,
            success=bool(hotel_payload),
            payload=payload,
            summary=summary,
            error=None if hotel_payload else "未检索到酒店结果或地图 API 不可用",
        )

    @staticmethod
    def _summary(city: str, hotels: list[dict[str, Any]]) -> str:
        if not hotels:
            return f"{city} 暂未检索到酒店候选。"
        names = "、".join(str(item.get("name") or "") for item in hotels[:5] if item.get("name"))
        return f"{city} 酒店候选：{names}"

    @staticmethod
    def _safe_int(value: Any) -> int | None:
        try:
            if value in (None, ""):
                return None
            return int(value)
        except (TypeError, ValueError):
            return None


class XiaohongshuSearchTool(BaseTool):
    spec = ToolSpec(
        name="xiaohongshu_search",
        description="检索小红书旅行笔记并提炼高频经验",
        category=ToolCategory.SOCIAL,
        required_fields=["keyword"],
        tags=["social", "notes", "ugc"],
    )

    def __init__(self, client: XiaohongshuTool) -> None:
        self._client = client

    async def execute(self, state: AgentState, arguments: dict[str, Any]) -> ToolResult:
        del state
        payload = self._client.search_notes(
            keyword=str(arguments.get("keyword") or ""),
            page=int(arguments.get("page") or 1),
            sort_type=str(arguments.get("sort_type") or arguments.get("sortType") or "general"),
            note_type=str(arguments.get("note_type") or arguments.get("noteType") or "ALL"),
            time_filter=str(arguments.get("time_filter") or arguments.get("timeFilter") or "ALL"),
            limit=int(arguments.get("limit") or 5),
        )
        error = str(payload.get("error") or "") or None
        return ToolResult(
            name=self.spec.name,
            arguments=arguments,
            success=error is None and bool(payload.get("notes")),
            payload=payload,
            summary=str(payload.get("summary") or error or ""),
            error=error,
        )


class WebSearchTool(BaseTool):
    spec = ToolSpec(
        name="web_search",
        description="通用搜索工具，用于补充公开信息",
        category=ToolCategory.INFORMATION,
        required_fields=["query"],
        tags=["search"],
    )

    def __init__(self, client: SearchTool) -> None:
        self._client = client

    async def execute(self, state: AgentState, arguments: dict[str, Any]) -> ToolResult:
        del state
        query = str(arguments.get("query") or "")
        results = self._client.search(query)
        payload = {"query": query, "results": results}
        summary = "；".join(str(item.get("title") or "") for item in results[:3])
        return ToolResult(
            name=self.spec.name,
            arguments=arguments,
            success=bool(results),
            payload=payload,
            summary=summary,
            error=None if results else "未检索到搜索结果",
        )

