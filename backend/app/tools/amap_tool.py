from __future__ import annotations

import logging
from typing import Any

import httpx

from backend.app.tools.base import BaseTool

logger = logging.getLogger(__name__)


class TencentMapTool(BaseTool):
    name = "tencent_map"

    def __init__(self, api_key: str = "", secret_key: str = "") -> None:
        self.api_key = api_key
        self.secret_key = secret_key
        self.base_url = "https://apis.map.qq.com"

    def run(self, **kwargs: Any) -> dict[str, Any]:
        tool = kwargs.get("tool")
        if tool == "route_plan":
            return self.route_plan(kwargs.get("origin", ""), kwargs.get("destination", ""), mode=kwargs.get("mode", "driving"))
        if tool == "place_search":
            return self.search_place(kwargs.get("keyword", ""), kwargs.get("city", ""))
        return {"provider": "tencent_map", "params": kwargs, "result": "unsupported tool"}

    def route_plan(self, origin: str, destination: str, mode: str = "driving") -> dict[str, Any]:
        origin_location = self._resolve_location(origin)
        destination_location = self._resolve_location(destination)
        if not origin_location or not destination_location:
            logger.warning(
                "Tencent map location resolution failed",
                extra={
                    "origin": origin,
                    "destination": destination,
                    "origin_location": origin_location,
                    "destination_location": destination_location,
                },
            )
            return {
                "provider": "tencent_map",
                "origin": origin,
                "destination": destination,
                "error": "无法解析起点或终点坐标",
                "fallback": True,
            }

        route_mode = self._normalize_route_mode(mode)
        try:
            data = self._request_json(
                f"/ws/direction/v1/{route_mode}/",
                params={"from": origin_location, "to": destination_location, "policy": "LEAST_TIME", "get_mp": 1, "get_speed": 1},
            )
            if data.get("status") not in (None, 0):
                logger.error(
                    "Tencent map direction API returned error",
                    extra={
                        "origin": origin,
                        "destination": destination,
                        "route_mode": route_mode,
                        "origin_location": origin_location,
                        "destination_location": destination_location,
                        "response": data,
                    },
                )
                return {
                    "provider": "tencent_map",
                    "origin": origin,
                    "destination": destination,
                    "origin_location": origin_location,
                    "destination_location": destination_location,
                    "error": data.get("message") or "路线接口返回异常",
                    "fallback": True,
                    "raw": data,
                }

            routes = (data.get("result") or {}).get("routes") or []
            route = routes[0] if routes else {}
            distance_m = route.get("distance", 0)
            duration_min = route.get("duration", 0)
            if not route:
                logger.warning(
                    "Tencent map direction returned empty routes",
                    extra={
                        "origin": origin,
                        "destination": destination,
                        "route_mode": route_mode,
                        "origin_location": origin_location,
                        "destination_location": destination_location,
                        "response": data,
                    },
                )
                return {
                    "provider": "tencent_map",
                    "origin": origin,
                    "destination": destination,
                    "origin_location": origin_location,
                    "destination_location": destination_location,
                    "error": data.get("message") or "未返回路线结果",
                    "fallback": True,
                    "raw": data,
                }
            return {
                "provider": "tencent_map",
                "origin": origin,
                "destination": destination,
                "origin_location": origin_location,
                "destination_location": destination_location,
                "mode": route.get("mode") or route_mode.upper(),
                "distance_km": round(distance_m / 1000, 1) if distance_m else None,
                "duration_minutes": round(duration_min) if duration_min else None,
                "toll": route.get("toll"),
                "tags": route.get("tags", []),
                "raw": route,
            }
        except Exception as exc:  # noqa: BLE001
            logger.exception(
                "Tencent map route plan failed",
                extra={
                    "origin": origin,
                    "destination": destination,
                    "route_mode": route_mode,
                    "origin_location": origin_location,
                    "destination_location": destination_location,
                },
            )
            return {
                "provider": "tencent_map",
                "origin": origin,
                "destination": destination,
                "origin_location": origin_location,
                "destination_location": destination_location,
                "error": str(exc),
                "fallback": True,
            }

    def search_place(self, keyword: str, city: str) -> list[dict[str, Any]]:
        keyword = (keyword or "").strip()
        city = (city or "").strip()
        if not keyword or not city:
            return []
        try:
            data = self._request_json(
                "/ws/place/v1/search",
                params={"keyword": keyword, "boundary": f"region({city},1)", "page_size": 5, "get_subpois": 1, "added_fields": "category_code"},
            )
            if data.get("status") not in (None, 0):
                logger.error(
                    "Tencent map place search returned error",
                    extra={"keyword": keyword, "city": city, "response": data},
                )
                raise RuntimeError(str(data.get("message") or "地点搜索失败"))
            items = (data.get("data") or [])[:5]
            if items:
                return [
                    {
                        "name": item.get("title") or item.get("name") or keyword,
                        "address": item.get("address") or item.get("ad_info", {}).get("district", ""),
                        "location": item.get("location", {}),
                        "id": item.get("id"),
                        "category": item.get("category"),
                    }
                    for item in items
                ]
        except Exception as exc:  # noqa: BLE001
            logger.exception("Tencent map place search failed", extra={"keyword": keyword, "city": city, "error": str(exc)})

        return []

    def _resolve_location(self, location: str) -> str | None:
        if not location:
            return None
        geocoded = self._geocode(location)
        if geocoded:
            return geocoded

        candidate = self._search_location(location)
        if candidate:
            return candidate

        fallback = self._city_center_fallback(location)
        if fallback:
            logger.warning("Tencent map location fallback used", extra={"location": location, "fallback": fallback})
        return fallback

    def _geocode(self, location: str) -> str | None:
        if not location:
            return None
        try:
            data = self._request_json(
                "/ws/geocoder/v1/",
                params={"address": location, "get_poi": 1},
            )
            if data.get("status") not in (None, 0):
                logger.error("Tencent map geocode returned error", extra={"location": location, "response": data})
                return None
            result = data.get("result") or {}
            location_data = result.get("location") or {}
            lat = location_data.get("lat")
            lng = location_data.get("lng")
            if lat is None or lng is None:
                logger.warning("Tencent map geocode returned empty location", extra={"location": location, "response": data})
                return None
            return f"{lat},{lng}"
        except Exception as exc:  # noqa: BLE001
            logger.exception("Tencent map geocode failed", extra={"location": location, "error": str(exc)})
            return None

    def _search_location(self, location: str) -> str | None:
        try:
            data = self._request_json(
                "/ws/place/v1/search",
                params={"keyword": location, "boundary": f"region({location},1)", "page_size": 1, "get_subpois": 1},
            )
            if data.get("status") not in (None, 0):
                logger.error("Tencent map location search returned error", extra={"location": location, "response": data})
                return None
            items = data.get("data") or []
            if not items:
                return None
            loc = items[0].get("location") or {}
            lat = loc.get("lat")
            lng = loc.get("lng")
            if lat is None or lng is None:
                return None
            return f"{lat},{lng}"
        except Exception as exc:  # noqa: BLE001
            logger.exception("Tencent map location search failed", extra={"location": location, "error": str(exc)})
            return None

    def _city_center_fallback(self, location: str) -> str | None:
        centers = {
            "济南": "36.6512,117.1201",
            "杭州": "30.2741,120.1551",
            "上海": "31.2304,121.4737",
            "北京": "39.9042,116.4074",
            "广州": "23.1291,113.2644",
            "深圳": "22.5431,114.0579",
            "南京": "32.0603,118.7969",
            "西安": "34.3416,108.9398",
            "重庆": "29.5630,106.5516",
            "苏州": "31.2989,120.5853",
            "青岛": "36.0671,120.3826",
            "厦门": "24.4798,118.0894",
            "武汉": "30.5928,114.3055",
            "天津": "39.0842,117.2009",
            "成都": "30.5728,104.0668",
        }
        for city, coords in centers.items():
            if city in location:
                return coords
        return None

    def _normalize_route_mode(self, mode: str) -> str:
        normalized = (mode or "driving").strip().lower()
        if normalized in {"walking", "bicycling", "ebicycling", "transit", "driving"}:
            return normalized
        return "driving"

    def _request_json(self, path: str, params: dict[str, Any]) -> dict[str, Any]:
        query = {k: v for k, v in params.items() if v is not None}
        query["key"] = self.api_key
        query.setdefault("output", "json")
        response = httpx.get(f"{self.base_url}{path}", params=query, timeout=15.0)
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError:
            logger.error(
                "Tencent map HTTP error",
                extra={"path": path, "status_code": response.status_code, "response_text": response.text, "params": {k: v for k, v in query.items() if k != 'key'}},
            )
            raise

        data = response.json()
        if isinstance(data, dict) and data.get("status") not in (None, 0):
            logger.error(
                "Tencent map API returned error",
                extra={"path": path, "params": {k: v for k, v in query.items() if k != 'key'}, "response": data},
            )
        return data
