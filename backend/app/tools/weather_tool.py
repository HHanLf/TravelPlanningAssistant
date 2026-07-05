from __future__ import annotations

from typing import Any

import httpx


class WeatherTool:
    def __init__(self, api_key: str = "") -> None:
        self.api_key = api_key
        self.base_url = "https://apis.map.qq.com"

    def get_weather(self, city: str, day_offset: int = 0, date_label: str = "") -> dict[str, Any]:
        city = (city or "").strip()
        if not city:
            return {"city": "", "error": "city is required", "fallback": True}
        if not self.api_key:
            return {"city": city, "error": "weather api key is not configured", "fallback": True}

        resolved = self._resolve_weather_location(city)
        if not resolved:
            return {"city": city, "error": "failed to resolve city for weather query", "fallback": True}

        try:
            now_payload = self._request_weather(
                {"adcode": resolved["adcode"], "type": "now", "added_fields": "air"}
            )
            future_payload = self._request_weather(
                {"adcode": resolved["adcode"], "type": "future", "added_fields": "index,air"}
            )
        except (httpx.HTTPError, ValueError) as exc:
            return {"city": city, "error": f"weather api request failed: {exc}", "fallback": True}

        now_error = self._extract_api_error(now_payload)
        future_error = self._extract_api_error(future_payload)
        if now_error and future_error:
            return {
                "city": city,
                "resolved_city": resolved.get("city") or city,
                "error": future_error or now_error,
                "fallback": True,
                "raw": {"now": now_payload, "future": future_payload},
            }

        realtime_entry = self._extract_realtime_entry(now_payload)
        future_entries = self._extract_future_entries(future_payload)
        selected_offset = max(0, int(day_offset or 0))
        if future_entries:
            selected_offset = min(selected_offset, len(future_entries) - 1)
        future_entry = future_entries[selected_offset] if future_entries else {}
        day_info = (future_entry.get("day") or {}) if future_entry else {}
        night_info = (future_entry.get("night") or {}) if future_entry else {}
        air_info = self._extract_air_info(now_payload, future_payload)

        forecast = (
            day_info.get("weather")
            or realtime_entry.get("infos", {}).get("weather")
            or night_info.get("weather")
            or "天气待确认"
        )
        recommendation = self._build_recommendation(day_info, night_info, realtime_entry)

        result = {
            "provider": "tencent_map_weather",
            "city": city,
            "resolved_city": resolved.get("city") or city,
            "district": resolved.get("district"),
            "adcode": resolved.get("adcode"),
            "forecast": forecast,
            "date_label": date_label or self._build_default_date_label(selected_offset),
            "forecast_day_offset": selected_offset,
            "forecast_date": future_entry.get("date") or future_entry.get("forecast_date"),
            "temperature_min": self._safe_int(night_info.get("temperature") or day_info.get("temperature")),
            "temperature_max": self._safe_int(day_info.get("temperature")),
            "current_temperature": self._safe_int((realtime_entry.get("infos") or {}).get("temperature")),
            "wind_direction": (
                day_info.get("wind_direction")
                or (realtime_entry.get("infos") or {}).get("wind_direction")
                or night_info.get("wind_direction")
            ),
            "wind_scale": (
                day_info.get("wind_power")
                or (realtime_entry.get("infos") or {}).get("wind_power_v2")
                or (realtime_entry.get("infos") or {}).get("wind_power")
                or night_info.get("wind_power")
            ),
            "humidity": self._safe_int(
                day_info.get("humidity")
                or (realtime_entry.get("infos") or {}).get("humidity")
                or night_info.get("humidity")
            ),
            "air_quality": self._safe_int((air_info or {}).get("aqi")),
            "weather_indexes": self._extract_indexes(future_payload),
            "daily_forecasts": future_entries[:5],
            "indoor_bias": self._should_prefer_indoor(day_info, night_info, realtime_entry),
            "travel_mode_hint": self._travel_mode_hint(day_info, night_info, realtime_entry),
            "recommendation": recommendation,
            "update_time": future_entry.get("update_time") or realtime_entry.get("update_time"),
            "raw": {
                "realtime": realtime_entry,
                "selected_forecast": future_entry,
                "air": air_info,
                "future": future_payload,
            },
        }
        return result

    def _resolve_weather_location(self, city: str) -> dict[str, Any] | None:
        try:
            with httpx.Client(timeout=10.0) as client:
                response = client.get(
                    f"{self.base_url}/ws/geocoder/v1/",
                    params={"address": city, "key": self.api_key, "output": "json"},
                )
                response.raise_for_status()
                payload = response.json()
        except (httpx.HTTPError, ValueError):
            return None

        if payload.get("status") != 0:
            return None

        result = payload.get("result") or {}
        ad_info = result.get("ad_info") or {}
        adcode = ad_info.get("adcode")
        if adcode is None:
            return None

        return {
            "city": ad_info.get("city") or city,
            "district": ad_info.get("district"),
            "province": ad_info.get("province"),
            "adcode": str(adcode),
            "location": result.get("location") or {},
        }

    def _request_weather(self, params: dict[str, Any]) -> dict[str, Any]:
        with httpx.Client(timeout=15.0) as client:
            response = client.get(
                f"{self.base_url}/ws/weather/v1/",
                params={**params, "key": self.api_key, "output": "json"},
            )
            response.raise_for_status()
            return response.json()

    def _extract_api_error(self, payload: dict[str, Any]) -> str | None:
        if payload.get("status") == 0:
            return None
        return str(payload.get("message") or f"weather api error: {payload.get('status')}")

    def _extract_realtime_entry(self, payload: dict[str, Any]) -> dict[str, Any]:
        realtime = payload.get("result", {}).get("realtime") or []
        if isinstance(realtime, list) and realtime:
            first = realtime[0]
            if isinstance(first, dict):
                return first
        return {}

    def _extract_future_entries(self, payload: dict[str, Any]) -> list[dict[str, Any]]:
        forecast = payload.get("result", {}).get("forecast") or []
        if not isinstance(forecast, list) or not forecast:
            return []
        first = forecast[0]
        if not isinstance(first, dict):
            return []
        infos = first.get("infos") or []
        if not isinstance(infos, list):
            return []

        entries: list[dict[str, Any]] = []
        for item in infos:
            if not isinstance(item, dict):
                continue
            entries.append(
                {
                    "province": first.get("province"),
                    "city": first.get("city"),
                    "district": first.get("district"),
                    "adcode": first.get("adcode"),
                    "update_time": first.get("update_time"),
                    "forecast_date": item.get("date") or item.get("forecast_date"),
                    **item,
                }
            )
        return entries

    def _extract_air_info(self, now_payload: dict[str, Any], future_payload: dict[str, Any]) -> dict[str, Any]:
        realtime = self._extract_realtime_entry(now_payload)
        realtime_air = realtime.get("air")
        if isinstance(realtime_air, dict) and realtime_air:
            return realtime_air

        forecast = future_payload.get("result", {}).get("forecast") or []
        if isinstance(forecast, list) and forecast:
            first = forecast[0]
            if isinstance(first, dict):
                air_list = first.get("air") or []
                if isinstance(air_list, list) and air_list:
                    air = air_list[0]
                    if isinstance(air, dict):
                        return air
        return {}

    def _extract_indexes(self, payload: dict[str, Any]) -> list[dict[str, Any]]:
        forecast = payload.get("result", {}).get("forecast") or []
        if not isinstance(forecast, list) or not forecast:
            return []
        first = forecast[0]
        if not isinstance(first, dict):
            return []
        indexes = first.get("indexes") or []
        if not isinstance(indexes, list):
            return []

        simplified: list[dict[str, Any]] = []
        for index_group in indexes[:2]:
            if not isinstance(index_group, dict):
                continue
            ids = index_group.get("ids") or []
            if not isinstance(ids, list):
                continue
            for item in ids[:5]:
                if not isinstance(item, dict):
                    continue
                simplified.append(
                    {
                        "date": index_group.get("index_date"),
                        "name": item.get("name"),
                        "level": item.get("level"),
                        "desc": item.get("desc"),
                    }
                )
        return simplified

    def _build_recommendation(
        self,
        day_info: dict[str, Any],
        night_info: dict[str, Any],
        realtime_entry: dict[str, Any],
    ) -> str:
        weather_text = self._weather_text(day_info, night_info, realtime_entry)
        day_temp = self._safe_int(day_info.get("temperature"))
        night_temp = self._safe_int(night_info.get("temperature"))
        humidity = self._safe_int(day_info.get("humidity") or (realtime_entry.get("infos") or {}).get("humidity"))

        if self._should_prefer_indoor(day_info, night_info, realtime_entry):
            return "建议优先安排室内景点，并准备雨具或防滑鞋，出行预留更多通勤时间。"
        if day_temp is not None and day_temp >= 32:
            return "白天气温偏高，建议把户外核心行程安排在上午或傍晚，并注意补水防晒。"
        if night_temp is not None and night_temp <= 10:
            return "早晚偏凉，建议带上薄外套，夜间户外活动不宜安排过久。"
        if humidity is not None and humidity >= 85:
            return "空气湿度较高，体感可能偏闷，建议行程节奏放缓并优先选择通风休息点。"
        if any(keyword in weather_text for keyword in ["阴", "多云"]):
            return "天气相对平稳，适合把主要户外景点安排在白天，室内项目作为机动备选。"
        return "整体天气较适合出行，可优先安排核心户外景点在白天完成。"

    def _should_prefer_indoor(
        self,
        day_info: dict[str, Any],
        night_info: dict[str, Any],
        realtime_entry: dict[str, Any],
    ) -> bool:
        weather_text = self._weather_text(day_info, night_info, realtime_entry)
        return any(keyword in weather_text for keyword in ["雨", "雪", "雷", "暴"])

    def _travel_mode_hint(
        self,
        day_info: dict[str, Any],
        night_info: dict[str, Any],
        realtime_entry: dict[str, Any],
    ) -> str:
        if self._should_prefer_indoor(day_info, night_info, realtime_entry):
            return "prefer_indoor"
        day_temp = self._safe_int(day_info.get("temperature"))
        if day_temp is not None and day_temp >= 32:
            return "avoid_midday_outdoor"
        return "balanced"

    def _weather_text(
        self,
        day_info: dict[str, Any],
        night_info: dict[str, Any],
        realtime_entry: dict[str, Any],
    ) -> str:
        return "".join(
            filter(
                None,
                [
                    str(day_info.get("weather") or ""),
                    str(night_info.get("weather") or ""),
                    str((realtime_entry.get("infos") or {}).get("weather") or ""),
                ],
            )
        )

    def _build_default_date_label(self, day_offset: int) -> str:
        if day_offset <= 0:
            return "今天"
        if day_offset == 1:
            return "明天"
        if day_offset == 2:
            return "后天"
        return f"第{day_offset + 1}天"

    def _safe_int(self, value: Any) -> int | None:
        try:
            return int(value)
        except (TypeError, ValueError):
            return None
