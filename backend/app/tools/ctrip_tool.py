from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from backend.app.tools.amap_tool import TencentMapTool


@dataclass
class HotelOption:
    name: str
    price: int | None
    rating: float | None
    location: str
    area: str | None = None
    address: str | None = None
    distance_hint: str | None = None
    source: str = "tencent_map"


class CtripTool:
    def __init__(self, api_key: str = "", map_tool: TencentMapTool | None = None) -> None:
        self.api_key = api_key
        self.map_tool = map_tool or TencentMapTool(api_key)

    def search_hotels(self, city: str, budget: int | None = None) -> list[HotelOption]:
        city = (city or "").strip()
        raw_places = self.map_tool.search_place("酒店", city)
        hotels = [self._normalize_place(city, item, budget) for item in raw_places]
        hotels = [hotel for hotel in hotels if hotel]
        ranked = sorted(hotels, key=lambda hotel: self._score_hotel(city, hotel, budget), reverse=True)
        return ranked[:5]

    def _normalize_place(self, city: str, item: dict[str, Any], budget: int | None) -> HotelOption | None:
        name = str(item.get("name") or "").strip()
        if not name:
            return None
        address = str(item.get("address") or "").strip()
        area = self._infer_area(city, name, address)
        price = self._estimate_price(name, area)
        if isinstance(budget, int) and budget > 0 and price is not None:
            soft_limit = max(int(budget * 1.15), budget + 80)
            if price > soft_limit:
                return None
        rating = self._estimate_rating(name)
        distance_hint = self._build_distance_hint(city, area)
        return HotelOption(
            name=name,
            price=price,
            rating=rating,
            location=address or city,
            area=area,
            address=address or city,
            distance_hint=distance_hint,
        )

    def _infer_area(self, city: str, name: str, address: str) -> str:
        text = f"{name} {address}"
        suffixes = ["区", "路", "街", "商圈", "地铁站", "广场", "中心", "门", "里", "坊"]
        for suffix in suffixes:
            idx = address.find(suffix)
            if idx > 0:
                start = max(0, idx - 4)
                return address[start : idx + len(suffix)]
        for token in text.split():
            cleaned = token.strip("，。！？,.")
            if len(cleaned) >= 2 and any(keyword in cleaned for keyword in ["酒店", "宾馆", "连锁", "亚朵", "全季", "如家", "汉庭", "桔子", "美居"]):
                continue
            if any(marker in cleaned for marker in ["商圈", "地铁", "广场", "中心", "景区", "机场", "火车站"]):
                return cleaned
        return address or f"{city}主要活动区"

    def _estimate_price(self, name: str, area: str | None) -> int:
        text = f"{name} {area or ''}"
        if any(keyword in text for keyword in ["国宾", "丽思", "四季", "君悦", "文华东方", "豪华", "洲际"]):
            return 980
        if any(keyword in text for keyword in ["亚朵", "智选假日", "桔子", "美居", "逸扉", "诺富特"]):
            return 520
        if any(keyword in text for keyword in ["全季", "和颐", "喆啡", "希岸", "维也纳"]):
            return 420
        if any(keyword in text for keyword in ["汉庭", "如家", "7天", "格林豪泰", "速8", "快捷"]):
            return 300
        return 460

    def _estimate_rating(self, name: str) -> float:
        if any(keyword in name for keyword in ["国宾", "丽思", "君悦", "四季"]):
            return 4.9
        if any(keyword in name for keyword in ["亚朵", "全季", "智选假日", "桔子"]):
            return 4.7
        return 4.6

    def _build_distance_hint(self, city: str, area: str | None) -> str:
        if not area:
            return f"位于{city}主要活动区附近"
        return f"位于{area}附近，适合衔接核心行程"

    def _score_hotel(self, city: str, hotel: HotelOption, budget: int | None) -> float:
        score = 0.0
        if hotel.area:
            score += self._area_score(hotel.area, hotel.address or "", city)
        if hotel.rating is not None:
            score += hotel.rating
        if isinstance(budget, int) and budget > 0 and hotel.price is not None:
            if hotel.price <= budget:
                score += 2.0
            else:
                score -= (hotel.price - budget) / 200
        if any(keyword in hotel.name for keyword in ["亚朵", "全季", "智选假日", "美居", "国宾"]):
            score += 1.0
        return score

    def _area_score(self, area: str, address: str, city: str) -> float:
        text = f"{area} {address}"
        score = 0.0
        if any(keyword in text for keyword in ["地铁", "火车站", "高铁站", "机场"]):
            score += 2.5
        if any(keyword in text for keyword in ["广场", "商圈", "中心", "核心", "景区"]):
            score += 1.5
        if city and city in address:
            score += 0.5
        return score
