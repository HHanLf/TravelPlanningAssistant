from __future__ import annotations

import re
from typing import Any

from app.agent.state import PlanningProblem
from app.agent.travel_semantics import infer_poi_theme


class ProblemAnalyzer:
    CITY_NAMES = (
        "北京",
        "上海",
        "广州",
        "深圳",
        "成都",
        "杭州",
        "重庆",
        "西安",
        "南京",
        "苏州",
        "厦门",
        "长沙",
        "武汉",
        "三亚",
        "青岛",
        "济南",
        "天津",
        "大理",
        "丽江",
        "昆明",
        "桂林",
        "拉萨",
        "哈尔滨",
        "香港",
        "澳门",
        "东京",
        "大阪",
        "首尔",
        "曼谷",
        "新加坡",
    )
    PREFERENCES = (
        "自然",
        "风景",
        "美食",
        "亲子",
        "历史建筑",
        "古建筑",
        "人文",
        "历史",
        "购物",
        "摄影",
        "夜景",
        "博物馆",
        "徒步",
        "海边",
        "温泉",
        "小众",
        "避坑",
        "慢节奏",
        "性价比",
        "情侣",
        "老人",
    )
    LOCAL_LOOKUP_KEYWORDS = (
        "餐厅",
        "饭店",
        "美食",
        "小吃",
        "咖啡",
        "酒店",
        "住宿",
        "民宿",
        "景点",
        "天气",
    )
    TRIP_CONTEXT_KEYWORDS = ("玩", "旅游", "旅行", "行程", "攻略", "规划", "自由行", "几日游")
    REGION_DEFAULT_DESTINATIONS: dict[str, tuple[str, str]] = {
        "内蒙古": ("呼和浩特", "如果想重点看呼伦贝尔草原、阿尔山或额济纳旗，可以再把目的地切到对应城市。"),
        "新疆": ("乌鲁木齐", "如果想重点看伊犁、喀纳斯或喀什，可以再把目的地切到对应城市。"),
        "西藏": ("拉萨", "如果想重点看林芝、日喀则或阿里，可以再把目的地切到对应城市。"),
        "青海": ("西宁", "如果想重点看青海湖、祁连或可可西里，可以再把目的地切到对应城市。"),
        "甘肃": ("兰州", "如果想重点看敦煌、张掖或甘南，可以再把目的地切到对应城市。"),
        "宁夏": ("银川", "如果想重点看中卫沙坡头或贺兰山，可以再把目的地切到对应城市。"),
        "广西": ("桂林", "如果想重点看北海、南宁或崇左，可以再把目的地切到对应城市。"),
        "云南": ("昆明", "如果想重点看大理、丽江、香格里拉或西双版纳，可以再把目的地切到对应城市。"),
        "贵州": ("贵阳", "如果想重点看黄果树、荔波或黔东南，可以再把目的地切到对应城市。"),
        "四川": ("成都", "如果想重点看九寨沟、川西或峨眉山，可以再把目的地切到对应城市。"),
        "福建": ("厦门", "如果想重点看福州、武夷山或泉州，可以再把目的地切到对应城市。"),
        "海南": ("三亚", "如果想重点看海口、万宁或陵水，可以再把目的地切到对应城市。"),
    }

    def analyze(self, message: str, memory_context: dict[str, Any] | None = None) -> PlanningProblem:
        memory_context = memory_context or {}
        profile = memory_context.get("user_profile") or {}
        text = message or ""

        explicit_origin = self._extract_origin(text)
        explicit_destination = self._extract_destination(text, explicit_origin)
        explicit_days = self._extract_days(text)
        explicit_budget = self._extract_budget(text)
        explicit_group_size = self._extract_group_size(text)
        explicit_date_range = self._extract_date_range(text)
        context_scope = self._context_scope(text, explicit_destination)
        previous_destination = profile.get("destination")
        is_new_destination = bool(
            explicit_destination and previous_destination and explicit_destination != previous_destination
        )

        can_inherit_trip_slots = context_scope != "local_lookup" and not is_new_destination
        origin = explicit_origin or (profile.get("departure") or profile.get("origin") if can_inherit_trip_slots else None)
        destination = explicit_destination or (profile.get("destination") if context_scope != "local_lookup" else None)
        days = explicit_days or (profile.get("days") if can_inherit_trip_slots else None)
        budget = explicit_budget or (profile.get("budget") if can_inherit_trip_slots else None)
        group_size = explicit_group_size or (
            profile.get("companions") or profile.get("group_size") if can_inherit_trip_slots else None
        )
        existing_preferences = profile.get("preferences") or [] if can_inherit_trip_slots else []
        preferences = self._merge_preferences(existing_preferences, text)
        date_range = explicit_date_range or (profile.get("date_range") if can_inherit_trip_slots else None)
        existing_constraints = profile.get("constraints") or {} if can_inherit_trip_slots else {}
        constraints = self._merge_constraints(existing_constraints, text)
        region_resolution = self._region_resolution(destination)
        if region_resolution:
            constraints = {
                **constraints,
                "requested_region": destination,
                "tool_destination": region_resolution["tool_destination"],
                "region_resolution_note": region_resolution["note"],
            }
        explicit_fields = self._explicit_fields(
            origin=explicit_origin,
            destination=explicit_destination,
            days=explicit_days,
            budget=explicit_budget,
            group_size=explicit_group_size,
            date_range=explicit_date_range,
            preferences=preferences,
            text=text,
        )

        missing_info = self._missing_info(
            message=text,
            destination=destination,
            days=days,
            budget=budget,
            group_size=group_size,
        )
        assumptions = [] if context_scope == "local_lookup" else self._assumptions(destination, days, budget, group_size)
        if region_resolution and context_scope != "local_lookup":
            assumptions.append(
                f"{destination}范围很大，先按{region_resolution['tool_destination']}及周边做可执行初版；"
                f"{region_resolution['note']}"
            )
        return PlanningProblem(
            origin=origin,
            destination=destination,
            days=self._safe_int(days),
            budget=self._safe_int(budget),
            group_size=self._safe_int(group_size),
            date_range=date_range,
            preferences=preferences,
            constraints=constraints,
            missing_info=missing_info,
            assumptions=assumptions,
            explicit_fields=explicit_fields,
            context_scope=context_scope,
        )

    def profile_updates(self, problem: PlanningProblem) -> dict[str, Any]:
        return {
            "_context_scope": problem.context_scope,
            "_explicit_fields": problem.explicit_fields,
            "departure": problem.origin,
            "destination": problem.destination,
            "days": problem.days,
            "budget": problem.budget,
            "companions": problem.group_size,
            "preferences": problem.preferences or None,
            "date_range": problem.date_range,
            "constraints": problem.constraints or None,
        }

    def _extract_origin(self, text: str) -> str | None:
        patterns = (
            r"从\s*([\u4e00-\u9fa5A-Za-z]{2,12}?)\s*(?:出发|到|去)",
            r"([\u4e00-\u9fa5A-Za-z]{2,12}?)\s*出发",
        )
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return self._clean_city(match.group(1))
        route_cities = self._ordered_city_mentions(text)
        if len(route_cities) >= 2 and any(token in text for token in ("到", "去", "出发")):
            return route_cities[0]
        return None

    def _extract_destination(self, text: str, origin: str | None = None) -> str | None:
        patterns = (
            r"(?:给我|帮我|请|推荐|找|搜|看看|查)?\s*([\u4e00-\u9fa5A-Za-z]{2,20}?)\s*的(?:餐厅|饭店|美食|小吃|咖啡|酒店|住宿|民宿|景点|天气)",
            r"(?:给我|帮我|请|推荐|找|搜|看看|查)\s*([\u4e00-\u9fa5A-Za-z]{2,20}?)(?:餐厅|饭店|美食|小吃|咖啡|酒店|住宿|民宿|景点|天气)",
            r"(?:去|到|玩|游)\s*([\u4e00-\u9fa5A-Za-z]{2,12})",
            r"([\u4e00-\u9fa5A-Za-z]{2,12})\s*(?:旅游|旅行|攻略|行程|路线|自由行|几日游)",
        )
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                city = self._clean_city(match.group(1))
                if city and city != origin:
                    return city

        found = [city for city in self._known_destinations() if city in text]
        ordered = self._ordered_city_mentions(text)
        if origin and origin in ordered and len(ordered) > 1:
            ordered = [city for city in ordered if city != origin]
        if ordered:
            return ordered[-1]
        if origin and origin in found and len(found) > 1:
            found = [city for city in found if city != origin]
        return found[-1] if found else None

    @classmethod
    def _context_scope(cls, text: str, explicit_destination: str | None) -> str:
        if not explicit_destination:
            return "trip"
        has_local_lookup = any(keyword in text for keyword in cls.LOCAL_LOOKUP_KEYWORDS)
        has_trip_context = any(keyword in text for keyword in cls.TRIP_CONTEXT_KEYWORDS)
        return "local_lookup" if has_local_lookup and not has_trip_context else "trip"

    @staticmethod
    def _explicit_fields(
        *,
        origin: str | None,
        destination: str | None,
        days: int | None,
        budget: int | None,
        group_size: int | None,
        date_range: dict[str, str] | None,
        preferences: list[str],
        text: str,
    ) -> list[str]:
        fields: list[str] = []
        if origin:
            fields.append("origin")
        if destination:
            fields.append("destination")
        if days:
            fields.append("days")
        if budget:
            fields.append("budget")
        if group_size:
            fields.append("group_size")
        if date_range:
            fields.append("date_range")
        if any(preference in text for preference in preferences):
            fields.append("preferences")
        return fields

    @staticmethod
    def _extract_days(text: str) -> int | None:
        patterns = (r"(\d+)\s*(?:天|日|晚)", r"([一二两三四五六七八九十])\s*(?:天|日|晚)")
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return ProblemAnalyzer._chinese_num(match.group(1))
        return None

    @staticmethod
    def _extract_budget(text: str) -> int | None:
        patterns = (
            r"(?:预算|人均|总共|花费|控制在)\s*(\d+(?:\.\d+)?)\s*(万|千|k|K|元)?",
            r"(\d+(?:\.\d+)?)\s*(万|千|k|K|元)\s*(?:预算|以内|左右)?",
        )
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                amount = float(match.group(1))
                unit = match.group(2) or ""
                if unit == "万":
                    amount *= 10000
                elif unit in {"千", "k", "K"}:
                    amount *= 1000
                return int(amount)
        return None

    @staticmethod
    def _extract_group_size(text: str) -> int | None:
        patterns = (
            r"(\d+)\s*(?:个人|人|位)",
            r"([一二两三四五六七八九十])\s*(?:个人|人|位)",
        )
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                value = ProblemAnalyzer._chinese_num(match.group(1))
                if value and value <= 30:
                    return value
        if "情侣" in text or "两个人" in text:
            return 2
        if "亲子" in text or "一家" in text:
            return 3
        return None

    @staticmethod
    def _extract_date_range(text: str) -> dict[str, str] | None:
        match = re.search(r"(\d{1,2})\s*月\s*(\d{1,2})\s*(?:日|号)?", text)
        if match:
            return {"start": f"{int(match.group(1)):02d}-{int(match.group(2)):02d}"}
        for label in ("今天", "明天", "后天", "周末", "春节", "五一", "国庆", "暑假"):
            if label in text:
                return {"label": label}
        return None

    def _merge_preferences(self, existing: list[Any], text: str) -> list[str]:
        merged = [str(item) for item in existing if item]
        for keyword in self.PREFERENCES:
            if keyword in text and keyword not in merged:
                merged.append(keyword)
        return merged[:12]

    @staticmethod
    def _merge_constraints(existing: dict[str, Any], text: str) -> dict[str, Any]:
        constraints = dict(existing or {})
        travel_theme = infer_poi_theme(text=text, constraints={})
        if travel_theme:
            constraints["travel_theme"] = travel_theme
        if any(keyword in text for keyword in ("公共交通", "地铁", "公交", "少打车")):
            constraints["transport_mode"] = "public"
        if any(keyword in text for keyword in ("自驾", "开车")):
            constraints["transport_mode"] = "driving"
        if any(keyword in text for keyword in ("亲子", "带娃", "孩子", "儿童")):
            constraints["family_friendly"] = True
        if any(keyword in text for keyword in ("老人", "父母", "少走路", "步行少")):
            constraints["low_walking"] = True
        if any(keyword in text for keyword in ("慢节奏", "轻松", "不赶")):
            constraints["pace"] = "relaxed"
        if any(keyword in text for keyword in ("特种兵", "紧凑", "多安排")):
            constraints["pace"] = "intensive"
        if any(keyword in text for keyword in ("靠近地铁", "地铁附近", "离地铁近")):
            constraints["hotel_near_metro"] = True
        if any(keyword in text for keyword in ("避开收费", "免费景点", "少花门票")):
            constraints["avoid_paid_attractions"] = True
        if any(keyword in text for keyword in ("清真", "素食", "不吃辣", "少辣")):
            constraints["dietary_preference"] = "、".join(
                keyword for keyword in ("清真", "素食", "不吃辣", "少辣") if keyword in text
            )
        return constraints

    @classmethod
    def _region_resolution(cls, destination: str | None) -> dict[str, str] | None:
        if not destination:
            return None
        normalized = destination.strip()
        if normalized.endswith(("省", "自治区", "特别行政区")):
            normalized = re.sub(r"(省|自治区|特别行政区)$", "", normalized)
        if normalized not in cls.REGION_DEFAULT_DESTINATIONS:
            return None
        tool_destination, note = cls.REGION_DEFAULT_DESTINATIONS[normalized]
        if tool_destination == destination:
            return None
        return {"tool_destination": tool_destination, "note": note}

    @staticmethod
    def _missing_info(
        message: str,
        destination: str | None,
        days: int | None,
        budget: int | None,
        group_size: int | None,
    ) -> list[str]:
        missing: list[str] = []
        if not destination:
            missing.append("destination")
        if any(keyword in message for keyword in ("规划", "攻略", "行程", "旅游", "旅行")):
            if not days:
                missing.append("days")
            if not budget:
                missing.append("budget")
            if not group_size:
                missing.append("group_size")
        return missing

    @staticmethod
    def _assumptions(
        destination: str | None,
        days: int | None,
        budget: int | None,
        group_size: int | None,
    ) -> list[str]:
        assumptions: list[str] = []
        if destination and not days:
            assumptions.append("暂按 3 天 2 晚做初版节奏。")
        if destination and not budget:
            assumptions.append("预算未说明时，默认按中等舒适度控制。")
        if destination and not group_size:
            assumptions.append("人数未说明时，默认按 2 人出行估算。")
        return assumptions

    @staticmethod
    def _clean_city(value: str | None) -> str | None:
        if not value:
            return None
        cleaned = re.sub(r"[，。！？,.!?；;\s]", "", value)
        cleaned = re.sub(r"(市|省|自治区|特别行政区)$", "", cleaned)
        stop_words = ("我想", "我们", "帮我", "给我", "请帮", "请", "推荐", "找", "搜", "看看", "查", "计划", "安排", "出发")
        for word in stop_words:
            cleaned = cleaned.replace(word, "")
        cleaned = re.split(r"(?:玩|旅游|旅行|攻略|行程|自由行|几日游)", cleaned, maxsplit=1)[0]
        if "的" in cleaned:
            tail = cleaned.rsplit("的", 1)[1]
            if 2 <= len(tail) <= 12:
                cleaned = tail
        cleaned = re.sub(r"(的|地|之旅|路线|安排|规划)$", "", cleaned)
        return cleaned or None

    @classmethod
    def _ordered_city_mentions(cls, text: str) -> list[str]:
        matches: list[tuple[int, str]] = []
        for city in cls._known_destinations():
            index = text.find(city)
            if index >= 0:
                matches.append((index, city))
        matches.sort(key=lambda item: item[0])
        ordered: list[str] = []
        for _, city in matches:
            if city not in ordered:
                ordered.append(city)
        return ordered

    @classmethod
    def _known_destinations(cls) -> tuple[str, ...]:
        region_names = tuple(cls.REGION_DEFAULT_DESTINATIONS.keys())
        return (*cls.CITY_NAMES, *region_names)

    @staticmethod
    def _safe_int(value: Any) -> int | None:
        try:
            if value in (None, ""):
                return None
            return int(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _chinese_num(value: str) -> int | None:
        if value.isdigit():
            return int(value)
        mapping = {"一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9, "十": 10}
        if value == "十":
            return 10
        if value in mapping:
            return mapping[value]
        if "十" in value:
            left, _, right = value.partition("十")
            return mapping.get(left, 1) * 10 + mapping.get(right, 0)
        return None
