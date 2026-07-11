from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class PoiThemeProfile:
    name: str
    label: str
    triggers: tuple[str, ...]
    query_keywords: tuple[str, ...]
    positive_name_keywords: tuple[str, ...]
    positive_category_keywords: tuple[str, ...]
    negative_keywords: tuple[str, ...]
    negative_category_keywords: tuple[str, ...] = ()


COMMON_COMMERCIAL_NEGATIVES = (
    "公司",
    "商店",
    "购物",
    "商场",
    "超市",
    "便利店",
    "培训",
    "教育",
    "学校",
    "中学",
    "小学",
    "幼儿园",
    "办公",
    "写字楼",
    "工作室",
    "酒店",
    "宾馆",
    "公寓",
    "诊所",
    "医院",
    "美容",
    "美甲",
    "助听器",
    "照相馆",
    "摄影",
    "写真",
    "影楼",
)


THEME_PROFILES: dict[str, PoiThemeProfile] = {
    "natural_scenery": PoiThemeProfile(
        name="natural_scenery",
        label="自然风景",
        triggers=("自然风景", "自然景色", "自然", "山水", "湿地", "森林", "公园", "风景", "徒步", "海边"),
        query_keywords=("风景名胜", "自然景区", "湿地公园", "森林公园", "草原景区", "景点", "公园"),
        positive_name_keywords=("风景", "景区", "公园", "湿地", "森林", "草原", "山", "湖", "溪", "江", "河", "岛", "海", "湾", "峰", "岭", "谷", "瀑", "古道", "绿道", "自然保护区", "植物园"),
        positive_category_keywords=("旅游景点", "风景名胜", "公园", "自然地物"),
        negative_keywords=(*COMMON_COMMERCIAL_NEGATIVES, "领域"),
        negative_category_keywords=("购物服务", "生活服务", "医疗保健", "公司企业"),
    ),
    "historical_architecture": PoiThemeProfile(
        name="historical_architecture",
        label="历史建筑",
        triggers=("历史建筑", "古建筑", "历史遗迹", "名胜古迹", "古迹", "古城", "古镇", "人文建筑", "历史文化", "历史", "建筑"),
        query_keywords=("历史建筑", "名胜古迹", "古建筑", "历史文化景点", "文物古迹", "古城墙", "故宫"),
        positive_name_keywords=("故宫", "天坛", "颐和园", "圆明园", "长城", "城墙", "古城", "古镇", "古街", "古巷", "胡同", "牌坊", "祠", "庙", "寺", "观", "宫", "楼", "塔", "桥", "门", "遗址", "旧址", "会馆", "王府", "园林", "文物", "名胜", "古建筑"),
        positive_category_keywords=("旅游景点", "风景名胜", "文物古迹", "历史遗迹", "纪念馆", "博物馆"),
        negative_keywords=(*COMMON_COMMERCIAL_NEGATIVES, "历史学系", "档案馆", "出版社", "研究院", "研究所", "协会", "图书馆", "文化馆"),
        negative_category_keywords=("科教文化服务;学校", "科教文化服务;科研机构", "公司企业", "政府机构"),
    ),
    "museum_culture": PoiThemeProfile(
        name="museum_culture",
        label="博物馆人文",
        triggers=("博物馆", "展览", "美术馆", "纪念馆", "文化馆", "人文"),
        query_keywords=("博物馆", "纪念馆", "美术馆", "展览馆", "文化景点"),
        positive_name_keywords=("博物馆", "纪念馆", "美术馆", "展览馆", "陈列馆", "艺术馆"),
        positive_category_keywords=("博物馆", "纪念馆", "美术馆", "文化场馆", "旅游景点"),
        negative_keywords=("公司", "培训", "商店", "购物", "学校", "出版社"),
        negative_category_keywords=("公司企业", "购物服务"),
    ),
    "night_view": PoiThemeProfile(
        name="night_view",
        label="夜景",
        triggers=("夜景", "夜游", "灯光", "看夜景"),
        query_keywords=("夜景", "观景台", "夜游", "城市地标", "步行街"),
        positive_name_keywords=("夜景", "观景", "观景台", "塔", "广场", "步行街", "江", "河", "湖", "地标"),
        positive_category_keywords=("旅游景点", "风景名胜", "休闲娱乐"),
        negative_keywords=("公司", "酒店", "公寓", "商店", "培训"),
        negative_category_keywords=("公司企业"),
    ),
    "family_friendly": PoiThemeProfile(
        name="family_friendly",
        label="亲子",
        triggers=("亲子", "带娃", "孩子", "儿童"),
        query_keywords=("亲子景点", "儿童乐园", "动物园", "科技馆", "海洋馆", "公园"),
        positive_name_keywords=("儿童", "亲子", "乐园", "动物园", "科技馆", "海洋馆", "公园", "植物园"),
        positive_category_keywords=("旅游景点", "公园", "科教文化服务", "休闲娱乐"),
        negative_keywords=("培训", "早教", "摄影", "商店", "公司"),
        negative_category_keywords=("公司企业"),
    ),
}


def infer_poi_theme(text: str = "", preferences: list[str] | None = None, constraints: dict[str, Any] | None = None) -> str:
    constraints = constraints or {}
    existing = str(constraints.get("travel_theme") or "")
    if existing in THEME_PROFILES:
        return existing

    haystack = " ".join([text or "", " ".join(preferences or [])])
    matches: list[tuple[int, str]] = []
    for name, profile in THEME_PROFILES.items():
        score = 0
        for trigger in profile.triggers:
            if trigger and trigger in haystack:
                score += len(trigger)
        if score:
            matches.append((score, name))
    if not matches:
        return ""
    matches.sort(reverse=True)
    return matches[0][1]


def get_poi_theme_profile(theme: str) -> PoiThemeProfile | None:
    return THEME_PROFILES.get(theme)


def score_poi_for_theme(place: dict[str, Any], theme: str) -> tuple[int, str]:
    profile = get_poi_theme_profile(theme)
    if profile is None:
        return 1, "no_theme"

    name = str(place.get("name") or "")
    address = str(place.get("address") or "")
    category = str(place.get("category") or "")
    matched_keyword = str(place.get("matched_keyword") or "")
    text = f"{name} {address} {category}"

    if any(keyword in text for keyword in profile.negative_keywords):
        return -100, "negative_keyword"
    if any(keyword and keyword in category for keyword in profile.negative_category_keywords):
        return -80, "negative_category"

    score = 0
    if any(keyword in category for keyword in profile.positive_category_keywords):
        score += 45
    if any(keyword in name for keyword in profile.positive_name_keywords):
        score += 35
    if any(keyword in address for keyword in profile.positive_name_keywords):
        score += 10
    if matched_keyword in profile.query_keywords:
        score += 8

    if score <= 0:
        return 0, "not_relevant_enough"
    return score, "matched_theme"


def is_polluted_poi(place: dict[str, Any], theme: str) -> bool:
    score, _ = score_poi_for_theme(place, theme)
    return score <= 0
