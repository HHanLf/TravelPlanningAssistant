from __future__ import annotations

import re
from typing import Any
from urllib.parse import urlencode

import httpx

from backend.app.tools.base import BaseTool


class XiaohongshuTool(BaseTool):
    name = "xiaohongshu_search"

    def __init__(self, token: str = "", base_url: str = "https://api.justoneapi.com") -> None:
        self.token = token.strip()
        self.base_url = base_url.rstrip("/")
        self.marketing_keywords = [
            "点击链接",
            "私信",
            "置顶",
            "合作",
            "推广",
            "广告",
            "团购",
            "返现",
            "优惠券",
            "加vx",
            "加v",
            "咨询客服",
            "代订",
            "带价",
            "拼单",
            "引流",
            "速来",
            "冲",
            "全网最低",
            "保姆级",
            "无广",
        ]
        self.travel_keywords = [
            "攻略",
            "景点",
            "路线",
            "行程",
            "住宿",
            "酒店",
            "民宿",
            "交通",
            "地铁",
            "高铁",
            "机场",
            "打车",
            "步行",
            "排队",
            "预约",
            "门票",
            "避坑",
            "美食",
            "小吃",
            "餐厅",
            "拍照",
            "机位",
            "日落",
            "夜景",
            "亲子",
            "情侣",
            "周末",
            "一日游",
            "两日游",
            "三日游",
        ]

    def available(self) -> bool:
        return bool(self.token)

    def search_notes(
        self,
        keyword: str,
        page: int = 1,
        sort_type: str = "general",
        note_type: str = "_0",
        time_filter: str = "",
        limit: int = 5,
    ) -> dict[str, Any]:
        return self.run(
            keyword=keyword,
            page=page,
            sort_type=sort_type,
            note_type=note_type,
            time_filter=time_filter,
            limit=limit,
        )

    def run(self, **kwargs: Any) -> dict[str, Any]:
        keyword = str(kwargs.get("keyword") or "").strip()
        if not keyword:
            return {"keyword": "", "notes": [], "summary": ""}
        if not self.available():
            return {
                "keyword": keyword,
                "notes": [],
                "summary": "",
                "error": "xiaohongshu token is not configured",
            }

        page = max(1, int(kwargs.get("page") or 1))
        sort_type = self._normalize_sort(str(kwargs.get("sort_type") or kwargs.get("sort") or "general"))
        note_type = self._normalize_note_type(str(kwargs.get("note_type") or kwargs.get("noteType") or "_0"))
        time_filter = self._normalize_note_time(str(kwargs.get("time_filter") or kwargs.get("noteTime") or ""))
        limit = max(1, min(int(kwargs.get("limit") or 5), 10))

        attempts = self._build_request_attempts(keyword, page, sort_type, note_type, time_filter)
        payload: dict[str, Any] | None = None
        last_error: dict[str, Any] | None = None

        try:
            with httpx.Client(timeout=60.0) as client:
                for attempt in attempts:
                    url = f"{self.base_url}/api/xiaohongshu/search-note/v2?{urlencode(attempt)}"
                    try:
                        response = client.get(url)
                        response.raise_for_status()
                        payload = response.json()
                    except (httpx.TimeoutException, httpx.HTTPError, ValueError) as exc:
                        last_error = {
                            "keyword": keyword,
                            "notes": [],
                            "summary": "",
                            "error": f"xiaohongshu api request failed: {exc}",
                        }
                        continue

                    code = payload.get("code")
                    if code == 0:
                        break
                    last_error = {
                        "keyword": keyword,
                        "notes": [],
                        "summary": "",
                        "error": payload.get("message") or payload.get("msg") or f"xiaohongshu api error: {code}",
                        "code": code,
                        "raw": payload,
                    }
                    if code not in {301, 302}:
                        break
        except Exception as exc:
            return {
                "keyword": keyword,
                "notes": [],
                "summary": "",
                "error": f"xiaohongshu api unexpected failure: {exc}",
            }

        if not isinstance(payload, dict) or payload.get("code") != 0:
            return last_error or {
                "keyword": keyword,
                "notes": [],
                "summary": "",
                "error": "xiaohongshu api request failed",
            }

        notes = self._extract_notes(payload)
        normalized_notes = [self._normalize_note(item, keyword) for item in notes]
        cleaned_notes = self._select_notes(normalized_notes, limit)

        return {
            "keyword": keyword,
            "notes": cleaned_notes,
            "summary": self._build_summary(keyword, cleaned_notes),
            "insights": self._build_insights(keyword, cleaned_notes),
            "raw_count": len(notes),
            "filtered_count": len(cleaned_notes),
            "api_info": self._extract_api_info(payload),
        }

    def _extract_notes(self, payload: dict[str, Any]) -> list[dict[str, Any]]:
        data = payload.get("data")
        if isinstance(data, list):
            return [item for item in data if isinstance(item, dict)]
        if not isinstance(data, dict):
            return []

        candidate_keys = ["items", "notes", "note_list", "list", "data", "result", "results"]
        for key in candidate_keys:
            value = data.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
            if isinstance(value, dict):
                nested_items = value.get("items") or value.get("list") or value.get("notes")
                if isinstance(nested_items, list):
                    return [item for item in nested_items if isinstance(item, dict)]
        return []

    def _normalize_sort(self, sort_value: str) -> str:
        allowed = {
            "general",
            "popularity_descending",
            "time_descending",
            "comment_descending",
            "collect_descending",
        }
        normalized = sort_value.strip() or "general"
        return normalized if normalized in allowed else "general"

    def _normalize_note_type(self, note_type: str) -> str:
        mapping = {
            "ALL": "_0",
            "VIDEO": "_1",
            "NORMAL": "_2",
            "_0": "_0",
            "_1": "_1",
            "_2": "_2",
        }
        normalized = note_type.strip().upper() if note_type else "_0"
        return mapping.get(normalized, "_0")

    def _normalize_note_time(self, note_time: str) -> str:
        allowed = {"", "ONE_DAY", "ONE_WEEK", "HALF_YEAR"}
        normalized = note_time.strip().upper()
        return normalized if normalized in allowed else ""

    def _build_request_attempts(
        self,
        keyword: str,
        page: int,
        sort_type: str,
        note_type: str,
        time_filter: str,
    ) -> list[dict[str, Any]]:
        attempts: list[dict[str, Any]] = []
        keyword_candidates = [keyword]
        simplified_keyword = self._simplify_keyword(keyword)
        if simplified_keyword and simplified_keyword not in keyword_candidates:
            keyword_candidates.append(simplified_keyword)

        for candidate in keyword_candidates[:2]:
            params: dict[str, Any] = {
                "token": self.token,
                "keyword": candidate,
                "page": page,
                "sort": sort_type,
                "noteType": note_type,
            }
            if time_filter:
                params["noteTime"] = time_filter
            attempts.append(params)
        return attempts

    def _simplify_keyword(self, keyword: str) -> str:
        text = re.sub(r"预算\s*\d+\s*元", "", keyword)
        text = re.sub(r"\d+\s*日游", "", text)
        text = re.sub(r"旅游攻略|攻略", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def _extract_api_info(self, payload: dict[str, Any]) -> dict[str, Any]:
        data = payload.get("data")
        if not isinstance(data, dict):
            return {}
        api_info = data.get("api_info")
        if isinstance(api_info, dict):
            return api_info
        return {}

    def _normalize_note(self, item: dict[str, Any], keyword: str) -> dict[str, Any]:
        title = self._clean_text(self._pick_first_str(item, ["title", "note_title", "display_title"]))
        desc = self._clean_text(self._pick_first_str(item, ["desc", "description", "content", "summary", "note_desc"]))
        author = self._pick_nested_str(
            item,
            [("author", "name"), ("user", "nickname"), ("user", "name"), ("author", "nickname")],
        ) or self._pick_first_str(item, ["nickname", "user_name", "author_name"])
        note_id = self._pick_first_str(item, ["note_id", "id", "noteId"])
        liked_count = self._pick_first(item, ["liked_count", "like_count", "likes", "digg_count"])
        collect_count = self._pick_first(item, ["collected_count", "collect_count", "favorites"])
        comment_count = self._pick_first(item, ["comment_count", "comments_count", "comments"])
        publish_time = self._pick_first_str(item, ["publish_time", "time", "create_time"])
        note_type = self._pick_first_str(item, ["type", "note_type"])
        url = self._pick_first_str(item, ["url", "note_url", "share_url"])
        cover = self._pick_nested_str(item, [("cover", "url"), ("image", "url")]) or self._pick_first_str(
            item, ["cover_url", "image_url"]
        )

        combined_text = f"{title} {desc}".strip()
        travel_tags = self._extract_travel_tags(combined_text)
        marketing_score = self._marketing_score(combined_text)
        relevance_score = self._relevance_score(keyword, combined_text, travel_tags)
        cleaned_summary = self._summarize_description(desc or title)
        engagement_score = self._engagement_score(liked_count, collect_count, comment_count)

        return {
            "title": title,
            "summary": cleaned_summary,
            "author": author,
            "note_id": note_id,
            "url": url,
            "cover": cover,
            "publish_time": publish_time,
            "note_type": note_type,
            "liked_count": liked_count,
            "collect_count": collect_count,
            "comment_count": comment_count,
            "travel_tags": travel_tags,
            "is_marketing": marketing_score >= 3,
            "is_relevant": relevance_score >= 1,
            "marketing_score": marketing_score,
            "relevance_score": relevance_score,
            "engagement_score": engagement_score,
        }

    def _select_notes(self, normalized_notes: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
        strong_candidates = [
            item for item in normalized_notes
            if item.get("is_relevant") and not item.get("is_marketing")
        ]
        if strong_candidates:
            ranked = sorted(
                strong_candidates,
                key=lambda item: (
                    -int(item.get("relevance_score") or 0),
                    -int(item.get("engagement_score") or 0),
                    -len(str(item.get("summary") or "")),
                ),
            )
            return ranked[:limit]

        fallback_candidates = [
            item for item in normalized_notes
            if not item.get("is_marketing")
            and (int(item.get("relevance_score") or 0) >= 0)
            and (str(item.get("title") or "").strip() or str(item.get("summary") or "").strip())
        ]
        ranked_fallback = sorted(
            fallback_candidates,
            key=lambda item: (
                -int(item.get("engagement_score") or 0),
                -int(item.get("relevance_score") or 0),
                -len(str(item.get("summary") or "")),
            ),
        )
        if ranked_fallback:
            return ranked_fallback[:limit]

        emergency_candidates = [
            item for item in normalized_notes
            if (str(item.get("title") or "").strip() or str(item.get("summary") or "").strip())
        ]
        emergency_ranked = sorted(
            emergency_candidates,
            key=lambda item: (
                int(item.get("marketing_score") or 0),
                -int(item.get("engagement_score") or 0),
                -len(str(item.get("summary") or "")),
            ),
        )
        return emergency_ranked[:limit]

    def _engagement_score(self, liked_count: Any, collect_count: Any, comment_count: Any) -> int:
        like_score = self._to_int(liked_count)
        collect_score = self._to_int(collect_count)
        comment_score = self._to_int(comment_count)
        return like_score + collect_score * 2 + comment_score * 3

    def _to_int(self, value: Any) -> int:
        try:
            if value in (None, ""):
                return 0
            if isinstance(value, str):
                normalized = value.replace("w", "0000").replace("W", "0000").replace(",", "").strip()
                return int(float(normalized))
            return int(float(value))
        except (TypeError, ValueError):
            return 0

    def _build_summary(self, keyword: str, notes: list[dict[str, Any]]) -> str:
        if not notes:
            return f"未检索到与“{keyword}”直接相关、且适合旅行规划参考的小红书笔记。"

        lines = [f"围绕“{keyword}”筛出 {len(notes)} 篇较有参考价值的小红书笔记："]
        for idx, note in enumerate(notes, start=1):
            metrics = []
            if note.get("liked_count") not in (None, ""):
                metrics.append(f"点赞{note['liked_count']}")
            if note.get("collect_count") not in (None, ""):
                metrics.append(f"收藏{note['collect_count']}")
            if note.get("comment_count") not in (None, ""):
                metrics.append(f"评论{note['comment_count']}")
            metric_text = f"（{'，'.join(metrics)}）" if metrics else ""
            summary = note.get("summary") or ""
            author = note.get("author") or "未知作者"
            title = note.get("title") or f"笔记{idx}"
            tags = note.get("travel_tags") or []
            tag_text = f" [关注点：{'、'.join(tags[:3])}]" if tags else ""
            lines.append(f"{idx}. {title} - {author}{metric_text}{tag_text}：{summary}")
        return "\n".join(lines)

    def _build_insights(self, keyword: str, notes: list[dict[str, Any]]) -> list[str]:
        if not notes:
            return [f"围绕“{keyword}”暂无足够干净的小红书经验可供总结。"]

        tag_counts: dict[str, int] = {}
        area_hints: list[str] = []
        action_hints: list[str] = []
        for note in notes:
            for tag in note.get("travel_tags") or []:
                tag_counts[tag] = tag_counts.get(tag, 0) + 1
            summary = note.get("summary") or ""
            if any(token in summary for token in ["住", "酒店", "民宿", "区域", "地铁"]):
                area_hints.append(summary)
            if any(token in summary for token in ["建议", "避坑", "预约", "排队", "打车", "步行"]):
                action_hints.append(summary)

        top_tags = sorted(tag_counts.items(), key=lambda item: (-item[1], item[0]))[:5]
        insights: list[str] = []
        if top_tags:
            insights.append("高频关注点：" + "、".join(tag for tag, _ in top_tags))
        if area_hints:
            insights.append("住宿/区域线索：" + self._truncate("；".join(area_hints[:2]), 120))
        if action_hints:
            insights.append("玩法/避坑线索：" + self._truncate("；".join(action_hints[:2]), 120))
        return insights or [f"围绕“{keyword}”有若干碎片经验，可结合其他工具进一步组织行程。"]

    def _clean_text(self, text: str) -> str:
        text = text or ""
        text = re.sub(r"#([^#\s]+)", r"\1", text)
        text = re.sub(r"@[^\s]+", "", text)
        text = re.sub(r"http[s]?://\S+", "", text)
        text = re.sub(r"[\u200b\ufeff]", "", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip(" ，。；;\n\t")

    def _summarize_description(self, text: str) -> str:
        cleaned = self._clean_text(text)
        if not cleaned:
            return ""

        fragments = re.split(r"[。！？!?.；;\n]", cleaned)
        useful_fragments: list[str] = []
        for fragment in fragments:
            fragment = fragment.strip(" ，")
            if len(fragment) < 4:
                continue
            if self._marketing_score(fragment) >= 2:
                continue
            useful_fragments.append(fragment)
            if len(useful_fragments) >= 2:
                break

        summary = "；".join(useful_fragments) if useful_fragments else cleaned
        return self._truncate(summary, 140)

    def _marketing_score(self, text: str) -> int:
        lowered = text.lower()
        score = 0
        for keyword in self.marketing_keywords:
            if keyword.lower() in lowered:
                score += 1
        if re.search(r"\bvx\b|v信|微信|weixin", lowered):
            score += 1
        if re.search(r"\d{5,}", lowered):
            score += 1
        return score

    def _relevance_score(self, keyword: str, text: str, travel_tags: list[str]) -> int:
        score = 0
        text_lower = text.lower()
        keyword_parts = [part.strip().lower() for part in re.split(r"\s+", keyword) if part.strip()]
        for part in keyword_parts:
            if len(part) >= 2 and part in text_lower:
                score += 1
        if travel_tags:
            score += 1
        if any(city in text for city in ["北京", "上海", "杭州", "成都", "重庆", "广州", "深圳", "西安", "厦门", "青岛", "武汉"]):
            score += 1
        return score

    def _extract_travel_tags(self, text: str) -> list[str]:
        tags = [keyword for keyword in self.travel_keywords if keyword in text]
        return tags[:6]

    def _pick_first_str(self, item: dict[str, Any], keys: list[str]) -> str:
        value = self._pick_first(item, keys)
        if value is None:
            return ""
        return str(value).strip()

    def _pick_first(self, item: dict[str, Any], keys: list[str]) -> Any:
        for key in keys:
            if key in item and item.get(key) not in (None, ""):
                return item.get(key)
        return None

    def _pick_nested_str(self, item: dict[str, Any], paths: list[tuple[str, str]]) -> str:
        for parent, child in paths:
            parent_value = item.get(parent)
            if isinstance(parent_value, dict):
                child_value = parent_value.get(child)
                if child_value not in (None, ""):
                    return str(child_value).strip()
        return ""

    def _truncate(self, text: str, limit: int) -> str:
        cleaned = " ".join((text or "").split())
        if len(cleaned) <= limit:
            return cleaned
        return cleaned[: limit - 1] + "…"
