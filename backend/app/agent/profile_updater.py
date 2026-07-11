from __future__ import annotations

from app.domain.models import UserProfile


class ProfileUpdater:
    def apply_intent(self, profile: UserProfile, intent: dict) -> UserProfile:
        if intent.get("destination"):
            profile.destination = intent["destination"]
        if intent.get("days"):
            profile.days = intent["days"]
        if intent.get("budget"):
            profile.budget = intent["budget"]
        if intent.get("companions"):
            profile.companions = intent["companions"]
        if intent.get("preferences"):
            merged = list(dict.fromkeys([*profile.preferences, *intent["preferences"]]))
            profile.preferences = merged
        return profile
