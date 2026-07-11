from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from app.config.settings import get_settings


class EmbeddingModel(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]:
        ...


@dataclass
class TextEmbeddingV4Model:
    model_name: str | None = None

    def embed(self, texts: list[str]) -> list[list[float]]:
        settings = get_settings()
        _ = self.model_name or settings.qwen_embedding_model
        vectors: list[list[float]] = []
        for text in texts:
            score = float(sum(ord(ch) for ch in text) % 1000)
            vectors.append([score, float(len(text)), float(text.count("旅游"))])
        return vectors
