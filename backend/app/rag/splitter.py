from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass
class Chunk:
    text: str
    metadata: dict[str, str]


class RecursiveSplitter:
    def split(self, text: str, max_length: int = 500) -> list[str]:
        if len(text) <= max_length:
            return [text]
        midpoint = len(text) // 2
        left = self.split(text[:midpoint], max_length)
        right = self.split(text[midpoint:], max_length)
        return left + right


class SemanticSplitter:
    def split(self, text: str) -> list[str]:
        chunks = [part.strip() for part in text.replace("\n", "。\n").split("。") if part.strip()]
        return chunks or [text]


class DocumentSplitter:
    def __init__(self) -> None:
        self.recursive = RecursiveSplitter()
        self.semantic = SemanticSplitter()

    def split(self, documents: Iterable[str]) -> list[Chunk]:
        chunks: list[Chunk] = []
        for doc in documents:
            for recursive_part in self.recursive.split(doc):
                for semantic_part in self.semantic.split(recursive_part):
                    chunks.append(Chunk(text=semantic_part, metadata={"source": "local"}))
        return chunks
