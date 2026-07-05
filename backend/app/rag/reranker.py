from __future__ import annotations

from dataclasses import dataclass

from backend.app.rag.retriever import RetrievedDoc


@dataclass
class Reranker:
    model_name: str

    def rerank(self, query: str, docs: list[RetrievedDoc]) -> list[RetrievedDoc]:
        return sorted(docs, key=lambda d: d.score + (0.3 if query in d.content else 0), reverse=True)
