from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.rag.embedding import TextEmbeddingV4Model


@dataclass
class RetrievedDoc:
    content: str
    score: float
    metadata: dict[str, Any]


class HybridRetriever:
    def __init__(self, embedding_model: TextEmbeddingV4Model) -> None:
        self.embedding_model = embedding_model

    def retrieve(self, query: str) -> list[dict[str, Any]]:
        bm25_hits = self._bm25(query)
        vector_hits = self._vector(query)
        merged = self._merge(bm25_hits, vector_hits)
        reranked = self._rerank(query, merged)
        return [doc.__dict__ for doc in reranked[:5]]

    def _bm25(self, query: str) -> list[RetrievedDoc]:
        return [RetrievedDoc(content=f"BM25: {query} 相关内容 {i}", score=1.0 / (i + 1), metadata={"method": "bm25"}) for i in range(5)]

    def _vector(self, query: str) -> list[RetrievedDoc]:
        _ = self.embedding_model.embed([query])
        return [RetrievedDoc(content=f"Vector: {query} 相关内容 {i}", score=0.9 / (i + 1), metadata={"method": "vector"}) for i in range(5)]

    def _merge(self, a: list[RetrievedDoc], b: list[RetrievedDoc]) -> list[RetrievedDoc]:
        seen: set[str] = set()
        merged: list[RetrievedDoc] = []
        for doc in a + b:
            if doc.content in seen:
                continue
            seen.add(doc.content)
            merged.append(doc)
        return merged

    def _rerank(self, query: str, docs: list[RetrievedDoc]) -> list[RetrievedDoc]:
        return sorted(docs, key=lambda d: d.score + (0.2 if query and query in d.content else 0), reverse=True)
