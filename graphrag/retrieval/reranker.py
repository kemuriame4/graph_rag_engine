"""重排层：BGE Cross-Encoder 优先，缺失时使用可解释的启发式融合重排。"""

from __future__ import annotations

import numpy as np

from ..config import Settings
from ..models import RetrievedDoc
from ..text import keyword_overlap


class HeuristicReranker:
    """无模型重排：归一化三路分数 + Query/正文词法重叠。"""

    name = "heuristic"

    def rerank(self, query: str, docs: list[RetrievedDoc], top_n: int) -> list[RetrievedDoc]:
        if not docs:
            return []
        rrf_values = np.array([doc.rrf_score for doc in docs], dtype=np.float32)
        rrf_norm = _minmax(rrf_values)

        def norm(value: float | None, arr: list[float]) -> float:
            if value is None or not arr:
                return 0.0
            low, high = min(arr), max(arr)
            return (value - low) / (high - low) if high > low else 1.0

        dense_values = [doc.dense_score for doc in docs if doc.dense_score is not None]
        sparse_values = [doc.sparse_score for doc in docs if doc.sparse_score is not None]
        graph_values = [doc.graph_score for doc in docs if doc.graph_score is not None]

        for index, doc in enumerate(docs):
            lexical = keyword_overlap(query, doc.text)
            score = (
                0.30 * float(rrf_norm[index])
                + 0.28 * norm(doc.dense_score, dense_values)
                + 0.18 * norm(doc.sparse_score, sparse_values)
                + 0.12 * norm(doc.graph_score, graph_values)
                + 0.12 * lexical
            )
            doc.rerank_score = round(min(1.0, score), 4)
        ranked = sorted(docs, key=lambda item: item.rerank_score or 0.0, reverse=True)
        return ranked[:top_n]


class CrossEncoderReranker:
    name = "cross-encoder"

    def __init__(self, model_name: str) -> None:
        from sentence_transformers import CrossEncoder  # type: ignore[import-not-found]

        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, docs: list[RetrievedDoc], top_n: int) -> list[RetrievedDoc]:
        if not docs:
            return []
        pairs = [(query, doc.text) for doc in docs]
        scores = self.model.predict(pairs)
        for doc, score in zip(docs, scores):
            doc.rerank_score = round(float(score), 4)
        ranked = sorted(docs, key=lambda item: item.rerank_score or 0.0, reverse=True)
        return ranked[:top_n]


class FallbackReranker:
    name = "heuristic-fallback"

    def __init__(self, primary, fallback) -> None:
        self.primary = primary
        self.fallback = fallback
        self.name = primary.name

    def rerank(self, query: str, docs: list[RetrievedDoc], top_n: int) -> list[RetrievedDoc]:
        try:
            return self.primary.rerank(query, docs, top_n)
        except Exception:  # noqa: BLE001
            self.name = "heuristic"
            return self.fallback.rerank(query, docs, top_n)


def build_reranker(settings: Settings):
    heuristic = HeuristicReranker()
    if settings.reranker_backend == "heuristic":
        return heuristic
    if settings.reranker_backend in {"auto", "cross-encoder"}:
        try:
            cross = CrossEncoderReranker(settings.rerank_model)
            return FallbackReranker(cross, heuristic)
        except Exception:  # noqa: BLE001
            return heuristic
    return heuristic


def _minmax(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return values
    low, high = float(values.min()), float(values.max())
    if high - low < 1e-9:
        return np.ones_like(values)
    return (values - low) / (high - low)
