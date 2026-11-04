"""稀疏检索：Elasticsearch BM25 适配 + 纯 Python BM25 降级。"""

from __future__ import annotations

import math
from collections import Counter, defaultdict

from ..models import Chunk
from ..text import tokenize


class InMemoryBM25:
    """标准 Okapi BM25（k1=1.5, b=0.75），中英混合分词在 text.tokenize 内完成。"""

    name = "local-bm25"

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.documents: dict[str, list[str]] = {}
        self.tenants: dict[str, str] = {}
        self.doc_freq: Counter[str] = Counter()
        self.avgdl: float = 0.0

    def __len__(self) -> int:
        return len(self.documents)

    def clear(self) -> None:
        self.documents.clear()
        self.tenants.clear()
        self.doc_freq.clear()
        self.avgdl = 0.0

    def upsert(self, chunks: list[Chunk]) -> None:
        for chunk in chunks:
            if chunk.chunk_id in self.documents:
                self._remove(chunk.chunk_id)
            tokens = tokenize(chunk.text)
            self.documents[chunk.chunk_id] = tokens
            self.tenants[chunk.chunk_id] = chunk.tenant
            self.doc_freq.update(set(tokens))
        self._recompute()

    def delete(self, chunk_ids: list[str]) -> None:
        for chunk_id in chunk_ids:
            self._remove(chunk_id)
        self._recompute()

    def _remove(self, chunk_id: str) -> None:
        tokens = self.documents.pop(chunk_id, None)
        self.tenants.pop(chunk_id, None)
        if tokens:
            self.doc_freq.subtract(set(tokens))
            self.doc_freq += Counter()  # 去掉 <=0 的项

    def _recompute(self) -> None:
        self.avgdl = (
            sum(len(tokens) for tokens in self.documents.values()) / len(self.documents)
            if self.documents
            else 0.0
        )

    def search(self, query: str, top_k: int, tenant: str) -> list[tuple[str, float]]:
        if not self.documents:
            return []
        query_tokens = tokenize(query)
        if not query_tokens:
            return []
        total = len(self.documents)
        scores: list[tuple[str, float]] = []
        for chunk_id, tokens in self.documents.items():
            if self.tenants.get(chunk_id, "default") != tenant:
                continue
            counter = Counter(tokens)
            length = len(tokens)
            score = 0.0
            for token in query_tokens:
                frequency = counter.get(token, 0)
                if not frequency:
                    continue
                df = self.doc_freq.get(token, 0)
                idf = math.log(1 + (total - df + 0.5) / (df + 0.5))
                denominator = frequency + self.k1 * (
                    1 - self.b + self.b * (length / (self.avgdl + 1e-9))
                )
                score += idf * (frequency * (self.k1 + 1)) / denominator
            if score > 0:
                scores.append((chunk_id, score))
        scores.sort(key=lambda item: item[1], reverse=True)
        return scores[:top_k]


class ElasticsearchBM25:
    name = "elasticsearch"

    def __init__(self, url: str, index: str, api_key: str = "") -> None:
        from elasticsearch import Elasticsearch  # type: ignore[import-not-found]

        self.index = index
        self.client = Elasticsearch(url, api_key=api_key or None, request_timeout=10)
        if not self.client.indices.exists(index=index):
            self.client.indices.create(
                index=index,
                mappings={
                    "properties": {
                        "text": {"type": "text"},
                        "doc_title": {"type": "text"},
                        "section": {"type": "text"},
                        "tenant": {"type": "keyword"},
                        "doc_id": {"type": "keyword"},
                        "chunk_id": {"type": "keyword"},
                    }
                },
            )

    def __len__(self) -> int:
        try:
            return int(self.client.count(index=self.index)["count"])
        except Exception:  # noqa: BLE001
            return 0

    def clear(self) -> None:
        self.client.delete_by_query(index=self.index, body={"query": {"match_all": {}}})

    def upsert(self, chunks: list[Chunk]) -> None:
        for chunk in chunks:
            self.client.index(
                index=self.index,
                id=chunk.chunk_id,
                document={
                    "chunk_id": chunk.chunk_id,
                    "doc_id": chunk.doc_id,
                    "doc_title": chunk.doc_title,
                    "section": " / ".join(chunk.section_path),
                    "tenant": chunk.tenant,
                    "text": chunk.text,
                },
                refresh=False,
            )
        self.client.indices.refresh(index=self.index)

    def delete(self, chunk_ids: list[str]) -> None:
        for chunk_id in chunk_ids:
            try:
                self.client.delete(index=self.index, id=chunk_id, refresh=False)
            except Exception:  # noqa: BLE001 - 文档已不存在
                continue

    def search(self, query: str, top_k: int, tenant: str) -> list[tuple[str, float]]:
        response = self.client.search(
            index=self.index,
            size=top_k,
            query={
                "bool": {
                    "must": [
                        {
                            "multi_match": {
                                "query": query,
                                "fields": ["text^2", "section^1.2", "doc_title^1.5"],
                                "type": "best_fields",
                            }
                        }
                    ],
                    "filter": [{"term": {"tenant": tenant}}],
                }
            },
        )
        return [(hit["_id"], float(hit["_score"])) for hit in response["hits"]["hits"]]


def build_bm25(settings):
    if settings.es_url:
        try:
            return ElasticsearchBM25(settings.es_url, settings.es_index, settings.es_api_key)
        except Exception:  # noqa: BLE001
            pass
    return InMemoryBM25()
