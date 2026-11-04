"""向量存储：Qdrant 生产适配 + 本地 NumPy 向量库降级。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from ..indexing.embeddings import cosine_scores
from ..models import Chunk


class InMemoryVectorStore:
    """本地向量库：向量矩阵 + Payload 持久化到磁盘，支持百万级以下规模。"""

    name = "local"

    def __init__(self) -> None:
        self.ids: list[str] = []
        self.vectors: np.ndarray | None = None
        self.payloads: dict[str, dict[str, Any]] = {}

    def __len__(self) -> int:
        return len(self.ids)

    def clear(self) -> None:
        self.ids, self.vectors, self.payloads = [], None, {}

    def upsert(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        if not chunks:
            return
        incoming = {chunk.chunk_id: (chunk, vector) for chunk, vector in zip(chunks, vectors)}
        keep = [i for i, cid in enumerate(self.ids) if cid not in incoming]
        if self.vectors is not None and keep:
            self.vectors = self.vectors[keep]
        elif not keep:
            self.vectors = None
        self.ids = [self.ids[i] for i in keep]
        if self.ids:
            new_vectors = np.vstack([self.vectors, np.asarray([v for _, v in incoming.values()], dtype=np.float32)])
        else:
            new_vectors = np.asarray([v for _, v in incoming.values()], dtype=np.float32)
        self.vectors = new_vectors
        self.ids.extend(incoming.keys())
        for chunk, _ in incoming.values():
            self.payloads[chunk.chunk_id] = {
                "doc_id": chunk.doc_id,
                "doc_title": chunk.doc_title,
                "section": " / ".join(chunk.section_path),
                "tenant": chunk.tenant,
                "page_start": chunk.page_start,
                "page_end": chunk.page_end,
            }

    def search(self, query: np.ndarray, top_k: int, tenant: str) -> list[tuple[str, float]]:
        if not self.ids or self.vectors is None:
            return []
        eligible = [i for i, cid in enumerate(self.ids) if self.payloads.get(cid, {}).get("tenant", "default") == tenant]
        if not eligible:
            return []
        subset = self.vectors[eligible]
        scores = cosine_scores(query, subset)
        order = np.argsort(-scores)[:top_k]
        return [(self.ids[eligible[int(i)]], float(scores[int(i)])) for i in order]

    def delete(self, chunk_ids: list[str]) -> None:
        target = set(chunk_ids)
        keep = [i for i, cid in enumerate(self.ids) if cid not in target]
        self.ids = [self.ids[i] for i in keep]
        self.vectors = self.vectors[keep] if self.vectors is not None and keep else None
        for cid in target:
            self.payloads.pop(cid, None)

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        np.save(directory / "vectors.npy", self.vectors if self.vectors is not None else np.zeros((0, 0)))
        (directory / "vector_store.json").write_text(
            json.dumps({"ids": self.ids, "payloads": self.payloads}, ensure_ascii=False),
            encoding="utf-8",
        )

    def load(self, directory: Path) -> None:
        meta_path = directory / "vector_store.json"
        vector_path = directory / "vectors.npy"
        if not meta_path.exists() or not vector_path.exists():
            return
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        self.ids = meta.get("ids", [])
        self.payloads = meta.get("payloads", {})
        vectors = np.load(vector_path)
        self.vectors = vectors if len(self.ids) else None


class QdrantVectorStore:
    name = "qdrant"

    def __init__(self, url: str, collection: str, dim: int, api_key: str = "") -> None:
        from qdrant_client import QdrantClient  # type: ignore[import-not-found]

        self.collection = collection
        self.dim = dim
        self.client = QdrantClient(url=url, api_key=api_key or None, timeout=10)
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        from qdrant_client.models import Distance, VectorParams  # type: ignore[import-not-found]

        if not self.client.collection_exists(self.collection):
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(size=self.dim, distance=Distance.COSINE),
            )

    def __len__(self) -> int:
        return int(self.client.count(self.collection).count)

    def clear(self) -> None:
        self.client.delete_collection(self.collection)
        self._ensure_collection()

    def upsert(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        from qdrant_client.models import PointStruct  # type: ignore[import-not-found]

        points = [
            PointStruct(
                id=_qdrant_id(chunk.chunk_id),
                vector=vector.tolist(),
                payload={
                    "chunk_id": chunk.chunk_id,
                    "doc_id": chunk.doc_id,
                    "doc_title": chunk.doc_title,
                    "section": " / ".join(chunk.section_path),
                    "tenant": chunk.tenant,
                    "page_start": chunk.page_start,
                    "page_end": chunk.page_end,
                },
            )
            for chunk, vector in zip(chunks, vectors)
        ]
        self.client.upsert(collection_name=self.collection, points=points, wait=True)

    def search(self, query: np.ndarray, top_k: int, tenant: str) -> list[tuple[str, float]]:
        from qdrant_client.models import FieldCondition, Filter, MatchValue  # type: ignore[import-not-found]

        flt = Filter(must=[FieldCondition(key="tenant", match=MatchValue(value=tenant))])
        hits = self.client.query_points(
            collection_name=self.collection,
            query=query.tolist(),
            query_filter=flt,
            limit=top_k,
            with_payload=True,
        ).points
        return [(hit.payload["chunk_id"], float(hit.score)) for hit in hits]

    def delete(self, chunk_ids: list[str]) -> None:
        if chunk_ids:
            self.client.delete(
                collection_name=self.collection,
                points_selector=[_qdrant_id(cid) for cid in chunk_ids],
                wait=True,
            )

    def save(self, directory: Path) -> None:  # Qdrant 自身持久化，无需落盘
        return None

    def load(self, directory: Path) -> None:
        return None


def _qdrant_id(chunk_id: str) -> int:
    """Qdrant 仅接受 u64 / UUID；把 16 位 hex 稳定映射为 64 位整数。"""
    return int(chunk_id[:16], 16)


def build_vector_store(settings, dim: int):
    if settings.qdrant_url:
        try:
            return QdrantVectorStore(settings.qdrant_url, settings.qdrant_collection, dim, settings.qdrant_api_key)
        except Exception:  # noqa: BLE001 - 连接失败时回退本地
            pass
    return InMemoryVectorStore()
