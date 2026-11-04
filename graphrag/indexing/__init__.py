"""索引编排：向量 / 稀疏 / 图谱三路索引的统一写入、查询与持久化。"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from ..chunking import build_embedding_text
from ..config import Settings
from ..models import Chunk, DocumentRecord, ParsedDocument
from .bm25 import InMemoryBM25, build_bm25
from .embeddings import build_embedder
from .entity_extractor import RuleEntityExtractor
from .graph_store import KnowledgeGraph
from .vector_store import InMemoryVectorStore, build_vector_store


class SearchIndex:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.embedder = build_embedder(settings)
        self.vector_store = build_vector_store(settings, self.embedder.dim or settings.embedding_dim)
        self.bm25 = build_bm25(settings)
        self.graph = KnowledgeGraph()
        self.extractor = RuleEntityExtractor()
        self.chunks: dict[str, Chunk] = {}
        self.documents: dict[str, DocumentRecord] = {}
        self._load()

    # ---------------- 写入 ----------------
    def index_document(self, document: ParsedDocument, chunks: list[Chunk]) -> dict:
        started = time.perf_counter()
        self.remove_document(document.doc_id)

        entity_names: set[str] = set()
        relation_map: dict[str, list] = {}
        for chunk in chunks:
            extraction = self.extractor.extract(chunk.text)
            chunk.entities = [entity.name for entity in extraction.entities]
            chunk.embedding_text = build_embedding_text(chunk)
            entity_names.update(entity.normalized for entity in extraction.entities)
            for relation in extraction.relations:
                relation_map.setdefault(chunk.chunk_id, []).append(relation)
            self.graph.add_chunk(chunk)
            for entity in extraction.entities:
                self.graph.add_entity(
                    name=entity.name,
                    normalized=entity.normalized,
                    entity_type=entity.type,
                    chunk_id=chunk.chunk_id,
                    doc_id=chunk.doc_id,
                    tenant=chunk.tenant,
                )
            for relation in extraction.relations:
                self.graph.add_relation(
                    relation=relation,
                    chunk_id=chunk.chunk_id,
                    doc_id=chunk.doc_id,
                    tenant=chunk.tenant,
                )

        if chunks:
            texts = [chunk.embedding_text or chunk.text for chunk in chunks]
            vectors = self.embedder.embed(texts)
            self.vector_store.upsert(chunks, vectors)
            self.bm25.upsert(chunks)

        self.chunks.update({chunk.chunk_id: chunk for chunk in chunks})
        self.documents[document.doc_id] = DocumentRecord(
            doc_id=document.doc_id,
            title=document.title,
            source=document.source,
            mime=document.mime,
            language=document.language,
            page_count=document.page_count,
            chunk_count=len(chunks),
            entity_count=len(entity_names),
            tenant=chunks[0].tenant if chunks else self.settings.default_tenant,
            created_at=time.time(),
            markdown=document.markdown[:2_000_000],
            metadata=document.metadata,
        )
        self.persist()

        return {
            "doc_id": document.doc_id,
            "chunks": len(chunks),
            "entities": len(entity_names),
            "relations": sum(len(v) for v in relation_map.values()),
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 2),
        }

    def remove_document(self, doc_id: str) -> None:
        chunk_ids = [chunk_id for chunk_id, chunk in self.chunks.items() if chunk.doc_id == doc_id]
        if chunk_ids:
            self.vector_store.delete(chunk_ids)
            self.bm25.delete(chunk_ids)
            for chunk_id in chunk_ids:
                self.chunks.pop(chunk_id, None)
        self.graph.remove_document(doc_id)
        self.documents.pop(doc_id, None)

    # ---------------- 查询 ----------------
    def search_dense(self, query: str, tenant: str, top_k: int | None = None) -> list[tuple[str, float]]:
        top_k = top_k or self.settings.top_k_dense
        vectors = self.embedder.embed([query])
        return self.vector_store.search(vectors[0], top_k, tenant)

    def search_sparse(self, query: str, tenant: str, top_k: int | None = None) -> list[tuple[str, float]]:
        return self.bm25.search(query, top_k or self.settings.top_k_sparse, tenant)

    def search_graph(
        self, query: str, tenant: str, top_k: int | None = None, max_hops: int | None = None
    ) -> tuple[list[tuple[str, float]], list[str], list[tuple[str, list[str]]]]:
        top_k = top_k or self.settings.top_k_graph
        max_hops = max_hops or self.settings.graph_max_hops
        links = self.graph.link_entities(query, tenant)
        if not links:
            return [], [], []
        hits, paths = self.graph.chunk_hits_for_entities(
            links, tenant=tenant, max_hops=max_hops
        )
        ordered = sorted(hits.items(), key=lambda item: item[1], reverse=True)[:top_k]
        entity_labels = [str(self.graph.graph.nodes[node_id].get("label", "")) for node_id, _ in links]
        return ordered, entity_labels, paths

    def get_chunk(self, chunk_id: str) -> Chunk | None:
        return self.chunks.get(chunk_id)

    def get_documents(self) -> list[DocumentRecord]:
        return sorted(self.documents.values(), key=lambda doc: doc.created_at, reverse=True)

    def get_document(self, doc_id: str) -> DocumentRecord | None:
        return self.documents.get(doc_id)

    def stats(self) -> dict:
        graph_stats = self.graph.stats()
        return {
            "documents": len(self.documents),
            "chunks": len(self.chunks),
            "vector_backend": self.vector_store.name,
            "sparse_backend": self.bm25.name,
            "embedding_backend": self.embedder.name,
            "embedding_dim": self.embedder.dim,
            "entities": graph_stats["entities"],
            "relations": graph_stats["relations"],
        }

    # ---------------- 持久化 ----------------
    def persist(self) -> None:
        directory = self.settings.index_dir
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "chunks.json").write_text(
            json.dumps([chunk.model_dump() for chunk in self.chunks.values()], ensure_ascii=False),
            encoding="utf-8",
        )
        (directory / "documents.json").write_text(
            json.dumps([doc.model_dump() for doc in self.documents.values()], ensure_ascii=False),
            encoding="utf-8",
        )
        self.vector_store.save(directory)
        self.graph.save(directory)

    def _load(self) -> None:
        directory = self.settings.index_dir
        chunks_path = directory / "chunks.json"
        documents_path = directory / "documents.json"
        if chunks_path.exists():
            try:
                payload = json.loads(chunks_path.read_text(encoding="utf-8"))
                for item in payload:
                    chunk = Chunk.model_validate(item)
                    self.chunks[chunk.chunk_id] = chunk
            except (json.JSONDecodeError, ValueError):
                self.chunks = {}
        if documents_path.exists():
            try:
                payload = json.loads(documents_path.read_text(encoding="utf-8"))
                for item in payload:
                    record = DocumentRecord.model_validate(item)
                    self.documents[record.doc_id] = record
            except (json.JSONDecodeError, ValueError):
                self.documents = {}

        if isinstance(self.vector_store, InMemoryVectorStore):
            self.vector_store.load(directory)
            self._repair_vector_dim()

        if isinstance(self.bm25, InMemoryBM25) and self.chunks:
            self.bm25.upsert(list(self.chunks.values()))

        self.graph.load(directory)

    def _repair_vector_dim(self) -> None:
        """配置变更导致 embedding 维度不一致时，自动重建本地向量。"""
        vectors = self.vector_store.vectors
        expected = self.embedder.dim or self.settings.embedding_dim
        if vectors is None or vectors.size == 0 or vectors.shape[1] == expected:
            return
        self.vector_store.clear()
        if not self.chunks:
            return
        chunks = list(self.chunks.values())
        texts = [chunk.embedding_text or chunk.text for chunk in chunks]
        try:
            matrix = self.embedder.embed(texts)
            self.vector_store.upsert(chunks, matrix)
        except Exception:  # noqa: BLE001
            self.vector_store.clear()

    def rebuild(self) -> dict:
        """从 chunks.json 全量重建三路索引（含 embedding）。"""
        chunks = list(self.chunks.values())
        self.vector_store.clear()
        self.bm25.clear()
        self.graph.clear()
        self.chunks = {}
        self.documents = {}
        if not chunks:
            self.persist()
            return self.stats()
        for chunk in chunks:
            chunk.embedding_text = chunk.embedding_text or build_embedding_text(chunk)
        vectors = self.embedder.embed([chunk.embedding_text for chunk in chunks])
        self.vector_store.upsert(chunks, vectors)
        self.bm25.upsert(chunks)
        for chunk in chunks:
            self.chunks[chunk.chunk_id] = chunk
            self.graph.add_chunk(chunk)
            for name in chunk.entities:
                self.graph.add_entity(name, name.lower().replace(" ", ""), "term", chunk.chunk_id, chunk.doc_id, chunk.tenant)
        self.persist()
        return self.stats()
