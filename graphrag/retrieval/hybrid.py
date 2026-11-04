"""三路混合召回：Dense + Sparse(BM25) + Graph，RRF 融合后 Cross-Encoder 重排。"""

from __future__ import annotations

import time

from ..config import Settings
from ..indexing import SearchIndex
from ..models import ChannelHit, RAGOptions, RetrievedDoc, RetrievalTrace
from ..text import highlight_terms
from .reranker import build_reranker
from .rrf import reciprocal_rank_fusion


class HybridRetriever:
    def __init__(self, index: SearchIndex, settings: Settings) -> None:
        self.index = index
        self.settings = settings
        self.reranker = build_reranker(settings)

    def search(
        self,
        query: str,
        options: RAGOptions,
        hyde_query: str | None = None,
    ) -> tuple[list[RetrievedDoc], RetrievalTrace]:
        started = time.perf_counter()
        tenant = options.tenant
        channel_results: dict[str, list[tuple[str, float]]] = {}

        channel_results["dense"] = self.index.search_dense(query, tenant, self.settings.top_k_dense)
        channel_results["sparse"] = self.index.search_sparse(query, tenant, self.settings.top_k_sparse)

        if hyde_query:
            channel_results["hyde_dense"] = self.index.search_dense(
                hyde_query, tenant, self.settings.top_k_dense
            )

        graph_entities: list[str] = []
        graph_paths: dict[str, list[str]] = {}
        if options.enable_graph:
            graph_hits, graph_entities, paths = self.index.search_graph(
                query, tenant, self.settings.top_k_graph, options.max_hops
            )
            channel_results["graph"] = graph_hits
            for chunk_id, path in paths:
                graph_paths.setdefault(chunk_id, path)

        fused = reciprocal_rank_fusion(channel_results, self.settings.rrf_k)

        docs: list[RetrievedDoc] = []
        for hit in fused:
            chunk = self.index.get_chunk(hit.chunk_id)
            if chunk is None or chunk.tenant != tenant:
                continue
            docs.append(
                RetrievedDoc(
                    chunk_id=chunk.chunk_id,
                    doc_id=chunk.doc_id,
                    doc_title=chunk.doc_title,
                    text=chunk.text,
                    section_path=chunk.section_path,
                    page_start=chunk.page_start,
                    page_end=chunk.page_end,
                    score=hit.rrf_score,
                    rrf_score=hit.rrf_score,
                    dense_score=hit.channel_scores.get("dense"),
                    sparse_score=hit.channel_scores.get("sparse"),
                    graph_score=hit.channel_scores.get("graph"),
                    channels=hit.channels,
                    graph_path=graph_paths.get(chunk.chunk_id, []),
                    highlights=highlight_terms(query, chunk.text),
                )
            )

        reranked = self.reranker.rerank(query, docs, options.top_k)

        trace = RetrievalTrace(
            channel_ranks={
                channel: [
                    ChannelHit(chunk_id=chunk_id, rank=rank, score=round(float(score), 4))
                    for rank, (chunk_id, score) in enumerate(results, start=1)
                ]
                for channel, results in channel_results.items()
            },
            fused_count=len(fused),
            reranked_count=len(reranked),
            hyde_query=hyde_query,
            graph_entities=graph_entities,
            elapsed_ms=round((time.perf_counter() - started) * 1000, 2),
        )
        return reranked, trace
