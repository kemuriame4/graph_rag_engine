"""索引、三路召回、RRF 融合与图谱多跳。"""

from __future__ import annotations

from pathlib import Path

from graphrag.bootstrap import Container
from graphrag.models import RAGOptions


def _ingest(container: Container, samples_dir: Path) -> None:
    for path in sorted(samples_dir.iterdir()):
        if path.suffix.lower() not in {".md", ".txt"}:
            continue
        container.ingestion.process_bytes_sync(path.read_bytes(), path.name)


def test_index_builds_three_channels(container: Container, samples_dir: Path) -> None:
    _ingest(container, samples_dir)
    stats = container.index.stats()
    assert stats["documents"] >= 3
    assert stats["chunks"] >= 10
    assert stats["entities"] > 0

    docs, trace = container.retriever.search("锂电池出口认证", RAGOptions(top_k=8))
    assert docs
    assert docs[0].doc_title
    assert "dense" in trace.channel_ranks and "sparse" in trace.channel_ranks
    assert trace.fused_count > 0
    assert trace.reranked_count <= 8


def test_graph_hits_are_recallable(container: Container, samples_dir: Path) -> None:
    _ingest(container, samples_dir)
    graph_hits, entities, _ = container.index.search_graph(
        "CE 认证", "default", top_k=10, max_hops=2
    )
    assert isinstance(graph_hits, list)
    assert entities, "应能从查询中链接到图谱实体"
    subgraph = container.index.graph.subgraph(entities[:2], hops=1, tenant="default")
    assert subgraph.nodes


def test_tenant_isolation(container: Container, samples_dir: Path) -> None:
    _ingest(container, samples_dir)
    docs, _ = container.retriever.search("锂电池", RAGOptions(tenant="other", top_k=8))
    assert docs == []
