"""Self-RAG 状态机：自适应直答、检索生成、反思与引用。"""

from __future__ import annotations

from pathlib import Path

from graphrag.bootstrap import Container
from graphrag.models import RAGOptions


def test_direct_answer_skips_retrieval(container: Container) -> None:
    response = container.engine.run("你好")
    assert response.query_type == "direct"
    assert response.need_retrieval is False
    assert response.retrieved == []
    assert response.citations == []
    assert response.reflections.is_use == 1.0


def test_grounded_answer_has_citations_and_trace(container: Container, samples_dir: Path) -> None:
    for path in sorted(samples_dir.iterdir()):
        if path.suffix.lower() not in {".md", ".txt"}:
            continue
        container.ingestion.process_bytes_sync(path.read_bytes(), path.name)

    response = container.engine.run(
        "Battery-X1 的循环寿命和充电时间是多少？", RAGOptions(top_k=8)
    )
    assert response.answer
    assert response.query_type in {"single_hop", "multi_hop"}
    assert response.citations, "回答应带引用"
    assert all(citation.chunk_id for citation in response.citations)
    assert response.retrieval.channel_ranks

    traces = container.tracer.list(kind="query", limit=5)
    assert traces
    assert traces[0].steps


def test_empty_knowledge_base_answers_gracefully(container: Container) -> None:
    response = container.engine.run("公司差旅报销标准是多少？")
    assert "未" in response.answer or "没有" in response.answer
    assert any("缺少文档依据" in warning for warning in response.warnings)
