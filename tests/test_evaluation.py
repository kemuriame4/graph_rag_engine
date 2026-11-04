"""Ragas 口径评测：四项指标范围与报告持久化。"""

from __future__ import annotations

from pathlib import Path

from graphrag.bootstrap import Container
from graphrag.models import EvalSample


def test_score_sample_is_bounded(container: Container) -> None:
    scores = container.evaluator.score_sample(
        question="Battery-X1 的循环寿命是多少？",
        answer="Battery-X1 的额定循环寿命为 3000 次。",
        ground_truth="Battery-X1 的额定循环寿命为 3000 次。",
        contexts=["Cycle life | 3000 | cycles"],
    )
    for value in (
        scores.faithfulness,
        scores.answer_relevance,
        scores.context_recall,
        scores.context_precision,
    ):
        assert 0.0 <= value <= 1.0


def test_evaluate_persists_and_lists_report(container: Container, samples_dir: Path) -> None:
    for path in sorted(samples_dir.iterdir()):
        if path.suffix.lower() not in {".md", ".txt"}:
            continue
        container.ingestion.process_bytes_sync(path.read_bytes(), path.name)

    report = container.evaluator.evaluate(
        [
            EvalSample(
                question="CE 认证和 FCC 认证分别适用于哪些市场？",
                ground_truth="CE 适用于欧盟市场，FCC 适用于美国市场。",
            )
        ],
        top_k=6,
    )
    assert report.num_samples == 1
    assert report.backend in {"internal", "ragas"}
    assert 0.0 <= report.faithfulness <= 1.0
    assert report.avg_latency_ms > 0

    listed = container.evaluator.list_reports(limit=5)
    assert any(item.report_id == report.report_id for item in listed)
    assert container.evaluator.get_report(report.report_id) is not None
    assert Path(container.settings.eval_dir / "reports" / f"{report.report_id}.json").exists()
