"""端到端演示脚本：入库内置示例 → 混合检索 → Self-RAG 回答 → 可选评测。

用法：
    python scripts/demo.py
    python scripts/demo.py --question "CE 认证和 FCC 认证有什么区别？"
    python scripts/demo.py --data-dir data/demo --eval
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from graphrag.bootstrap import Container  # noqa: E402
from graphrag.config import Settings  # noqa: E402
from graphrag.models import RAGOptions  # noqa: E402


def build_settings(data_dir: str) -> Settings:
    settings = Settings.from_env()
    return settings.model_copy(
        update={
            "data_dir": (ROOT / data_dir).resolve(),
            "embedding_backend": "hashing",
            "enable_hyde": True,
        }
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Graph-RAG 端到端演示")
    parser.add_argument("--data-dir", default="data/demo", help="数据目录")
    parser.add_argument("--question", action="append", help="问题，可重复传入")
    parser.add_argument("--eval", action="store_true", help="运行内置评测集")
    parser.add_argument("--skip-ingest", action="store_true", help="跳过示例入库")
    args = parser.parse_args()

    settings = build_settings(args.data_dir)
    container = Container(settings)

    if not args.skip_ingest:
        sample_dir = ROOT / "samples"
        for path in sorted(sample_dir.iterdir()):
            if path.suffix.lower() not in {".md", ".txt"}:
                continue
            container.ingestion.process_bytes_sync(path.read_bytes(), path.name)
            print(f"[ingest] {path.name}")

    print("\n[index]", container.index.stats())

    questions = args.question or [
        "锂电池产品出口需要哪些认证？",
        "CE 认证和 FCC 认证有什么区别？",
        "Battery-X1 的循环寿命和充电时间是多少？",
    ]
    for question in questions:
        response = container.engine.run(question, RAGOptions(top_k=8))
        print("\n" + "=" * 78)
        print("Q:", question)
        print(
            f"[type={response.query_type}] [latency={response.latency_ms}ms] "
            f"[rel={response.reflections.is_rel:.2f}] [sup={response.reflections.is_sup:.2f}] "
            f"[use={response.reflections.is_use:.2f}]"
        )
        print("A:", response.answer)
        print("citations:", [(c.index, c.doc_title, c.section, c.page) for c in response.citations])
        if response.warnings:
            print("warnings:", response.warnings)
        if response.rewritten_queries:
            print("rewrites:", response.rewritten_queries)
        print("channels:", {
            name: len(hits) for name, hits in response.retrieval.channel_ranks.items()
        })

    if args.eval:
        samples = container.evaluator.load_samples(ROOT / "samples" / "eval_dataset.jsonl")
        report = container.evaluator.evaluate(samples, top_k=8)
        print("\n[eval]", {
            "samples": report.num_samples,
            "faithfulness": report.faithfulness,
            "answer_relevance": report.answer_relevance,
            "context_recall": report.context_recall,
            "context_precision": report.context_precision,
            "avg_latency_ms": report.avg_latency_ms,
            "p95_latency_ms": report.p95_latency_ms,
        })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
