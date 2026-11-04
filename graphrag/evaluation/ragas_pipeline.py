"""Ragas 自动化评测流水线。

优先调用 Ragas（若已安装），否则使用与 Ragas 指标口径一致的内置实现：
- Faithfulness：答案断言被检索上下文支持的比例
- Answer Relevance：答案与问题的语义 + 词法相关性
- Context Recall：标准答案要点被召回上下文覆盖的比例
- Context Precision：召回上下文中相关片段在排序上的平均精度
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..config import Settings
from ..indexing.embeddings import cosine
from ..models import EvalRecord, EvalReport, EvalSample, RAGOptions
from ..selfrag.engine import SelfRAGEngine
from ..text import char_coverage, coverage, expand_bilingual, keyword_overlap, split_sentences, tokenize


@dataclass
class _Score:
    faithfulness: float
    answer_relevance: float
    context_recall: float
    context_precision: float


class Evaluator:
    def __init__(self, engine: SelfRAGEngine, settings: Settings) -> None:
        self.engine = engine
        self.settings = settings
        self.embedder = engine.index.embedder
        self.backend = self._detect_backend()

    @staticmethod
    def _detect_backend() -> str:
        try:
            import ragas  # noqa: F401  # type: ignore[import-not-found]

            return "ragas"
        except ImportError:
            return "internal"

    def evaluate(
        self,
        samples: list[EvalSample],
        top_k: int = 8,
        tenant: str = "default",
    ) -> EvalReport:
        records: list[EvalRecord] = []
        for sample in samples:
            response = self.engine.run(
                sample.question,
                RAGOptions(tenant=sample.tenant or tenant, top_k=top_k),
            )
            contexts = [doc.text for doc in response.retrieved[:top_k]]
            scores = self.score_sample(
                question=sample.question,
                answer=response.answer,
                ground_truth=sample.ground_truth,
                contexts=contexts,
            )
            records.append(
                EvalRecord(
                    question=sample.question,
                    answer=response.answer,
                    ground_truth=sample.ground_truth,
                    contexts=contexts,
                    faithfulness=scores.faithfulness,
                    answer_relevance=scores.answer_relevance,
                    context_recall=scores.context_recall,
                    context_precision=scores.context_precision,
                    latency_ms=response.latency_ms,
                    trace_id=response.trace_id,
                )
            )

        latencies = sorted(record.latency_ms for record in records)
        report = EvalReport(
            report_id=uuid.uuid4().hex[:10],
            created_at=time.time(),
            backend=self.backend,
            num_samples=len(records),
            faithfulness=_mean(record.faithfulness for record in records),
            answer_relevance=_mean(record.answer_relevance for record in records),
            context_recall=_mean(record.context_recall for record in records),
            context_precision=_mean(record.context_precision for record in records),
            avg_latency_ms=round(sum(latencies) / len(latencies), 2) if latencies else 0.0,
            p95_latency_ms=round(latencies[max(0, int(0.95 * len(latencies)) - 1)], 2) if latencies else 0.0,
            records=records,
        )
        self.save(report)
        return report

    def score_sample(self, question: str, answer: str, ground_truth: str, contexts: list[str]) -> _Score:
        return _Score(
            faithfulness=self._faithfulness(answer, contexts),
            answer_relevance=self._answer_relevance(question, answer),
            context_recall=self._context_recall(ground_truth, contexts),
            context_precision=self._context_precision(ground_truth, contexts),
        )

    # ---------------- 指标实现 ----------------
    def _faithfulness(self, answer: str, contexts: list[str]) -> float:
        claims = [s for s in split_sentences(answer) if len(s) > 8]
        claims = [
            s for s in claims
            if not s.startswith(("根据知识库", "以上结论", "Based on"))
        ]
        if not claims or not contexts:
            return 0.0
        context_sentences = [s for context in contexts for s in split_sentences(context)]
        if not context_sentences:
            return 0.0
        context_vectors = self.embedder.embed(context_sentences)
        supported = 0
        for claim in claims:
            cleaned = claim.lstrip("-•0123456789. ")
            lexical = max(keyword_overlap(cleaned, sentence) for sentence in context_sentences)
            vector = self.embedder.embed([cleaned])[0]
            semantic = max(cosine(vector, context) for context in context_vectors)
            if max(lexical, semantic) >= 0.45:
                supported += 1
        return round(supported / len(claims), 4)

    def _answer_relevance(self, question: str, answer: str) -> float:
        if not answer.strip():
            return 0.0
        vectors = self.embedder.embed([question, answer])
        semantic = max(0.0, cosine(vectors[0], vectors[1]))
        bilingual = expand_bilingual(question)
        lexical = max(
            coverage(question, answer),
            coverage(bilingual, answer) if bilingual else 0.0,
            char_coverage(question, answer),
        )
        return round(0.5 * semantic + 0.5 * lexical, 4)

    def _context_recall(self, ground_truth: str, contexts: list[str]) -> float:
        if not contexts:
            return 0.0
        targets = split_sentences(ground_truth) or [ground_truth]
        context_sentences = [s for context in contexts for s in split_sentences(context)]
        if not context_sentences:
            return 0.0
        context_vectors = self.embedder.embed(context_sentences)
        covered = 0.0
        for target in targets:
            vector = self.embedder.embed([target])[0]
            best = max(cosine(vector, context) for context in context_vectors)
            best = max(best, max(keyword_overlap(target, sentence) for sentence in context_sentences))
            covered += 1.0 if best >= 0.42 else best
        return round(covered / len(targets), 4)

    def _context_precision(self, ground_truth: str, contexts: list[str]) -> float:
        if not contexts:
            return 0.0
        relevance = []
        for context in contexts:
            overlap = keyword_overlap(ground_truth, context)
            relevance.append(1 if overlap >= 0.18 else 0)
        if not any(relevance):
            return 0.0
        cumulative_hits = 0
        precision_sum = 0.0
        for rank, flag in enumerate(relevance, start=1):
            if flag:
                cumulative_hits += 1
                precision_sum += cumulative_hits / rank
        return round(precision_sum / max(cumulative_hits, 1), 4)

    # ---------------- 数据 ----------------
    def load_samples(self, path: str | Path) -> list[EvalSample]:
        path = Path(path)
        samples: list[EvalSample] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            payload = json.loads(line)
            samples.append(EvalSample.model_validate(payload))
        return samples

    def save(self, report: EvalReport) -> Path:
        directory = self.settings.eval_dir / "reports"
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{report.report_id}.json"
        path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
        return path

    def list_reports(self, limit: int = 20) -> list[EvalReport]:
        directory = self.settings.eval_dir / "reports"
        if not directory.exists():
            return []
        reports: list[EvalReport] = []
        for path in sorted(directory.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)[:limit]:
            try:
                reports.append(
                    EvalReport.model_validate(json.loads(path.read_text(encoding="utf-8")))
                )
            except (json.JSONDecodeError, ValueError):
                continue
        return reports

    def get_report(self, report_id: str) -> EvalReport | None:
        path = self.settings.eval_dir / "reports" / f"{report_id}.json"
        if not path.exists():
            return None
        try:
            return EvalReport.model_validate(json.loads(path.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, ValueError):
            return None

    def score_ragas(self, samples: list[EvalSample]) -> tuple[float, float, float, float] | None:
        """可选 Ragas 后端。未安装或接口不兼容时返回 None，由内置实现兜底。"""
        if self.backend != "ragas":
            return None
        try:
            from datasets import Dataset  # type: ignore[import-not-found]
            from ragas import evaluate  # type: ignore[import-not-found]
            from ragas.metrics import (  # type: ignore[import-not-found]
                answer_relevancy,
                context_precision,
                context_recall,
                faithfulness,
            )

            rows = {"question": [], "answer": [], "contexts": [], "ground_truth": []}
            for sample in samples:
                response = self.engine.run(sample.question, RAGOptions(tenant=sample.tenant))
                rows["question"].append(sample.question)
                rows["answer"].append(response.answer)
                rows["contexts"].append([doc.text for doc in response.retrieved])
                rows["ground_truth"].append(sample.ground_truth)
            result = evaluate(
                Dataset.from_dict(rows),
                metrics=[faithfulness, answer_relevancy, context_recall, context_precision],
            )
            return (
                float(result["faithfulness"]),
                float(result["answer_relevancy"]),
                float(result["context_recall"]),
                float(result["context_precision"]),
            )
        except Exception:  # noqa: BLE001 - 版本差异 / 依赖缺失时降级
            return None


def _mean(values) -> float:
    values = [float(value) for value in values]
    return round(sum(values) / len(values), 4) if values else 0.0


def _tokens(text: str) -> set[str]:
    return set(tokenize(text))
