"""可观测性：全链路 Trace 采集、环形缓冲与持久化。"""

from __future__ import annotations

import json
import threading
import time
import uuid
from collections import deque
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from ..models import TraceRecord, TraceStep


class Tracer:
    def __init__(self, trace_dir: Path, max_records: int = 300) -> None:
        self.trace_dir = trace_dir
        self._records: deque[TraceRecord] = deque(maxlen=max_records)
        self._lock = threading.Lock()
        self._counters = {"query": 0, "ingest": 0, "eval": 0, "error": 0}
        self._token_totals = {"prompt": 0, "completion": 0}
        self._latencies: deque[float] = deque(maxlen=500)
        self.trace_dir.mkdir(parents=True, exist_ok=True)

    def start(self, kind: str = "query", attributes: dict[str, Any] | None = None) -> TraceRecord:
        record = TraceRecord(
            trace_id=uuid.uuid4().hex[:12],
            kind=kind,  # type: ignore[arg-type]
            started_at=time.time(),
            metrics={"attributes": attributes or {}},
        )
        with self._lock:
            self._records.append(record)
            self._counters[kind] = self._counters.get(kind, 0) + 1
        return record

    def add_step(
        self,
        record: TraceRecord,
        node: str,
        detail: str = "",
        elapsed_ms: float = 0.0,
        attributes: dict[str, Any] | None = None,
    ) -> None:
        record.steps.append(
            TraceStep(
                node=node,
                detail=detail,
                started_at=time.time(),
                elapsed_ms=round(elapsed_ms, 2),
                attributes=attributes or {},
            )
        )

    @contextmanager
    def span(self, record: TraceRecord, node: str, detail: str = "") -> Iterator[dict[str, Any]]:
        started = time.perf_counter()
        attributes: dict[str, Any] = {}
        try:
            yield attributes
        finally:
            self.add_step(record, node, detail, (time.perf_counter() - started) * 1000, attributes)

    def finish(
        self,
        record: TraceRecord,
        status: str = "success",
        metrics: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> None:
        record.status = status  # type: ignore[assignment]
        record.finished_at = time.time()
        record.latency_ms = round((record.finished_at - record.started_at) * 1000, 2)
        if metrics:
            record.metrics.update(metrics)
        if error:
            record.error = error
            with self._lock:
                self._counters["error"] += 1
        with self._lock:
            self._latencies.append(record.latency_ms)
            tokens = record.metrics.get("tokens", {})
            self._token_totals["prompt"] += int(tokens.get("prompt", 0))
            self._token_totals["completion"] += int(tokens.get("completion", 0))
        self._persist(record)

    def _persist(self, record: TraceRecord) -> None:
        try:
            day = time.strftime("%Y%m%d", time.localtime(record.started_at))
            directory = self.trace_dir / day
            directory.mkdir(parents=True, exist_ok=True)
            path = directory / f"{record.kind}_{record.trace_id}.jsonl"
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record.model_dump(), ensure_ascii=False) + "\n")
        except OSError:
            return

    def get(self, trace_id: str) -> TraceRecord | None:
        for record in self._records:
            if record.trace_id == trace_id:
                return record
        return None

    def list(self, kind: str | None = None, limit: int = 50) -> list[TraceRecord]:
        with self._lock:
            records = list(self._records)
        if kind:
            records = [record for record in records if record.kind == kind]
        return records[-limit:][::-1]

    def metrics(self) -> dict[str, Any]:
        with self._lock:
            latencies = sorted(self._latencies)
            counters = dict(self._counters)
            tokens = dict(self._token_totals)
        return {
            "counters": counters,
            "tokens": tokens,
            "latency": {
                "p50": _percentile(latencies, 50),
                "p95": _percentile(latencies, 95),
                "avg": round(sum(latencies) / len(latencies), 2) if latencies else 0.0,
            },
        }


def _percentile(values: list[float], percentile: int) -> float:
    if not values:
        return 0.0
    index = min(len(values) - 1, max(0, int(round(percentile / 100 * (len(values) - 1)))))
    return round(values[index], 2)
