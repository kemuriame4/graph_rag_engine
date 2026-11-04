"""文档入库服务：解析 → 语义切片 → 三路索引，支持线程池与 Celery 两种后端。"""

from __future__ import annotations

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .chunking import SemanticChunker
from .config import Settings
from .indexing import SearchIndex
from .models import JobRecord, ParsedDocument
from .observability import Tracer
from .parsing import DocumentParser, UnsupportedDocumentError
from .text import sha1_id


class IngestionService:
    def __init__(
        self,
        settings: Settings,
        index: SearchIndex,
        tracer: Tracer | None = None,
        parser: DocumentParser | None = None,
    ) -> None:
        self.settings = settings
        self.index = index
        self.tracer = tracer
        self.parser = parser or DocumentParser()
        self.chunker = SemanticChunker(settings)
        self.executor = ThreadPoolExecutor(max_workers=settings.max_ingest_workers)
        self.jobs: dict[str, JobRecord] = {}
        self._lock = threading.Lock()
        self._load_jobs()

    def submit_bytes(self, data: bytes, filename: str, tenant: str | None = None) -> JobRecord:
        tenant = tenant or self.settings.default_tenant
        job = JobRecord(
            job_id=sha1_id(filename, str(time.time()), size=12),
            filename=filename,
            status="queued",
            created_at=time.time(),
        )
        with self._lock:
            self.jobs[job.job_id] = job
        save_dir = self.settings.upload_dir / job.job_id
        save_dir.mkdir(parents=True, exist_ok=True)
        target = save_dir / filename
        target.write_bytes(data)

        if self.settings.job_backend == "celery":
            self._submit_celery(job, str(target), tenant)
        else:
            self.executor.submit(self._run, job.job_id, str(target), tenant, filename)
        return job

    def _submit_celery(self, job: JobRecord, path: str, tenant: str) -> None:
        try:
            from .tasks.celery_app import ingest_document  # type: ignore[import-not-found]

            ingest_document.delay(job.job_id, path, tenant, job.filename)
        except Exception:  # noqa: BLE001 - Celery 不可用时退回线程池
            self.executor.submit(self._run, job.job_id, path, tenant, job.filename)

    def process_bytes_sync(self, data: bytes, filename: str, tenant: str | None = None) -> tuple[JobRecord, str]:
        """同步入库（CLI / 测试用），返回 (job, doc_id)。"""
        tenant = tenant or self.settings.default_tenant
        job = JobRecord(
            job_id=sha1_id(filename, str(time.time()), size=12),
            filename=filename,
            status="queued",
            created_at=time.time(),
        )
        with self._lock:
            self.jobs[job.job_id] = job
        save_dir = self.settings.upload_dir / job.job_id
        save_dir.mkdir(parents=True, exist_ok=True)
        target = save_dir / filename
        target.write_bytes(data)
        self._run(job.job_id, str(target), tenant, filename)
        completed = self.jobs[job.job_id]
        if completed.status == "error":
            raise RuntimeError(completed.error or "入库失败")
        return completed, completed.doc_id or ""

    def get_job(self, job_id: str) -> JobRecord | None:
        with self._lock:
            return self.jobs.get(job_id)

    def list_jobs(self, limit: int = 50) -> list[JobRecord]:
        with self._lock:
            jobs = sorted(self.jobs.values(), key=lambda job: job.created_at, reverse=True)
        return jobs[:limit]

    def _run(self, job_id: str, path: str, tenant: str, filename: str) -> None:
        record = self.tracer.start("ingest", {"filename": filename}) if self.tracer else None
        stage_started = time.perf_counter()
        try:
            self._update(job_id, status="parsing", progress=0.15)
            parsed: ParsedDocument = self.parser.parse_bytes(
                Path(path).read_bytes(), filename
            )
            if record and self.tracer:
                self.tracer.add_step(record, "parse", f"{len(parsed.blocks)} blocks", (time.perf_counter() - stage_started) * 1000)

            self._update(job_id, status="chunking", progress=0.4, doc_id=parsed.doc_id, title=parsed.title)
            chunks = self.chunker.chunk(parsed, tenant=tenant)
            if record and self.tracer:
                self.tracer.add_step(record, "chunk", f"{len(chunks)} chunks", (time.perf_counter() - stage_started) * 1000)

            self._update(job_id, status="indexing", progress=0.7)
            stats = self.index.index_document(parsed, chunks)
            if record and self.tracer:
                self.tracer.add_step(record, "index", json.dumps(stats, ensure_ascii=False), (time.perf_counter() - stage_started) * 1000)

            self._update(
                job_id,
                status="success",
                progress=1.0,
                doc_id=parsed.doc_id,
                title=parsed.title,
                chunks=stats["chunks"],
                entities=stats["entities"],
            )
            if record and self.tracer:
                self.tracer.finish(
                    record,
                    metrics={"stats": stats, "tokens": {"prompt": 0, "completion": 0}},
                )
                self._update(job_id, trace_id=record.trace_id)
        except Exception as exc:  # noqa: BLE001
            message = str(exc)
            if isinstance(exc, UnsupportedDocumentError):
                message = f"解析依赖缺失：{message}"
            self._update(job_id, status="error", error=message)
            if record and self.tracer:
                self.tracer.finish(record, status="error", error=message)

    def _update(self, job_id: str, **fields) -> None:
        with self._lock:
            job = self.jobs.get(job_id)
            if job is None:
                return
            for key, value in fields.items():
                setattr(job, key, value)
            if job.status in {"success", "error"}:
                job.finished_at = time.time()
            self._persist_jobs()

    def _persist_jobs(self) -> None:
        try:
            path = self.settings.data_dir / "jobs.json"
            path.write_text(
                json.dumps([job.model_dump() for job in self.jobs.values()], ensure_ascii=False),
                encoding="utf-8",
            )
        except OSError:
            return

    def _load_jobs(self) -> None:
        path = self.settings.data_dir / "jobs.json"
        if not path.exists():
            return
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            for item in payload:
                job = JobRecord.model_validate(item)
                self.jobs[job.job_id] = job
        except (json.JSONDecodeError, ValueError, OSError):
            return
