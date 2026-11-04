"""Celery 入口（可选）：docker-compose 中 worker 服务使用。"""

from __future__ import annotations

from celery import Celery  # type: ignore[import-not-found]

from ..config import settings
from ..bootstrap import get_container

celery_app = Celery(
    "graphrag",
    broker=settings.redis_url,
    backend=settings.redis_url,
)
celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="Asia/Shanghai",
    task_track_started=True,
)


@celery_app.task(name="graphrag.ingest_document")
def ingest_document(job_id: str, path: str, tenant: str, filename: str) -> dict:
    service = get_container().ingestion
    service._run(job_id, path, tenant, filename)  # noqa: SLF001 - 复用统一入库流程
    job = service.get_job(job_id)
    return job.model_dump() if job else {}
