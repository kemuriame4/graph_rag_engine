"""FastAPI 接口层。"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

from fastapi import (
    APIRouter,
    BackgroundTasks,
    FastAPI,
    File,
    Form,
    HTTPException,
    Query,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from graphrag.bootstrap import Container, get_container
from graphrag.models import EvalSample, RAGOptions

STATIC_DIR = Path(__file__).parent / "static"


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, description="用户问题")
    tenant: str | None = None
    top_k: int = Field(default=8, ge=1, le=30)
    force_retrieval: bool = False
    enable_hyde: bool = True
    enable_graph: bool = True
    max_hops: int = Field(default=2, ge=1, le=3)


class EvalRequest(BaseModel):
    dataset_path: str | None = None
    top_k: int = Field(default=8, ge=1, le=30)
    tenant: str | None = None
    samples: list[EvalSample] | None = None


def create_app(container: Container | None = None) -> FastAPI:
    app = FastAPI(
        title="企业级自适应 Graph-RAG 知识引擎",
        version="0.1.0",
        description="多源异构文档解析 + Dense/BM25/Graph 混合检索 + Self-RAG 反思 + Ragas 评测",
    )
    state: dict = {"container": container}

    def svc() -> Container:
        if state["container"] is None:
            state["container"] = get_container()
        return state["container"]

    @app.get("/", response_class=HTMLResponse)
    async def dashboard() -> HTMLResponse:
        index_path = STATIC_DIR / "index.html"
        if not index_path.exists():
            return HTMLResponse("<h1>Graph-RAG Engine</h1><p>前端看板未安装</p>")
        return HTMLResponse(index_path.read_text(encoding="utf-8"))

    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    router = APIRouter(prefix="/api/v1")

    # ---------------- 健康 / 统计 ----------------
    @router.get("/health")
    async def health() -> dict:
        service = svc()
        return {
            "status": "ok",
            "version": service.settings.version,
            "llm": service.llm.name,
            "embedding": service.index.embedder.name,
            "vector": service.index.vector_store.name,
            "sparse": service.index.bm25.name,
            "judge": service.evaluator.backend,
        }

    @router.get("/stats")
    async def stats() -> dict:
        service = svc()
        return {
            **service.index.stats(),
            "jobs": len(service.ingestion.jobs),
            "traces": len(service.tracer.list(limit=1000)),
        }

    # ---------------- 文档入库 ----------------
    @router.post("/documents")
    async def upload_document(
        file: UploadFile = File(...),
        tenant: str = Form(default="default"),
    ) -> dict:
        service = svc()
        data = await file.read()
        max_bytes = service.settings.max_upload_mb * 1024 * 1024
        if len(data) > max_bytes:
            raise HTTPException(
                status_code=413,
                detail=f"文件超过 {service.settings.max_upload_mb}MB 限制",
            )
        filename = file.filename or "upload.txt"
        job = service.ingestion.submit_bytes(data, filename, tenant)
        return job.model_dump()

    @router.get("/documents")
    async def list_documents() -> list[dict]:
        service = svc()
        documents = []
        for record in service.index.get_documents():
            payload = record.model_dump()
            payload.pop("markdown", None)
            documents.append(payload)
        return documents

    @router.get("/documents/{doc_id}")
    async def get_document(doc_id: str) -> dict:
        service = svc()
        record = service.index.get_document(doc_id)
        if record is None:
            raise HTTPException(status_code=404, detail="文档不存在")
        payload = record.model_dump()
        payload["chunk_count"] = sum(
            1 for chunk in service.index.chunks.values() if chunk.doc_id == doc_id
        )
        return payload

    @router.get("/documents/{doc_id}/chunks")
    async def get_document_chunks(doc_id: str) -> list[dict]:
        service = svc()
        if service.index.get_document(doc_id) is None:
            raise HTTPException(status_code=404, detail="文档不存在")
        return [
            chunk.model_dump()
            for chunk in sorted(
                service.index.chunks.values(), key=lambda item: item.chunk_index
            )
            if chunk.doc_id == doc_id
        ]

    @router.get("/documents/{doc_id}/file")
    async def download_document(doc_id: str):
        service = svc()
        for job in service.ingestion.list_jobs(limit=500):
            if job.doc_id != doc_id:
                continue
            path = service.settings.upload_dir / job.job_id / job.filename
            if path.exists():
                return FileResponse(path, filename=job.filename)
        raise HTTPException(status_code=404, detail="原始文件不存在")

    @router.delete("/documents/{doc_id}")
    async def delete_document(doc_id: str) -> dict:
        service = svc()
        if service.index.get_document(doc_id) is None:
            raise HTTPException(status_code=404, detail="文档不存在")
        service.index.remove_document(doc_id)
        service.index.persist()
        return {"deleted": doc_id}

    @router.get("/jobs")
    async def list_jobs(limit: int = Query(default=30, ge=1, le=200)) -> list[dict]:
        return [job.model_dump() for job in svc().ingestion.list_jobs(limit)]

    @router.get("/jobs/{job_id}")
    async def get_job(job_id: str) -> dict:
        job = svc().ingestion.get_job(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="任务不存在")
        return job.model_dump()

    # ---------------- 查询 ----------------
    @router.post("/query")
    async def query(payload: QueryRequest) -> dict:
        service = svc()
        options = RAGOptions(
            tenant=payload.tenant or service.settings.default_tenant,
            top_k=payload.top_k,
            force_retrieval=payload.force_retrieval,
            enable_hyde=payload.enable_hyde,
            enable_graph=payload.enable_graph,
            max_hops=payload.max_hops,
        )
        response = await asyncio.to_thread(
            service.engine.run, payload.question, options
        )
        return response.model_dump()

    @app.websocket("/api/v1/ws/query")
    async def query_stream(websocket: WebSocket) -> None:
        await websocket.accept()
        service = svc()
        try:
            while True:
                raw = await websocket.receive_text()
                payload = json.loads(raw)
                question = payload.get("question", "").strip()
                if not question:
                    await websocket.send_json(
                        {"type": "error", "detail": "question 不能为空"}
                    )
                    continue
                options = RAGOptions(
                    tenant=payload.get("tenant") or service.settings.default_tenant,
                    top_k=int(payload.get("top_k", 8)),
                    force_retrieval=bool(payload.get("force_retrieval", False)),
                    enable_hyde=bool(payload.get("enable_hyde", True)),
                    enable_graph=bool(payload.get("enable_graph", True)),
                    max_hops=int(payload.get("max_hops", 2)),
                )
                queue: asyncio.Queue = asyncio.Queue()
                loop = asyncio.get_running_loop()

                def event_cb(event: dict) -> None:
                    loop.call_soon_threadsafe(queue.put_nowait, event)

                task = asyncio.create_task(
                    asyncio.to_thread(service.engine.run, question, options, event_cb)
                )
                while True:
                    try:
                        event = await asyncio.wait_for(queue.get(), timeout=0.2)
                        await websocket.send_json(event)
                    except asyncio.TimeoutError:
                        if task.done():
                            break
                response = await task
                await websocket.send_json({"type": "result", **response.model_dump()})
        except WebSocketDisconnect:
            return
        except Exception as exc:  # noqa: BLE001
            try:
                await websocket.send_json({"type": "error", "detail": str(exc)})
            except Exception:  # noqa: BLE001
                return

    # ---------------- 知识图谱 ----------------
    @router.get("/graph")
    async def graph(
        entity: str = Query(default="", description="实体名，多个用逗号分隔"),
        hops: int = Query(default=2, ge=1, le=3),
        tenant: str | None = None,
    ) -> dict:
        service = svc()
        names = [name.strip() for name in entity.split(",") if name.strip()]
        subgraph = service.index.graph.subgraph(
            names, hops=hops, tenant=tenant or service.settings.default_tenant
        )
        return subgraph.model_dump()

    @router.get("/graph/stats")
    async def graph_stats() -> dict:
        return svc().index.graph.stats()

    # ---------------- 可观测性 ----------------
    @router.get("/traces")
    async def list_traces(
        kind: str | None = None,
        limit: int = Query(default=30, ge=1, le=200),
    ) -> list[dict]:
        return [
            record.model_dump() for record in svc().tracer.list(kind=kind, limit=limit)
        ]

    @router.get("/traces/{trace_id}")
    async def get_trace(trace_id: str) -> dict:
        record = svc().tracer.get(trace_id)
        if record is None:
            raise HTTPException(status_code=404, detail="Trace 不存在")
        return record.model_dump()

    @router.get("/metrics")
    async def metrics() -> dict:
        service = svc()
        return {**service.tracer.metrics(), "index": service.index.stats()}

    # ---------------- 评测 ----------------
    @router.post("/eval/run")
    async def run_eval(payload: EvalRequest) -> dict:
        service = svc()
        if payload.samples:
            samples = payload.samples
        else:
            dataset = (
                Path(payload.dataset_path)
                if payload.dataset_path
                else (
                    Path(__file__).resolve().parents[1]
                    / "samples"
                    / "eval_dataset.jsonl"
                )
            )
            if not dataset.exists():
                raise HTTPException(status_code=404, detail=f"评测集不存在：{dataset}")
            samples = service.evaluator.load_samples(dataset)
        report = await asyncio.to_thread(
            service.evaluator.evaluate,
            samples,
            payload.top_k,
            payload.tenant or service.settings.default_tenant,
        )
        summary = report.model_dump()
        summary.pop("records", None)
        return summary

    @router.get("/eval/reports")
    async def list_reports(limit: int = Query(default=20, ge=1, le=100)) -> list[dict]:
        return [report.model_dump() for report in svc().evaluator.list_reports(limit)]

    @router.get("/eval/reports/{report_id}")
    async def get_report(report_id: str) -> dict:
        report = svc().evaluator.get_report(report_id)
        if report is None:
            raise HTTPException(status_code=404, detail="评测报告不存在")
        return report.model_dump()

    # ---------------- 运维 ----------------
    @router.post("/reindex")
    async def reindex() -> dict:
        service = svc()
        return await asyncio.to_thread(service.index.rebuild)

    @router.post("/samples/load")
    async def load_samples() -> dict:
        """一键导入内置示例文档，便于快速体验（幂等：同 doc_id 覆盖）。"""
        service = svc()
        sample_dir = Path(__file__).resolve().parents[1] / "samples"
        loaded = []
        for path in sorted(sample_dir.glob("*")):
            if path.suffix.lower() not in {".md", ".txt", ".pdf", ".docx"}:
                continue
            job = service.ingestion.submit_bytes(
                path.read_bytes(), path.name, "default"
            )
            loaded.append({"filename": path.name, "job_id": job.job_id})
        return {"submitted": loaded}

    app.include_router(router)
    return app


app = create_app()
