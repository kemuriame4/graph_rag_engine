"""FastAPI 接口：健康检查、文档入库、问答与图谱。"""

from __future__ import annotations

import time

from fastapi.testclient import TestClient

from app.main import create_app
from graphrag.bootstrap import Container


def _client(container: Container) -> TestClient:
    return TestClient(create_app(container))


def test_health_and_stats(container: Container) -> None:
    client = _client(container)
    health = client.get("/api/v1/health").json()
    assert health["status"] == "ok"
    stats = client.get("/api/v1/stats").json()
    assert "chunks" in stats


def test_upload_query_and_graph(container: Container) -> None:
    client = _client(container)
    content = (
        "# Battery-X1\n\n## 电气参数\n\nCycle life is 3000 cycles. "
        "Charge time is 1.8 hours.\n"
    )
    response = client.post(
        "/api/v1/documents",
        files={"file": ("battery.md", content.encode("utf-8"), "text/markdown")},
        data={"tenant": "default"},
    )
    assert response.status_code == 200
    job_id = response.json()["job_id"]

    deadline = time.time() + 10
    while time.time() < deadline:
        job = client.get(f"/api/v1/jobs/{job_id}").json()
        if job["status"] in {"success", "error"}:
            break
        time.sleep(0.1)
    assert job["status"] == "success", job.get("error")

    answer = client.post(
        "/api/v1/query",
        json={"question": "Battery-X1 的循环寿命是多少？", "top_k": 5},
    ).json()
    assert answer["answer"]
    assert answer["citations"]

    graph = client.get("/api/v1/graph", params={"entity": "Battery-X1", "hops": 1}).json()
    assert "nodes" in graph
