"""全局配置：环境变量驱动，全部带安全默认值，缺省即可离线运行。"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _env(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


def _env_int(name: str, default: int) -> int:
    try:
        return int(_env(name, str(default)))
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(_env(name, str(default)))
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = _env(name, "").lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on"}


class Settings(BaseModel):
    """引擎配置。所有外部依赖均可关闭，自动降级为本地确定性实现。"""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    app_name: str = "Adaptive Graph-RAG Knowledge Engine"
    version: str = "0.1.0"
    env: str = "dev"
    data_dir: Path = Field(default_factory=lambda: PROJECT_ROOT / "data")
    default_tenant: str = "default"

    # LLM / Embedding
    llm_api_key: str = ""
    llm_base_url: str = "https://api.openai.com/v1"
    llm_chat_model: str = "gpt-4o-mini"
    llm_embed_model: str = "text-embedding-3-small"
    embedding_backend: Literal["auto", "hashing", "openai", "sentence-transformers"] = "auto"
    embedding_dim: int = 256
    st_model: str = "BAAI/bge-m3"
    llm_timeout_seconds: float = 30.0

    # Vector / Sparse / Graph
    qdrant_url: str = ""
    qdrant_api_key: str = ""
    qdrant_collection: str = "graphrag_chunks"
    es_url: str = ""
    es_index: str = "graphrag_chunks"
    es_api_key: str = ""

    # Retrieval
    top_k_dense: int = 20
    top_k_sparse: int = 20
    top_k_graph: int = 20
    rrf_k: int = 60
    rerank_top_n: int = 8
    reranker_backend: Literal["auto", "heuristic", "cross-encoder"] = "auto"
    rerank_model: str = "BAAI/bge-reranker-v2-m3"
    graph_max_hops: int = 2

    # Chunking
    chunk_max_tokens: int = 450
    chunk_overlap_sentences: int = 1

    # Self-RAG
    adaptive_retrieval: bool = True
    enable_hyde: bool = True
    max_reflection_loops: int = 2
    is_rel_threshold: float = 0.22
    is_sup_threshold: float = 0.72
    is_use_threshold: float = 0.60

    # Ingestion
    max_ingest_workers: int = 4
    job_backend: Literal["thread", "celery"] = "thread"
    redis_url: str = "redis://localhost:6379/0"
    max_upload_mb: int = 200

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            env=_env("GRAGRAG_ENV", "dev"),
            data_dir=Path(_env("DATA_DIR", str(PROJECT_ROOT / "data"))).expanduser(),
            default_tenant=_env("DEFAULT_TENANT", "default"),
            llm_api_key=_env("LLM_API_KEY"),
            llm_base_url=_env("LLM_BASE_URL", "https://api.openai.com/v1"),
            llm_chat_model=_env("LLM_CHAT_MODEL", "gpt-4o-mini"),
            llm_embed_model=_env("LLM_EMBED_MODEL", "text-embedding-3-small"),
            embedding_backend=_env("EMBEDDING_BACKEND", "auto"),  # type: ignore[arg-type]
            embedding_dim=_env_int("EMBEDDING_DIM", 256),
            st_model=_env("ST_MODEL", "BAAI/bge-m3"),
            llm_timeout_seconds=_env_float("LLM_TIMEOUT_SECONDS", 30.0),
            qdrant_url=_env("QDRANT_URL"),
            qdrant_api_key=_env("QDRANT_API_KEY"),
            qdrant_collection=_env("QDRANT_COLLECTION", "graphrag_chunks"),
            es_url=_env("ES_URL"),
            es_index=_env("ES_INDEX", "graphrag_chunks"),
            es_api_key=_env("ES_API_KEY"),
            top_k_dense=_env_int("TOP_K_DENSE", 20),
            top_k_sparse=_env_int("TOP_K_SPARSE", 20),
            top_k_graph=_env_int("TOP_K_GRAPH", 20),
            rrf_k=_env_int("RRF_K", 60),
            rerank_top_n=_env_int("RERANK_TOP_N", 8),
            reranker_backend=_env("RERANKER_BACKEND", "auto"),  # type: ignore[arg-type]
            rerank_model=_env("RERANK_MODEL", "BAAI/bge-reranker-v2-m3"),
            graph_max_hops=_env_int("GRAPH_MAX_HOPS", 2),
            chunk_max_tokens=_env_int("CHUNK_MAX_TOKENS", 450),
            chunk_overlap_sentences=_env_int("CHUNK_OVERLAP_SENTENCES", 1),
            adaptive_retrieval=_env_bool("ADAPTIVE_RETRIEVAL", True),
            enable_hyde=_env_bool("ENABLE_HYDE", True),
            max_reflection_loops=_env_int("MAX_REFLECTION_LOOPS", 2),
            is_rel_threshold=_env_float("IS_REL_THRESHOLD", 0.22),
            is_sup_threshold=_env_float("IS_SUP_THRESHOLD", 0.72),
            is_use_threshold=_env_float("IS_USE_THRESHOLD", 0.60),
            max_ingest_workers=_env_int("MAX_INGEST_WORKERS", 4),
            job_backend=_env("JOB_BACKEND", "thread"),  # type: ignore[arg-type]
            redis_url=_env("REDIS_URL", "redis://localhost:6379/0"),
            max_upload_mb=_env_int("MAX_UPLOAD_MB", 200),
        )

    @property
    def index_dir(self) -> Path:
        return self.data_dir / "index"

    @property
    def upload_dir(self) -> Path:
        return self.data_dir / "uploads"

    @property
    def trace_dir(self) -> Path:
        return self.data_dir / "traces"

    @property
    def eval_dir(self) -> Path:
        return self.data_dir / "eval"

    @property
    def llm_enabled(self) -> bool:
        return bool(self.llm_api_key)

    def ensure_dirs(self) -> None:
        for path in (self.data_dir, self.index_dir, self.upload_dir, self.trace_dir, self.eval_dir):
            path.mkdir(parents=True, exist_ok=True)


settings = Settings.from_env()
