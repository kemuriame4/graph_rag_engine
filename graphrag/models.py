"""领域数据模型：解析 → 切片 → 检索 → 反思 → 评测 全链路契约。"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

BlockType = Literal["heading", "paragraph", "table", "figure", "formula", "list_item", "code"]


class Block(BaseModel):
    type: BlockType
    text: str
    level: int = 0
    page: int | None = None
    bbox: list[float] | None = None
    caption: str | None = None


class ParsedDocument(BaseModel):
    doc_id: str
    title: str
    source: str
    mime: str
    language: str = "unknown"
    blocks: list[Block] = Field(default_factory=list)
    markdown: str = ""
    page_count: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)


class ExtractedEntity(BaseModel):
    name: str
    normalized: str
    type: str = "term"
    aliases: list[str] = Field(default_factory=list)
    mentions: int = 1


class ExtractedRelation(BaseModel):
    source: str
    target: str
    relation: str = "related_to"
    weight: float = 1.0
    evidence: list[str] = Field(default_factory=list)


class ExtractionResult(BaseModel):
    entities: list[ExtractedEntity] = Field(default_factory=list)
    relations: list[ExtractedRelation] = Field(default_factory=list)


class Chunk(BaseModel):
    chunk_id: str
    doc_id: str
    doc_title: str
    text: str
    embedding_text: str = ""
    section_path: list[str] = Field(default_factory=list)
    chunk_index: int = 0
    token_count: int = 0
    page_start: int | None = None
    page_end: int | None = None
    summary: str = ""
    keywords: list[str] = Field(default_factory=list)
    entities: list[str] = Field(default_factory=list)
    prev_chunk_id: str | None = None
    next_chunk_id: str | None = None
    tenant: str = "default"

    @property
    def section(self) -> str:
        return " / ".join(self.section_path) if self.section_path else self.doc_title


class ChannelHit(BaseModel):
    chunk_id: str
    rank: int
    score: float


class RetrievedDoc(BaseModel):
    chunk_id: str
    doc_id: str
    doc_title: str
    text: str
    section_path: list[str] = Field(default_factory=list)
    page_start: int | None = None
    page_end: int | None = None
    score: float = 0.0
    dense_score: float | None = None
    sparse_score: float | None = None
    graph_score: float | None = None
    rrf_score: float = 0.0
    rerank_score: float | None = None
    channels: list[str] = Field(default_factory=list)
    graph_path: list[str] = Field(default_factory=list)
    highlights: list[str] = Field(default_factory=list)


class RetrievalTrace(BaseModel):
    channel_ranks: dict[str, list[ChannelHit]] = Field(default_factory=dict)
    fused_count: int = 0
    reranked_count: int = 0
    hyde_query: str | None = None
    graph_entities: list[str] = Field(default_factory=list)
    elapsed_ms: float = 0.0


class Citation(BaseModel):
    index: int
    chunk_id: str
    doc_id: str
    doc_title: str
    section: str
    snippet: str
    page: int | None = None
    score: float = 0.0


class SupportJudgement(BaseModel):
    score: float = 0.0
    supported_claims: list[str] = Field(default_factory=list)
    unsupported_claims: list[str] = Field(default_factory=list)


class ReflectionResult(BaseModel):
    is_rel: float = 0.0
    is_sup: float = 0.0
    is_use: float = 0.0
    relevance_pass: bool = False
    support_pass: bool = False
    usefulness_pass: bool = False
    unsupported_claims: list[str] = Field(default_factory=list)
    relevance_rounds: int = 0
    support_rounds: int = 0
    usefulness_rounds: int = 0


class RAGOptions(BaseModel):
    tenant: str = "default"
    top_k: int = 8
    force_retrieval: bool = False
    enable_hyde: bool = True
    enable_graph: bool = True
    max_hops: int = 2


class RAGResponse(BaseModel):
    trace_id: str
    query: str
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    query_type: Literal["direct", "single_hop", "multi_hop"] = "single_hop"
    need_retrieval: bool = True
    reflections: ReflectionResult = Field(default_factory=ReflectionResult)
    retrieval: RetrievalTrace = Field(default_factory=RetrievalTrace)
    retrieved: list[RetrievedDoc] = Field(default_factory=list)
    rewritten_queries: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    latency_ms: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0


class TraceStep(BaseModel):
    node: str
    detail: str = ""
    started_at: float = 0.0
    elapsed_ms: float = 0.0
    attributes: dict[str, Any] = Field(default_factory=dict)


class TraceRecord(BaseModel):
    trace_id: str
    kind: Literal["query", "ingest", "eval", "graph"] = "query"
    status: Literal["running", "success", "error"] = "running"
    started_at: float = 0.0
    finished_at: float | None = None
    latency_ms: float = 0.0
    steps: list[TraceStep] = Field(default_factory=list)
    metrics: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None


class EvalSample(BaseModel):
    question: str
    ground_truth: str
    tenant: str = "default"


class EvalRecord(BaseModel):
    question: str
    answer: str
    ground_truth: str
    contexts: list[str] = Field(default_factory=list)
    faithfulness: float = 0.0
    answer_relevance: float = 0.0
    context_recall: float = 0.0
    context_precision: float = 0.0
    latency_ms: float = 0.0
    trace_id: str = ""


class EvalReport(BaseModel):
    report_id: str
    created_at: float = 0.0
    backend: str = "internal"
    num_samples: int = 0
    faithfulness: float = 0.0
    answer_relevance: float = 0.0
    context_recall: float = 0.0
    context_precision: float = 0.0
    avg_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    records: list[EvalRecord] = Field(default_factory=list)


class JobRecord(BaseModel):
    job_id: str
    filename: str
    status: Literal["queued", "parsing", "chunking", "indexing", "success", "error"] = "queued"
    progress: float = 0.0
    doc_id: str | None = None
    title: str | None = None
    chunks: int = 0
    entities: int = 0
    error: str | None = None
    created_at: float = 0.0
    finished_at: float | None = None
    trace_id: str | None = None


class DocumentRecord(BaseModel):
    doc_id: str
    title: str
    source: str
    mime: str = ""
    language: str = "unknown"
    page_count: int = 0
    chunk_count: int = 0
    entity_count: int = 0
    tenant: str = "default"
    created_at: float = 0.0
    markdown: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class GraphNode(BaseModel):
    id: str
    label: str
    type: str
    degree: int = 0
    doc_ids: list[str] = Field(default_factory=list)


class GraphEdge(BaseModel):
    source: str
    target: str
    relation: str
    weight: float = 1.0


class SubGraph(BaseModel):
    seed_entities: list[str] = Field(default_factory=list)
    nodes: list[GraphNode] = Field(default_factory=list)
    edges: list[GraphEdge] = Field(default_factory=list)
