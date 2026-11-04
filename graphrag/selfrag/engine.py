"""Self-RAG 自适应决策引擎。

状态机（与 LangGraph 语义对齐）：

    analyze ──┬─> direct_answer ──> END
              └─> retrieve ──┬─> generate ──> reflect_support ──┬─> generate(strict)
                             │                                └─> reflect_usefulness ──┬─> supplement
                             └─> rewrite ──> retrieve                                 └─> finalize ──> END
"""

from __future__ import annotations

import re
import time
import uuid

from ..config import Settings
from ..indexing import SearchIndex
from ..models import (
    Citation,
    RAGOptions,
    RAGResponse,
    ReflectionResult,
    RetrievalTrace,
    RetrievedDoc,
    SupportJudgement,
)
from ..observability import Tracer
from ..retrieval import HybridRetriever
from ..text import content_words, detect_language, keyword_overlap, normalize_text, truncate
from .llm import BaseLLM, LLMResponse, build_llm
from .state_graph import END, CompiledGraph, StateGraph

DIRECT_PATTERNS = {
    "greeting": re.compile(r"^(你好|您好|hi|hello|hey)[!！。.?\s]*$", re.IGNORECASE),
    "thanks": re.compile(r"^(谢谢|感谢你?|thanks|thank you)[!！。.?\s]*$", re.IGNORECASE),
    "identity": re.compile(r"^(你是谁|你能做什么|who are you|what can you do)[?？!！。.\s]*$", re.IGNORECASE),
}

MULTI_HOP_KEYWORDS = (
    "关系", "区别", "不同", "对比", "比较", "为什么", "流程", "依赖", "前置", "影响",
    "上下游", "同时", "分别", "以及", "compare", "relationship", "difference",
    "process", "depend", "impact", "why", "trade-off", "tradeoff",
)


class SelfRAGEngine:
    def __init__(
        self,
        index: SearchIndex,
        settings: Settings,
        retriever: HybridRetriever | None = None,
        llm: BaseLLM | None = None,
        tracer: Tracer | None = None,
    ) -> None:
        self.index = index
        self.settings = settings
        self.retriever = retriever or HybridRetriever(index, settings)
        self.llm = llm or build_llm(settings, index.embedder)
        self.tracer = tracer
        self.graph = self._build_graph()

    # ---------------- 对外接口 ----------------
    def run(
        self,
        question: str,
        options: RAGOptions | None = None,
        event_cb=None,
    ) -> RAGResponse:
        options = options or RAGOptions()
        started = time.perf_counter()
        record = self.tracer.start("query", {"question": question}) if self.tracer else None
        trace_id = record.trace_id if record else uuid.uuid4().hex[:12]
        state = self._initial_state(question, options, trace_id)
        latencies: list[tuple[str, float]] = []

        def on_step(node: str, _patch, elapsed_ms: float) -> None:
            latencies.append((node, elapsed_ms))
            detail = _describe(node, state)
            if self.tracer and record:
                self.tracer.add_step(record, node, detail, elapsed_ms)
            if event_cb:
                event_cb(
                    {
                        "type": "step",
                        "node": node,
                        "detail": detail,
                        "elapsed_ms": round(elapsed_ms, 2),
                        "query_type": state.get("query_type"),
                    }
                )

        try:
            final_state = self.graph.invoke(state, recursion_limit=24, on_step=on_step)
        except Exception as exc:  # noqa: BLE001
            if self.tracer and record:
                self.tracer.finish(record, status="error", error=str(exc))
            raise

        response = self._build_response(final_state, question, options, started)
        if self.tracer and record:
            self.tracer.finish(
                record,
                status="success",
                metrics={
                    "tokens": dict(final_state["tokens"]),
                    "query_type": final_state["query_type"],
                    "retrieved": len(final_state["docs"]),
                    "citation_count": len(response.citations),
                    "nodes": [node for node, _ in latencies],
                },
            )
        if event_cb:
            event_cb(
                {
                    "type": "final",
                    "answer": response.answer,
                    "trace_id": response.trace_id,
                    "latency_ms": response.latency_ms,
                }
            )
        return response

    # ---------------- 状态初始化 ----------------
    def _initial_state(self, question: str, options: RAGOptions, trace_id: str) -> dict:
        return {
            "question": normalize_text(question),
            "current_query": normalize_text(question),
            "query_type": "single_hop",
            "need_retrieval": True,
            "hyde_query": None,
            "docs": [],
            "retrieval_trace": RetrievalTrace(),
            "answer": "",
            "support": SupportJudgement(),
            "reflection": ReflectionResult(),
            "rewritten_queries": [],
            "relevance_rounds": 0,
            "support_rounds": 0,
            "usefulness_rounds": 0,
            "supplement_done": False,
            "regenerate": False,
            "warnings": [],
            "tokens": {"prompt": 0, "completion": 0},
            "options": options,
            "trace_id": trace_id,
        }

    # ---------------- 图定义 ----------------
    def _build_graph(self) -> CompiledGraph:
        graph = StateGraph()
        graph.add_node("analyze", self._node_analyze)
        graph.add_node("direct_answer", self._node_direct)
        graph.add_node("retrieve", self._node_retrieve)
        graph.add_node("rewrite", self._node_rewrite)
        graph.add_node("generate", self._node_generate)
        graph.add_node("reflect_support", self._node_reflect_support)
        graph.add_node("reflect_usefulness", self._node_reflect_usefulness)
        graph.add_node("supplement", self._node_supplement)
        graph.add_node("finalize", self._node_finalize)
        graph.set_entry_point("analyze")

        graph.add_conditional_edges(
            "analyze",
            lambda state: "direct" if state["query_type"] == "direct" else "retrieve",
            {"direct": "direct_answer", "retrieve": "retrieve"},
        )
        graph.add_conditional_edges(
            "direct_answer", lambda state: "end", {"end": END}
        )
        graph.add_conditional_edges(
            "retrieve",
            self._route_retrieve,
            {"generate": "generate", "rewrite": "rewrite"},
        )
        graph.add_conditional_edges(
            "rewrite", lambda state: "retrieve", {"retrieve": "retrieve"}
        )
        graph.add_conditional_edges(
            "generate", lambda state: "support", {"support": "reflect_support"}
        )
        graph.add_conditional_edges(
            "reflect_support",
            lambda state: "regenerate" if state["regenerate"] else "usefulness",
            {"regenerate": "generate", "usefulness": "reflect_usefulness"},
        )
        graph.add_conditional_edges(
            "reflect_usefulness",
            lambda state: "supplement" if state.get("supplement_needed") else "finalize",
            {"supplement": "supplement", "finalize": "finalize"},
        )
        graph.add_conditional_edges(
            "supplement", lambda state: "retrieve", {"retrieve": "retrieve"}
        )
        graph.add_conditional_edges("finalize", lambda state: "end", {"end": END})
        return graph.compile()

    # ---------------- 节点实现 ----------------
    def _node_analyze(self, state: dict) -> dict:
        question = state["question"]
        query_type = "single_hop"
        need_retrieval = True
        options: RAGOptions = state["options"]

        for kind, pattern in DIRECT_PATTERNS.items():
            if pattern.match(question):
                state["direct_kind"] = kind
                query_type = "direct"
                need_retrieval = False
                break

        if query_type != "direct":
            if any(word in question.lower() for word in MULTI_HOP_KEYWORDS):
                query_type = "multi_hop"
            else:
                linked = self.index.graph.link_entities(question, options.tenant, limit=3)
                if len(linked) >= 2:
                    query_type = "multi_hop"
            if not self.settings.adaptive_retrieval:
                need_retrieval = True

        state["query_type"] = query_type
        state["need_retrieval"] = need_retrieval
        return {"query_type": query_type, "need_retrieval": need_retrieval}

    def _node_direct(self, state: dict) -> dict:
        kind = state.get("direct_kind", "greeting")
        if kind == "greeting":
            answer = "你好，我是企业知识库助手。可以基于内部文档回答产品规格、政策条款与业务流程等问题，并给出来源引用。"
        elif kind == "thanks":
            answer = "不客气。如需继续核查其他条款或跨文档信息，随时提问。"
        else:
            answer = (
                "我基于企业文档的三路混合检索（Dense + BM25 + 知识图谱）与 Self-RAG 反思机制作答，"
                "答案中的 [n] 可直接溯源到原文切片。"
            )
        state["answer"] = answer
        state["docs"] = []
        state["reflection"] = ReflectionResult(
            is_rel=1.0, is_sup=1.0, is_use=1.0,
            relevance_pass=True, support_pass=True, usefulness_pass=True,
        )
        return {"answer": answer}

    def _node_retrieve(self, state: dict) -> dict:
        options: RAGOptions = state["options"]
        query = state["current_query"]
        hyde_query = state.get("hyde_query")

        if (
            not hyde_query
            and options.enable_hyde
            and self.settings.enable_hyde
            and (state["query_type"] == "multi_hop" or len(content_words(query)) < 3)
        ):
            response = self.llm.hyde(query)
            hyde_query = response.text
            state["hyde_query"] = hyde_query
            self._track_tokens(state, response)

        docs, trace = self.retriever.search(query, options, hyde_query=hyde_query)
        state["docs"] = docs
        state["retrieval_trace"] = trace

        reflection: ReflectionResult = state["reflection"]
        reflection.is_rel = self._relevance_score(query, docs)
        reflection.relevance_pass = reflection.is_rel >= self.settings.is_rel_threshold
        reflection.relevance_rounds = state["relevance_rounds"]
        return {"docs": docs, "retrieval_trace": trace}

    def _route_retrieve(self, state: dict) -> str:
        reflection: ReflectionResult = state["reflection"]
        if state["docs"] and reflection.relevance_pass:
            return "generate"
        if state["relevance_rounds"] < self.settings.max_reflection_loops:
            if not state["docs"]:
                state["warnings"].append("首轮检索为空，触发 Query 重写")
            return "rewrite"
        return "generate"

    def _node_rewrite(self, state: dict) -> dict:
        state["relevance_rounds"] += 1
        options: RAGOptions = state["options"]
        hints = self.index.graph.related_terms(state["question"], options.tenant)
        response = self.llm.rewrite(state["current_query"], hints)
        self._track_tokens(state, response)
        rewritten = normalize_text(response.text).strip()
        if rewritten == state["current_query"] or not rewritten:
            rewritten = f"{state['question']} 定义 参数 规格 流程"
        state["current_query"] = rewritten
        state["rewritten_queries"].append(rewritten)
        reflection: ReflectionResult = state["reflection"]
        reflection.relevance_rounds = state["relevance_rounds"]
        return {"current_query": rewritten}

    def _node_generate(self, state: dict) -> dict:
        docs: list[RetrievedDoc] = state["docs"]
        strict = state["support_rounds"] > 0
        response = self.llm.answer(state["question"], docs, strict=strict)
        self._track_tokens(state, response)
        state["answer"] = response.text
        return {"answer": response.text}

    def _node_reflect_support(self, state: dict) -> dict:
        docs = state["docs"]
        answer = state["answer"]
        judgement = self.llm.judge_support(state["question"], answer, docs)
        state["support"] = judgement
        reflection: ReflectionResult = state["reflection"]
        reflection.is_sup = judgement.score
        reflection.support_pass = judgement.score >= self.settings.is_sup_threshold
        reflection.support_rounds = state["support_rounds"]
        reflection.unsupported_claims = judgement.unsupported_claims
        state["regenerate"] = (
            not reflection.support_pass and state["support_rounds"] < 1 and bool(docs)
        )
        if state["regenerate"]:
            state["support_rounds"] += 1
        elif not reflection.support_pass:
            state["warnings"].append("部分表述未能在原文中直接找到依据，已标注为低置信内容")
        return {"support": judgement}

    def _node_reflect_usefulness(self, state: dict) -> dict:
        score = self.llm.judge_usefulness(state["question"], state["answer"])
        reflection: ReflectionResult = state["reflection"]
        reflection.is_use = round(score, 4)
        reflection.usefulness_pass = score >= self.settings.is_use_threshold
        reflection.usefulness_rounds = state["usefulness_rounds"]
        state["supplement_needed"] = (
            not reflection.usefulness_pass
            and not state["supplement_done"]
            and self.settings.max_reflection_loops > 0
        )
        if state["supplement_needed"]:
            state["warnings"].append("答案覆盖度不足，触发补充检索")
        return {}

    def _node_supplement(self, state: dict) -> dict:
        state["supplement_done"] = True
        state["usefulness_rounds"] += 1
        options: RAGOptions = state["options"]
        hints = self.index.graph.related_terms(state["question"], options.tenant)
        state["current_query"] = (
            f"{state['question']} {' '.join(hints[:3])} 补充 相关条款 例外".strip()
        )
        state["warnings"].append("已执行补充检索（Usefulness 未达标）")
        return {"current_query": state["current_query"]}

    def _node_finalize(self, state: dict) -> dict:
        state["citations"] = self._build_citations(state["answer"], state["docs"])
        return {"citations": state["citations"]}

    # ---------------- 辅助 ----------------
    def _relevance_score(self, query: str, docs: list[RetrievedDoc]) -> float:
        if not docs:
            return 0.0
        top = docs[:3]
        rerank = sum(doc.rerank_score or 0.0 for doc in top) / len(top)
        lexical = max(keyword_overlap(query, doc.text) for doc in docs[:5])
        return round(min(1.0, 0.65 * rerank + 0.35 * lexical), 4)

    def _build_citations(self, answer: str, docs: list[RetrievedDoc]) -> list[Citation]:
        cited = [int(value) for value in re.findall(r"\[(\d{1,2})\]", answer)]
        citations: list[Citation] = []
        seen: set[str] = set()
        for doc_index in cited:
            if not 1 <= doc_index <= len(docs):
                continue
            doc = docs[doc_index - 1]
            if doc.chunk_id in seen:
                continue
            seen.add(doc.chunk_id)
            citations.append(
                Citation(
                    index=doc_index,
                    chunk_id=doc.chunk_id,
                    doc_id=doc.doc_id,
                    doc_title=doc.doc_title,
                    section=" / ".join(doc.section_path),
                    snippet=truncate(doc.text, 260),
                    page=doc.page_start,
                    score=doc.rerank_score or doc.rrf_score,
                )
            )
        if not citations:
            for offset, doc in enumerate(docs[:3], start=1):
                citations.append(
                    Citation(
                        index=offset,
                        chunk_id=doc.chunk_id,
                        doc_id=doc.doc_id,
                        doc_title=doc.doc_title,
                        section=" / ".join(doc.section_path),
                        snippet=truncate(doc.text, 260),
                        page=doc.page_start,
                        score=doc.rerank_score or doc.rrf_score,
                    )
                )
        return citations

    def _build_response(
        self,
        state: dict,
        question: str,
        options: RAGOptions,
        started: float,
    ) -> RAGResponse:
        answer = state["answer"] or "未能生成回答。"
        if not state["docs"] and state["query_type"] != "direct":
            state["warnings"].append("知识库未命中相关切片，回答缺少文档依据")
        reflection: ReflectionResult = state["reflection"]
        citations = state.get("citations") or self._build_citations(answer, state["docs"])
        return RAGResponse(
            trace_id=state["trace_id"],
            query=question,
            answer=answer,
            citations=citations,
            query_type=state["query_type"],
            need_retrieval=state["need_retrieval"],
            reflections=reflection,
            retrieval=state["retrieval_trace"],
            retrieved=state["docs"],
            rewritten_queries=list(state["rewritten_queries"]),
            warnings=list(dict.fromkeys(state["warnings"])),
            latency_ms=round((time.perf_counter() - started) * 1000, 2),
            prompt_tokens=state["tokens"]["prompt"],
            completion_tokens=state["tokens"]["completion"],
        )

    @staticmethod
    def _track_tokens(state: dict, response: LLMResponse) -> None:
        state["tokens"]["prompt"] += response.prompt_tokens
        state["tokens"]["completion"] += response.completion_tokens


CompiledStateGraph = CompiledGraph  # 兼容旧类型别名


def _describe(node: str, state: dict) -> str:
    if node == "analyze":
        return f"query_type={state['query_type']} need_retrieval={state['need_retrieval']}"
    if node == "retrieve":
        docs = state["docs"]
        return f"召回 {len(docs)} 条切片，Is-Rel={state['reflection'].is_rel:.2f}"
    if node == "rewrite":
        return f"重写为：{state['current_query']}"
    if node == "generate":
        return f"生成答案 {len(state['answer'])} 字"
    if node == "reflect_support":
        return f"Is-Sup={state['reflection'].is_sup:.2f} 不支持句数={len(state['support'].unsupported_claims)}"
    if node == "reflect_usefulness":
        return f"Is-Use={state['reflection'].is_use:.2f}"
    if node == "supplement":
        return "补充检索"
    if node == "finalize":
        return f"输出引用 {len(state.get('citations', []))} 条"
    if node == "direct_answer":
        return "自适应判定：无需检索，直接回答"
    return ""


def build_engine(index: SearchIndex, settings: Settings, tracer: Tracer | None = None) -> SelfRAGEngine:
    return SelfRAGEngine(index, settings, tracer=tracer)
