"""LLM 层：OpenAI 兼容接口 + 离线启发式实现。

离线实现不是"假数据"：它以抽取式摘要 + 词法/向量校验完成生成、忠实度判定、
Query 重写与 HyDE，保证在无 API Key 时 Self-RAG 的每个反思节点都能真实运行。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

import httpx

from ..config import Settings
from ..indexing.embeddings import BaseEmbedder, cosine
from ..models import RetrievedDoc, SupportJudgement
from ..text import (
    char_coverage,
    clamp,
    content_words,
    coverage,
    detect_language,
    expand_bilingual,
    keyword_overlap,
    normalize_text,
    split_sentences,
    tokenize,
)

NUMERIC_INTENT_RE = re.compile(r"多少|几[个项种]|数值|参数|规格|how many|how much|what is the|specification", re.I)
DIGIT_RE = re.compile(r"\d")


@dataclass
class LLMResponse:
    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class BaseLLM:
    name = "base"

    def answer(self, question: str, docs: list[RetrievedDoc], strict: bool = False) -> LLMResponse:
        raise NotImplementedError

    def rewrite(self, question: str, hints: list[str] | None = None) -> LLMResponse:
        raise NotImplementedError

    def hyde(self, question: str) -> LLMResponse:
        raise NotImplementedError

    def judge_support(self, question: str, answer: str, docs: list[RetrievedDoc]) -> SupportJudgement:
        raise NotImplementedError

    def judge_usefulness(self, question: str, answer: str) -> float:
        raise NotImplementedError


class HeuristicLLM(BaseLLM):
    name = "heuristic-offline"

    def __init__(self, embedder: BaseEmbedder) -> None:
        self.embedder = embedder

    # ---------------- 生成 ----------------
    def answer(self, question: str, docs: list[RetrievedDoc], strict: bool = False) -> LLMResponse:
        if not docs:
            return LLMResponse("知识库中未检索到与该问题相关的资料，暂时无法给出有依据的回答。")

        language = detect_language(question)
        bilingual = expand_bilingual(question)
        enriched_question = f"{question} {bilingual}".strip()
        question_tokens = set(content_words(enriched_question))
        numeric_intent = bool(NUMERIC_INTENT_RE.search(question))
        candidates: list[tuple[float, int, str]] = []
        for doc_index, doc in enumerate(docs, start=1):
            for sentence in split_sentences(doc.text):
                if len(sentence) < 8:
                    continue
                tokens = set(tokenize(sentence))
                overlap = len(question_tokens & tokens) / max(len(question_tokens), 1)
                specific = keyword_overlap(enriched_question, sentence)
                chars = char_coverage(question, sentence)
                cross_lingual = coverage(bilingual, sentence) if bilingual else 0.0
                score = (
                    0.40 * overlap
                    + 0.20 * specific
                    + 0.25 * chars
                    + 0.15 * cross_lingual
                )
                if NUMERIC_INTENT_RE.search(sentence) or (numeric_intent and DIGIT_RE.search(sentence)):
                    score += 0.12
                if doc_index <= 3:
                    score += 0.04
                candidates.append((score, doc_index, sentence))

        candidates.sort(key=lambda item: item[0], reverse=True)
        best = candidates[0][0] if candidates else 0.0
        cutoff = max(0.22, best * 0.62)
        chosen: list[tuple[float, int, str]] = []
        total_chars = 0
        for candidate in candidates:
            if candidate[0] < cutoff or total_chars > 520:
                continue
            if any(keyword_overlap(candidate[2], existing[2]) > 0.72 for existing in chosen):
                continue
            chosen.append(candidate)
            total_chars += len(candidate[2])
            if len(chosen) >= 4:
                break

        if not chosen:
            snippet = docs[0].text[:260].replace("\n", " ")
            chosen = [(0.0, 1, snippet)]

        chosen.sort(key=lambda item: item[1])
        if language == "en":
            lines = ["Based on the retrieved knowledge base:"]
            for _, doc_index, sentence in chosen:
                lines.append(f"- {sentence} [{doc_index}]")
        else:
            lines = ["根据知识库中的资料："]
            for _, doc_index, sentence in chosen:
                lines.append(f"- {sentence} [{doc_index}]")
        if strict:
            lines.append("\n以上结论仅基于检索到的原文片段，未做额外推断。")
        text = "\n".join(lines)
        return LLMResponse(
            text=text,
            prompt_tokens=sum(len(tokenize(doc.text)) for doc in docs) + len(tokenize(question)),
            completion_tokens=len(tokenize(text)),
        )

    # ---------------- Query 重写 / HyDE ----------------
    def rewrite(self, question: str, hints: list[str] | None = None) -> LLMResponse:
        cleaned = re.sub(
            r"(帮我|请问|我想知道|麻烦|请|能不能|可以|告诉我|一下|关于)", "", question
        ).strip(" ，,。？?！!")
        expansion: list[str] = []
        if len(content_words(cleaned)) < 3:
            expansion.append("定义 参数 规格 流程")
        if hints:
            expansion.extend(hints[:3])
        if not expansion:
            expansion = content_words(cleaned)[:3]
        rewritten = f"{cleaned} {' '.join(dict.fromkeys(expansion))}".strip()
        return LLMResponse(rewritten or question)

    def hyde(self, question: str) -> LLMResponse:
        language = detect_language(question)
        if language == "en":
            text = (
                f"A reference document about {question} would define the key entities, "
                "list applicable specifications and constraints, and describe the required process "
                "and related policies in detail."
            )
        else:
            text = (
                f"关于「{question}」的标准资料通常包含：核心概念定义、适用对象与范围、"
                "关键参数与规格指标、执行流程与前置条件、相关政策条款及例外情形。"
            )
        return LLMResponse(text)

    # ---------------- 反思判定 ----------------
    def judge_support(self, question: str, answer: str, docs: list[RetrievedDoc]) -> SupportJudgement:
        sentences = [s for s in split_sentences(answer) if len(s) > 8]
        if not sentences:
            return SupportJudgement(score=0.0, unsupported_claims=["empty-answer"])
        context = "\n".join(doc.text for doc in docs)
        context_sentences = split_sentences(context)
        context_vectors = self.embedder.embed(context_sentences) if context_sentences else None

        supported: list[str] = []
        unsupported: list[str] = []
        scores: list[float] = []
        for sentence in sentences:
            cleaned = re.sub(r"^[-•\d.\s]+", "", sentence).strip()
            if not cleaned or cleaned.startswith(("根据知识库", "以上结论", "Based on")):
                scores.append(1.0)
                continue
            lexical = max((keyword_overlap(cleaned, candidate) for candidate in context_sentences), default=0.0)
            semantic = 0.0
            if context_vectors is not None and context_vectors.size:
                vector = self.embedder.embed([cleaned])[0]
                semantic = max(cosine(vector, ctx) for ctx in context_vectors)
            score = max(lexical, semantic)
            scores.append(score)
            if score >= 0.42:
                supported.append(cleaned)
            else:
                unsupported.append(cleaned)
        average = sum(scores) / len(scores) if scores else 0.0
        return SupportJudgement(
            score=round(min(1.0, average), 4),
            supported_claims=supported,
            unsupported_claims=unsupported,
        )

    def judge_usefulness(self, question: str, answer: str) -> float:
        if not answer.strip():
            return 0.0
        bilingual = expand_bilingual(question)
        lexical = coverage(question, answer)
        cross_lingual = coverage(bilingual, answer) if bilingual else 0.0
        chars = char_coverage(question, answer)
        semantic = 0.0
        try:
            vectors = self.embedder.embed([question, answer])
            if len(vectors) >= 2:
                semantic = cosine(vectors[0], vectors[1])
        except Exception:  # noqa: BLE001 - 覆盖度判定不应中断主流程
            semantic = 0.0
        score = max(lexical, cross_lingual, chars, semantic)
        if NUMERIC_INTENT_RE.search(question) and DIGIT_RE.search(answer):
            score = max(score, 0.62)
        return round(clamp(score), 4)


class OpenAICompatLLM(BaseLLM):
    name = "openai-compatible"

    def __init__(self, settings: Settings) -> None:
        self.model = settings.llm_chat_model
        self.client = httpx.Client(
            base_url=settings.llm_base_url.rstrip("/"),
            headers={"Authorization": f"Bearer {settings.llm_api_key}"},
            timeout=settings.llm_timeout_seconds,
        )

    def _chat(self, system: str, user: str, json_mode: bool = False) -> LLMResponse:
        payload: dict = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0.1,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        response = self.client.post("/chat/completions", json=payload)
        response.raise_for_status()
        data = response.json()
        usage = data.get("usage", {})
        return LLMResponse(
            text=data["choices"][0]["message"]["content"] or "",
            prompt_tokens=int(usage.get("prompt_tokens", 0)),
            completion_tokens=int(usage.get("completion_tokens", 0)),
        )

    def answer(self, question: str, docs: list[RetrievedDoc], strict: bool = False) -> LLMResponse:
        if not docs:
            return LLMResponse("知识库中未检索到与该问题相关的资料，暂时无法给出有依据的回答。")
        context = "\n\n".join(
            f"[{index}] 文档：{doc.doc_title}｜章节：{' / '.join(doc.section_path) or '-'}\n{doc.text}"
            for index, doc in enumerate(docs, start=1)
        )
        system = (
            "你是企业知识库问答助手。只能依据给定资料作答，不得编造；"
            "每个事实性句子后用 [n] 标注来源编号；资料不足时明确说明缺少哪类信息。"
        )
        if strict:
            system += "本次为严格模式：删除所有无法由资料直接支持的内容。"
        return self._chat(system, f"资料：\n{context}\n\n问题：{question}")

    def rewrite(self, question: str, hints: list[str] | None = None) -> LLMResponse:
        system = "你是检索查询重写器。输出一条更利于知识库检索的查询，保留原始约束，不要解释。"
        user = f"原始问题：{question}"
        if hints:
            user += f"\n可参考相关术语：{', '.join(hints[:5])}"
        return self._chat(system, user)

    def hyde(self, question: str) -> LLMResponse:
        system = (
            "请写一段 120 字以内、看起来像知识库原文的假设性文档（HyDE），"
            "覆盖该问题的定义、参数与流程，不要回答问题本身。"
        )
        return self._chat(system, f"问题：{question}")

    def judge_support(self, question: str, answer: str, docs: list[RetrievedDoc]) -> SupportJudgement:
        context = "\n\n".join(doc.text for doc in docs)
        system = (
            "你是严格的事实核查器。给定资料与答案，逐句判断答案是否被资料直接支持。"
            '仅输出 JSON：{"score": 0-1, "unsupported_claims": ["..."]}'
        )
        try:
            response = self._chat(system, f"资料：\n{context}\n\n答案：\n{answer}", json_mode=True)
            payload = _parse_json(response.text)
            return SupportJudgement(
                score=float(payload.get("score", 0.0)),
                unsupported_claims=[str(c) for c in payload.get("unsupported_claims", [])],
            )
        except Exception:  # noqa: BLE001 - 判定失败时不阻塞主流程
            return SupportJudgement(score=0.0, unsupported_claims=["judge-failed"])

    def judge_usefulness(self, question: str, answer: str) -> float:
        system = '判断答案是否完整解决了用户问题。仅输出 JSON：{"score": 0-1}'
        try:
            response = self._chat(system, f"问题：{question}\n答案：{answer}", json_mode=True)
            return float(_parse_json(response.text).get("score", 0.0))
        except Exception:  # noqa: BLE001
            return 0.0


def _parse_json(text: str) -> dict:
    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        return {}
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}


def build_llm(settings: Settings, embedder: BaseEmbedder) -> BaseLLM:
    heuristic = HeuristicLLM(embedder)
    if not settings.llm_enabled:
        return heuristic
    try:
        primary = OpenAICompatLLM(settings)
        return ResilientLLM(primary, heuristic)
    except Exception:  # noqa: BLE001
        return heuristic


class ResilientLLM(BaseLLM):
    """主 LLM 调用失败时逐方法降级，保证可用性。"""

    name = "openai+fallback"

    def __init__(self, primary: BaseLLM, fallback: BaseLLM) -> None:
        self.primary = primary
        self.fallback = fallback

    def _call(self, method: str, *args, **kwargs):
        try:
            return getattr(self.primary, method)(*args, **kwargs)
        except Exception:  # noqa: BLE001
            self.name = "heuristic-offline"
            return getattr(self.fallback, method)(*args, **kwargs)

    def answer(self, question, docs, strict=False):
        return self._call("answer", question, docs, strict)

    def rewrite(self, question, hints=None):
        return self._call("rewrite", question, hints)

    def hyde(self, question):
        return self._call("hyde", question)

    def judge_support(self, question, answer, docs):
        return self._call("judge_support", question, answer, docs)

    def judge_usefulness(self, question, answer):
        return self._call("judge_usefulness", question, answer)
