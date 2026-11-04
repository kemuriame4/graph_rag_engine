"""动态语义切片：按文档语义大纲切片，而非固定字符窗口。"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..config import Settings
from ..models import Block, Chunk, ParsedDocument
from ..text import estimate_tokens, normalize_text, sha1_id, split_sentences, top_keywords, truncate


@dataclass
class _Section:
    path: list[str] = field(default_factory=list)
    blocks: list[Block] = field(default_factory=list)


ATOMIC_TYPES = {"table", "formula", "figure", "code"}


class SemanticChunker:
    """基于标题大纲构造 section，再在 section 内按语义单元合并成 chunk。

    - 表格 / 公式 / 图示保持原子性，绝不从中间切断；
    - 超长段落按句子边界切分；
    - 相邻 chunk 保留可配置的句子级 overlap，维持上下文连续性。
    """

    def __init__(self, settings: Settings) -> None:
        self.max_tokens = max(120, settings.chunk_max_tokens)
        self.overlap_sentences = max(0, settings.chunk_overlap_sentences)

    def chunk(self, document: ParsedDocument, tenant: str = "default") -> list[Chunk]:
        sections = self._build_sections(document)
        chunks: list[Chunk] = []
        index = 0
        for section in sections:
            for text in self._split_section(section.blocks):
                if not text.strip():
                    continue
                path = section.path or [document.title]
                chunk_id = sha1_id(document.doc_id, "/".join(path), str(index))
                keywords = top_keywords(text, 8)
                summary = self._summarize(text, path)
                pages = [b.page for b in section.blocks if b.page]
                chunks.append(
                    Chunk(
                        chunk_id=chunk_id,
                        doc_id=document.doc_id,
                        doc_title=document.title,
                        text=text,
                        section_path=path,
                        chunk_index=index,
                        token_count=estimate_tokens(text),
                        page_start=min(pages) if pages else None,
                        page_end=max(pages) if pages else None,
                        summary=summary,
                        keywords=keywords,
                        tenant=tenant,
                    )
                )
                index += 1

        for position, chunk in enumerate(chunks):
            if position > 0:
                chunk.prev_chunk_id = chunks[position - 1].chunk_id
            if position + 1 < len(chunks):
                chunk.next_chunk_id = chunks[position + 1].chunk_id
        return chunks

    def _build_sections(self, document: ParsedDocument) -> list[_Section]:
        sections: list[_Section] = []
        stack: list[tuple[int, str]] = []
        current = _Section(path=[])
        for block in document.blocks:
            if block.type == "heading":
                if current.blocks:
                    sections.append(current)
                while stack and stack[-1][0] >= block.level:
                    stack.pop()
                stack.append((block.level, block.text))
                current = _Section(path=[text for _, text in stack])
                continue
            if not block.text.strip():
                continue
            if not current.path:
                current.path = [document.title]
            current.blocks.append(block)
        if current.blocks:
            sections.append(current)
        return sections

    def _split_section(self, blocks: list[Block]) -> list[str]:
        units: list[tuple[str, bool]] = []  # (text, is_atomic)
        for block in blocks:
            text = normalize_text(block.text)
            if not text:
                continue
            if block.type in ATOMIC_TYPES:
                if block.caption and block.type == "figure" and block.caption not in text:
                    text = f"{text}\n题注：{block.caption}"
                units.append((text, True))
                continue
            if estimate_tokens(text) > self.max_tokens:
                for sentence_group in self._group_sentences(text):
                    units.append((sentence_group, False))
            else:
                units.append((text, False))

        chunks: list[str] = []
        buffer: list[str] = []
        buffer_tokens = 0
        overlap_tail: list[str] = []

        def flush() -> None:
            nonlocal buffer, buffer_tokens
            if not buffer:
                return
            body = "\n\n".join(buffer).strip()
            if overlap_tail and chunks:
                prefix = " ".join(overlap_tail)
                body = f"（承接上文）{prefix}\n\n{body}"
            chunks.append(body)
            buffer = []
            buffer_tokens = 0

        for text, is_atomic in units:
            tokens = estimate_tokens(text)
            if is_atomic and buffer:
                flush()
            if buffer and buffer_tokens + tokens > self.max_tokens:
                flush()
                overlap_tail = self._tail_sentences(buffer if buffer else chunks[-1])
            buffer.append(text)
            buffer_tokens += tokens
            if is_atomic:
                flush()
                overlap_tail = []
        flush()
        return chunks

    def _group_sentences(self, text: str) -> list[str]:
        sentences = split_sentences(text)
        groups: list[str] = []
        current: list[str] = []
        current_tokens = 0
        for sentence in sentences:
            tokens = estimate_tokens(sentence)
            if current and current_tokens + tokens > self.max_tokens:
                groups.append(" ".join(current))
                current, current_tokens = [], 0
            current.append(sentence)
            current_tokens += tokens
        if current:
            groups.append(" ".join(current))
        return groups

    def _tail_sentences(self, text: str) -> list[str]:
        if not self.overlap_sentences:
            return []
        sentences = split_sentences(text)
        return sentences[-self.overlap_sentences :]

    @staticmethod
    def _summarize(text: str, path: list[str]) -> str:
        clean = text.replace("|", " ").replace("#", " ")
        clean = " ".join(clean.split())
        heading = path[-1] if path else ""
        if clean.startswith(("（承接上文）", "题注")):
            heading = ""
        body = truncate(clean, 120)
        return f"{heading}：{body}" if heading and heading not in body[:40] else body
