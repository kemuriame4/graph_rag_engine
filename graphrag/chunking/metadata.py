"""元数据注入：为切片构造用于向量化的语义增强文本。"""

from __future__ import annotations

from ..models import Chunk


def build_embedding_text(chunk: Chunk) -> str:
    """在正文前注入文档 / 章节 / 实体 / 关键词上下文，提升召回语义密度。"""
    lines = [
        f"[文档] {chunk.doc_title}",
        f"[章节] {chunk.section}",
    ]
    if chunk.keywords:
        lines.append(f"[关键词] {', '.join(chunk.keywords)}")
    if chunk.entities:
        lines.append(f"[实体] {', '.join(chunk.entities[:10])}")
    if chunk.summary:
        lines.append(f"[摘要] {chunk.summary}")
    lines.append("[正文]")
    lines.append(chunk.text)
    return "\n".join(lines)
