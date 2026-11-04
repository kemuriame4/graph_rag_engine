"""Markdown 解析器：标题层级、管道表格、题注、公式与列表。"""

from __future__ import annotations

import re
from pathlib import Path

from ..models import Block, ParsedDocument
from ..text import detect_language, normalize_text, sha1_id
from .helpers import CAPTION_RE, clean_heading, markdown_table, parse_pipe_table

HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")
LIST_RE = re.compile(r"^\s*(?:[-*+]|\d+[.)、])\s+")


def parse_markdown(content: str, filename: str, doc_id: str | None = None) -> ParsedDocument:
    content = normalize_text(content)
    doc_id = doc_id or sha1_id(filename, content[:400])
    lines = content.split("\n")

    blocks: list[Block] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        if not stripped:
            i += 1
            continue

        # 代码块：整体保留，避免内部 # 被误识别为标题
        if stripped.startswith("```"):
            body: list[str] = []
            i += 1
            while i < len(lines) and not lines[i].strip().startswith("```"):
                body.append(lines[i])
                i += 1
            i += 1
            blocks.append(Block(type="code", text="\n".join(body)))
            continue

        heading = HEADING_RE.match(line)
        if heading:
            blocks.append(
                Block(type="heading", text=clean_heading(heading.group(2)), level=len(heading.group(1)))
            )
            i += 1
            continue

        if stripped.startswith("|") and i + 1 < len(lines):
            rows = parse_pipe_table(lines[i:])
            if rows:
                blocks.append(Block(type="table", text=markdown_table(rows)))
                i += len(rows) + 1
                continue

        caption = CAPTION_RE.match(stripped)
        if caption:
            blocks.append(
                Block(type="figure", text=stripped, caption=normalize_text(caption.group(1)))
            )
            i += 1
            continue

        if stripped.startswith(("$$", "\\[", "\\begin{")):
            formula_lines: list[str] = [stripped]
            i += 1
            while i < len(lines) and not lines[i].strip().endswith(("$$", "\\]", "\\end}")):
                formula_lines.append(lines[i].strip())
                i += 1
            if i < len(lines):
                formula_lines.append(lines[i].strip())
                i += 1
            blocks.append(Block(type="formula", text="\n".join(formula_lines)))
            continue

        if LIST_RE.match(line):
            blocks.append(Block(type="list_item", text=normalize_text(stripped)))
            i += 1
            continue

        # 普通段落：合并连续文本行（保留软换行语义）
        paragraph: list[str] = [stripped]
        i += 1
        while i < len(lines):
            nxt = lines[i].strip()
            if (
                not nxt
                or HEADING_RE.match(lines[i])
                or nxt.startswith("|")
                or LIST_RE.match(lines[i])
                or nxt.startswith(("```", "$$", "\\[", "\\begin{"))
                or CAPTION_RE.match(nxt)
            ):
                break
            paragraph.append(nxt)
            i += 1
        blocks.append(Block(type="paragraph", text=" ".join(paragraph)))

    title = _infer_title(blocks, filename)
    return ParsedDocument(
        doc_id=doc_id,
        title=title,
        source=filename,
        mime="text/markdown",
        language=detect_language(content),
        blocks=blocks,
        markdown=render_markdown(blocks),
        page_count=0,
        metadata={"tables": sum(1 for b in blocks if b.type == "table")},
    )


def _infer_title(blocks: list[Block], filename: str) -> str:
    for block in blocks:
        if block.type == "heading" and block.level <= 2 and block.text:
            return block.text
    return Path(filename).stem


def render_markdown(blocks: list[Block]) -> str:
    parts: list[str] = []
    for block in blocks:
        if block.type == "heading":
            parts.append("#" * max(1, min(block.level, 6)) + " " + block.text)
        elif block.type == "table":
            parts.append(block.text)
        elif block.type == "formula":
            parts.append(block.text)
        elif block.type == "code":
            parts.append("```\n" + block.text + "\n```")
        elif block.type == "list_item":
            parts.append("- " + block.text)
        else:
            parts.append(block.text)
    return "\n\n".join(p for p in parts if p)
