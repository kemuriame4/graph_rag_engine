"""纯文本解析器：编号标题、下划线标题、制表符/管道表格的版面还原。"""

from __future__ import annotations

import re
from pathlib import Path

from ..models import Block, ParsedDocument
from ..text import detect_language, normalize_text, sha1_id
from .helpers import CAPTION_RE, is_caption, markdown_table, numbered_heading_level

UNDERLINE_RE = re.compile(r"^(?:=+|-+){3,}$")
CHAPTER_RE = re.compile(r"^(第[一二三四五六七八九十百0-9]+[章节篇部分])\s*[、.：:]?\s*(\S.*)?$")


def _heading_of(line: str) -> tuple[int, str] | None:
    stripped = line.strip()
    if len(stripped) > 80:
        return None

    chapter = CHAPTER_RE.match(stripped)
    if chapter:
        return 1, stripped

    numbered = numbered_heading_level(stripped)
    if numbered:
        level, text = numbered
        return min(level, 4), text

    # 全大写英文短行 / 无标点短行视作标题（保守策略：必须无句末标点）
    if 2 <= len(stripped) <= 50 and stripped[-1] not in "。！？.!?,，;；":
        if stripped.isupper() and len(stripped.split()) <= 8:
            return 2, stripped.title()
        if stripped.startswith(("附录", "前言", "目录", "摘要", "结论")):
            return 1, stripped
    return None


def parse_plaintext(
    content: str, filename: str, doc_id: str | None = None, page_offsets: list[tuple[int, int]] | None = None
) -> ParsedDocument:
    """page_offsets 为 [(起始字符下标, 页码)]，PDF 逐页解析时用于回填页码。"""
    content = normalize_text(content)
    doc_id = doc_id or sha1_id(filename, content[:400])
    lines = content.split("\n")
    blocks: list[Block] = []

    def page_at(pos: int) -> int | None:
        if not page_offsets:
            return None
        page = page_offsets[0][1]
        for offset, number in page_offsets:
            if pos >= offset:
                page = number
        return page

    position = 0
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        if not stripped:
            position += len(line) + 1
            i += 1
            continue

        # 文档首行短标题：无句末标点且不超过 80 字符
        if (
            i == 0
            and len(stripped) <= 80
            and stripped[-1] not in "。！？.!?,，;；：:"
            and not UNDERLINE_RE.match(stripped)
        ):
            blocks.append(Block(type="heading", text=stripped, level=1, page=page_at(position)))
            position += len(line) + 1
            i += 1
            continue

        # 下划线式标题：Title\n=====
        if i + 1 < len(lines) and UNDERLINE_RE.match(lines[i + 1].strip()) and len(stripped) <= 80:
            level = 1 if lines[i + 1].strip().startswith("=") else 2
            blocks.append(Block(type="heading", text=stripped, level=level, page=page_at(position)))
            position += len(line) + len(lines[i + 1]) + 2
            i += 2
            continue

        heading = _heading_of(line)
        if heading:
            level, text = heading
            blocks.append(Block(type="heading", text=text, level=level, page=page_at(position)))
            position += len(line) + 1
            i += 1
            continue

        if is_caption(stripped):
            caption = CAPTION_RE.match(stripped)
            blocks.append(
                Block(
                    type="figure",
                    text=stripped,
                    caption=normalize_text(caption.group(1) if caption else ""),
                    page=page_at(position),
                )
            )
            position += len(line) + 1
            i += 1
            continue

        # 表格：连续多行含制表符或 |，且列数一致
        if "\t" in line or stripped.count("|") >= 2:
            table_lines = [line]
            j = i + 1
            while j < len(lines) and ("\t" in lines[j] or lines[j].strip().count("|") >= 2):
                table_lines.append(lines[j])
                j += 1
            if len(table_lines) >= 2:
                rows = []
                for raw in table_lines:
                    if "\t" in raw:
                        cells = [c.strip() for c in raw.split("\t")]
                    else:
                        cells = [c.strip() for c in raw.strip().strip("|").split("|")]
                    if any(cells):
                        rows.append(cells)
                blocks.append(
                    Block(type="table", text=markdown_table(rows), page=page_at(position))
                )
                position += sum(len(x) + 1 for x in table_lines)
                i = j
                continue

        # 普通段落：遇到空行或标题/表格即结束
        paragraph = [stripped]
        start = position
        position += len(line) + 1
        i += 1
        while i < len(lines):
            nxt = lines[i]
            if not nxt.strip() or _heading_of(nxt) or is_caption(nxt.strip()):
                break
            if "\t" in nxt or nxt.strip().count("|") >= 2:
                break
            paragraph.append(nxt.strip())
            position += len(nxt) + 1
            i += 1
        blocks.append(
            Block(type="paragraph", text=" ".join(paragraph), page=page_at(start))
        )

    title = Path(filename).stem
    for block in blocks:
        if block.type == "heading" and block.level == 1:
            title = block.text
            break

    return ParsedDocument(
        doc_id=doc_id,
        title=title,
        source=filename,
        mime="text/plain",
        language=detect_language(content),
        blocks=blocks,
        markdown=content,
        page_count=len({p for _, p in page_offsets}) if page_offsets else 0,
        metadata={"tables": sum(1 for b in blocks if b.type == "table")},
    )
