"""解析器共享工具：表格 Markdown 化、标题识别、题注识别。"""

from __future__ import annotations

import re

from ..text import normalize_text

HEADING_NUM_RE = re.compile(r"^(\d+(?:\.\d+)*)[.、]?\s+(\S.{0,60})$")
CN_HEADING_RE = re.compile(r"^([一二三四五六七八九十百]+)、\s*(\S.{0,60})$")
CN_SUB_HEADING_RE = re.compile(r"^（([一二三四五六七八九十]+)）\s*(\S.{0,60})$")
CAPTION_RE = re.compile(
    r"^(?:图|表|附图|Figure|Fig\.?|Table)\s*[0-9一二三四五六七八九十]+(?:[-.][0-9]+)?\s*[:：.\-]?\s*(.*)$",
    re.IGNORECASE,
)
FORMULA_RE = re.compile(r"^\s*(?:\$\$|\\begin\{|\\\[)")
SEPARATOR_CELL_RE = re.compile(r"^:?-{2,}:?$")


def markdown_table(rows: list[list[str]]) -> str:
    """把二维单元格转为标准 Markdown 表格，自动对齐列数。"""
    if not rows:
        return ""
    width = max(len(row) for row in rows)
    padded = [row + [""] * (width - len(row)) for row in rows]
    header = padded[0]
    body = padded[1:]

    def fmt(cells: list[str]) -> str:
        escaped = [normalize_text(c).replace("|", "\\|") for c in cells]
        return "| " + " | ".join(escaped) + " |"

    lines = [fmt(header), "| " + " | ".join(["---"] * width) + " |"]
    lines.extend(fmt(row) for row in body)
    return "\n".join(lines)


def parse_pipe_table(lines: list[str]) -> list[list[str]] | None:
    """解析 Markdown 管道表格；若第二行不是分隔行则返回 None。"""
    if len(lines) < 2:
        return None
    split = lambda line: [c.strip() for c in line.strip().strip("|").split("|")]
    cells = split(lines[0])
    sep = split(lines[1])
    if not all(SEPARATOR_CELL_RE.match(c) for c in sep if c):
        return None
    rows = [cells]
    for line in lines[2:]:
        if "|" not in line:
            break
        rows.append(split(line))
    return rows


def is_caption(line: str) -> bool:
    return bool(CAPTION_RE.match(line.strip()))


def numbered_heading_level(line: str) -> tuple[int, str] | None:
    """识别 ``1.`` / ``1.1`` / ``一、`` / ``（二）`` 等编号标题。"""
    stripped = line.strip()
    match = HEADING_NUM_RE.match(stripped)
    if match:
        return len(match.group(1).split(".")), match.group(2).strip()
    match = CN_HEADING_RE.match(stripped)
    if match:
        return 1, match.group(2).strip()
    match = CN_SUB_HEADING_RE.match(stripped)
    if match:
        return 2, match.group(2).strip()
    return None


def clean_heading(text: str) -> str:
    return normalize_text(text).strip(" #：:")
