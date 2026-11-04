"""DOCX 解析器：优先 python-docx，缺失时用标准库 zip + XML 解析（零额外依赖）。"""

from __future__ import annotations

import io
import re
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

from ..models import Block, ParsedDocument
from ..text import detect_language, normalize_text, sha1_id
from .helpers import markdown_table

W_NS = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"


def parse_docx(data: bytes, filename: str, doc_id: str | None = None) -> ParsedDocument:
    try:
        return _parse_with_python_docx(data, filename, doc_id)
    except ImportError:
        return _parse_with_xml(data, filename, doc_id)


def _parse_with_python_docx(data: bytes, filename: str, doc_id: str | None) -> ParsedDocument:
    from docx import Document  # type: ignore[import-not-found]

    document = Document(io.BytesIO(data))
    blocks: list[Block] = []
    for para in document.paragraphs:
        text = normalize_text(para.text)
        if not text:
            continue
        style = (para.style.name or "").lower() if para.style else ""
        level = _style_level(style)
        if level:
            blocks.append(Block(type="heading", text=text, level=level))
        elif _is_caption(text):
            blocks.append(Block(type="figure", text=text, caption=re.sub(r"^[^:：]*[:：]?", "", text)))
        else:
            blocks.append(Block(type="paragraph", text=text))

    for table in document.tables:
        rows = [[normalize_text(cell.text) for cell in row.cells] for row in table.rows]
        if rows:
            blocks.append(Block(type="table", text=markdown_table(rows)))

    content = "\n".join(b.text for b in blocks)
    title = Path(filename).stem
    for block in blocks:
        if block.type == "heading":
            title = block.text
            break
    return ParsedDocument(
        doc_id=doc_id or sha1_id(filename, content[:400]),
        title=title,
        source=filename,
        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        language=detect_language(content),
        blocks=blocks,
        markdown=content,
        metadata={"tables": sum(1 for b in blocks if b.type == "table"), "parser": "python-docx"},
    )


def _parse_with_xml(data: bytes, filename: str, doc_id: str | None) -> ParsedDocument:
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            xml = archive.read("word/document.xml")
    except (KeyError, zipfile.BadZipFile) as exc:
        raise ValueError(f"无法解析 DOCX 文件 {filename}: {exc}") from exc

    root = ET.fromstring(xml)
    body = root.find(f"{W_NS}body")
    blocks: list[Block] = []
    if body is None:
        raise ValueError("DOCX 结构缺少 body 节点")

    for child in body:
        if child.tag == f"{W_NS}p":
            text = _paragraph_text(child)
            if not text:
                continue
            style = _paragraph_style(child)
            level = _style_level(style)
            if level:
                blocks.append(Block(type="heading", text=text, level=level))
            elif _is_caption(text):
                blocks.append(Block(type="figure", text=text, caption=text))
            else:
                blocks.append(Block(type="paragraph", text=text))
        elif child.tag == f"{W_NS}tbl":
            rows: list[list[str]] = []
            for row in child.findall(f"{W_NS}tr"):
                cells = []
                for cell in row.findall(f"{W_NS}tc"):
                    cell_text = "".join(_paragraph_text(p) for p in cell.findall(f"{W_NS}p"))
                    cells.append(normalize_text(cell_text))
                if any(cells):
                    rows.append(cells)
            if rows:
                blocks.append(Block(type="table", text=markdown_table(rows)))

    content = "\n".join(b.text for b in blocks)
    title = Path(filename).stem
    for block in blocks:
        if block.type == "heading":
            title = block.text
            break
    return ParsedDocument(
        doc_id=doc_id or sha1_id(filename, content[:400]),
        title=title,
        source=filename,
        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        language=detect_language(content),
        blocks=blocks,
        markdown=content,
        metadata={"tables": sum(1 for b in blocks if b.type == "table"), "parser": "zipxml"},
    )


def _paragraph_text(node: ET.Element) -> str:
    parts: list[str] = []
    for elem in node.iter():
        if elem.tag == f"{W_NS}t" and elem.text:
            parts.append(elem.text)
        elif elem.tag == f"{W_NS}tab":
            parts.append("\t")
        elif elem.tag == f"{W_NS}br":
            parts.append("\n")
    return normalize_text("".join(parts))


def _paragraph_style(node: ET.Element) -> str:
    ppr = node.find(f"{W_NS}pPr")
    if ppr is None:
        return ""
    style = ppr.find(f"{W_NS}pStyle")
    if style is None:
        return ""
    return (style.get(f"{W_NS}val") or "").lower()


def _style_level(style: str) -> int:
    if not style:
        return 0
    match = re.search(r"(?:heading|标题)\s*([1-6])", style)
    if match:
        return int(match.group(1))
    if style in {"title", "标题"}:
        return 1
    return 0


def _is_caption(text: str) -> bool:
    return bool(re.match(r"^(图|表|Figure|Table)\s*[0-9一二三四五六七八九十]", text, re.IGNORECASE))
