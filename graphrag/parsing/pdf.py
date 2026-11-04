"""PDF 解析器：PyMuPDF 版面/表格优先，pypdf 纯文本降级。"""

from __future__ import annotations

import io
from collections import Counter
from pathlib import Path

from ..models import Block, ParsedDocument
from ..text import detect_language, normalize_text, sha1_id
from .helpers import is_caption, markdown_table
from .plaintext import parse_plaintext


def parse_pdf(data: bytes, filename: str, doc_id: str | None = None) -> ParsedDocument:
    try:
        return _parse_with_pymupdf(data, filename, doc_id)
    except ImportError:
        pass
    try:
        return _parse_with_pypdf(data, filename, doc_id)
    except ImportError as exc:
        raise ImportError(
            "解析 PDF 需要安装 PyMuPDF 或 pypdf：pip install pymupdf  或  pip install pypdf"
        ) from exc


def _parse_with_pymupdf(data: bytes, filename: str, doc_id: str | None) -> ParsedDocument:
    import fitz  # type: ignore[import-not-found]  # PyMuPDF

    document = fitz.open(stream=data, filetype="pdf")
    content_blocks: list[Block] = []
    line_sizes: Counter[float] = Counter()
    raw_pages: list[list[tuple[str, float, list[float]]]] = []

    for page_index, page in enumerate(document, start=1):
        page_lines: list[tuple[str, float, list[float]]] = []
        for block in page.get_text("dict").get("blocks", []):
            if block.get("type") != 0:
                continue
            for line in block.get("lines", []):
                spans = line.get("spans", [])
                text = normalize_text("".join(span.get("text", "") for span in spans))
                if not text:
                    continue
                size = max((float(span.get("size", 0)) for span in spans), default=0.0)
                page_lines.append((text, round(size, 1), [round(v, 1) for v in line.get("bbox", [])]))
                line_sizes[round(size, 1)] += len(text)
        raw_pages.append(page_lines)

    median_size = _weighted_median(line_sizes) or 10.0

    for page_index, page_lines in enumerate(raw_pages, start=1):
        page = document[page_index - 1]

        # 表格优先提取（PyMuPDF >= 1.23），失败静默降级
        tables: list[str] = []
        try:
            for table in page.find_tables().tables:
                rows = [[normalize_text(cell or "") for cell in row] for row in table.extract()]
                if rows:
                    tables.append(markdown_table(rows))
        except Exception:  # noqa: BLE001 - 表格识别属于增强能力
            tables = []

        paragraph: list[str] = []
        paragraph_bbox: list[float] | None = None
        paragraph_size = 0.0

        def flush() -> None:
            nonlocal paragraph, paragraph_bbox, paragraph_size
            if paragraph:
                content_blocks.append(
                    Block(
                        type="paragraph",
                        text=" ".join(paragraph),
                        page=page_index,
                        bbox=paragraph_bbox,
                    )
                )
            paragraph, paragraph_bbox, paragraph_size = [], None, 0.0

        for text, size, bbox in page_lines:
            is_heading = (
                size >= median_size * 1.22
                and len(text) <= 80
                and text[-1:] not in "。！？.!?,，;；"
            )
            if is_heading:
                flush()
                level = 1 if size >= median_size * 1.6 else 2 if size >= median_size * 1.35 else 3
                content_blocks.append(
                    Block(type="heading", text=text, level=level, page=page_index, bbox=bbox)
                )
                continue
            if is_caption(text):
                flush()
                content_blocks.append(Block(type="figure", text=text, page=page_index, bbox=bbox))
                continue
            if paragraph and abs(size - paragraph_size) > 1.2:
                flush()
            if not paragraph:
                paragraph_bbox = bbox
                paragraph_size = size
            paragraph.append(text)
        flush()

        for table_text in tables:
            content_blocks.append(Block(type="table", text=table_text, page=page_index))

    content = "\n".join(block.text for block in content_blocks)
    title = Path(filename).stem
    for block in content_blocks:
        if block.type == "heading" and len(block.text) <= 60:
            title = block.text
            break

    return ParsedDocument(
        doc_id=doc_id or sha1_id(filename, content[:400]),
        title=title,
        source=filename,
        mime="application/pdf",
        language=detect_language(content),
        blocks=content_blocks,
        markdown=content,
        page_count=len(raw_pages),
        metadata={
            "parser": "pymupdf",
            "tables": sum(1 for b in content_blocks if b.type == "table"),
            "median_font_size": median_size,
        },
    )


def _parse_with_pypdf(data: bytes, filename: str, doc_id: str | None) -> ParsedDocument:
    from pypdf import PdfReader  # type: ignore[import-not-found]

    reader = PdfReader(io.BytesIO(data))
    parts: list[str] = []
    offsets: list[tuple[int, int]] = []
    cursor = 0
    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        offsets.append((cursor, page_number))
        parts.append(text)
        cursor += len(text) + 2
    content = "\n\n".join(parts)
    parsed = parse_plaintext(content, filename, doc_id, page_offsets=offsets)
    parsed.mime = "application/pdf"
    parsed.page_count = len(reader.pages)
    parsed.metadata["parser"] = "pypdf"
    return parsed


def _weighted_median(counter: Counter[float]) -> float:
    if not counter:
        return 0.0
    total = sum(counter.values())
    running = 0
    for size in sorted(counter):
        running += counter[size]
        if running >= total / 2:
            return float(size)
    return float(max(counter))
