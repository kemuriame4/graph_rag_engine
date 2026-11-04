"""HTML 解析器：标题层级、段落、列表与表格。"""

from __future__ import annotations

import re
from html.parser import HTMLParser
from pathlib import Path

from ..models import Block, ParsedDocument
from ..text import detect_language, normalize_text, sha1_id
from .helpers import markdown_table

HEADING_TAGS = {"h1": 1, "h2": 2, "h3": 3, "h4": 4, "h5": 5, "h6": 6}


class _BlockHTMLParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.blocks: list[Block] = []
        self._tag_stack: list[str] = []
        self._buffer: list[str] = []
        self._table_rows: list[list[str]] | None = None
        self._row: list[str] | None = None
        self._cell: list[str] | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self._tag_stack.append(tag)
        if tag in HEADING_TAGS:
            self._flush_text()
        elif tag == "table":
            self._flush_text()
            self._table_rows = []
        elif tag == "tr" and self._table_rows is not None:
            self._row = []
        elif tag in {"td", "th"} and self._row is not None:
            self._cell = []
        elif tag in {"p", "li"}:
            self._flush_text()

    def handle_endtag(self, tag: str) -> None:
        if tag in HEADING_TAGS:
            text = normalize_text(" ".join(self._buffer))
            if text:
                self.blocks.append(Block(type="heading", text=text, level=HEADING_TAGS[tag]))
            self._buffer = []
        elif tag in {"td", "th"} and self._cell is not None and self._row is not None:
            self._row.append(normalize_text(" ".join(self._cell)))
            self._cell = None
        elif tag == "tr" and self._row is not None and self._table_rows is not None:
            if any(self._row):
                self._table_rows.append(self._row)
            self._row = None
        elif tag == "table":
            if self._table_rows:
                self.blocks.append(Block(type="table", text=markdown_table(self._table_rows)))
            self._table_rows = None
        elif tag in {"p", "li"}:
            text = normalize_text(" ".join(self._buffer))
            if text:
                self.blocks.append(
                    Block(type="list_item" if tag == "li" else "paragraph", text=text)
                )
            self._buffer = []
        if self._tag_stack and self._tag_stack[-1] == tag:
            self._tag_stack.pop()

    def handle_data(self, data: str) -> None:
        if not data.strip():
            return
        if self._cell is not None:
            self._cell.append(data)
        elif self._tag_stack and self._tag_stack[-1] not in {"script", "style"}:
            self._buffer.append(data)

    def _flush_text(self) -> None:
        text = normalize_text(" ".join(self._buffer))
        if text and not self._cell:
            self.blocks.append(Block(type="paragraph", text=text))
        self._buffer = []

    def close(self) -> None:
        super().close()
        self._flush_text()


def parse_html(content: str, filename: str, doc_id: str | None = None) -> ParsedDocument:
    parser = _BlockHTMLParser()
    parser.feed(content)
    parser.close()
    title = Path(filename).stem
    match = re.search(r"<title[^>]*>(.*?)</title>", content, re.IGNORECASE | re.DOTALL)
    if match:
        title = normalize_text(re.sub(r"<[^>]+>", "", match.group(1)))
    for block in parser.blocks:
        if block.type == "heading" and block.level == 1:
            title = block.text
            break
    text = "\n".join(block.text for block in parser.blocks)
    return ParsedDocument(
        doc_id=doc_id or sha1_id(filename, text[:400]),
        title=title,
        source=filename,
        mime="text/html",
        language=detect_language(text),
        blocks=parser.blocks,
        markdown=text,
        metadata={"tables": sum(1 for b in parser.blocks if b.type == "table")},
    )
