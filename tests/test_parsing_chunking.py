"""解析与动态语义切片：标题层级、表格原子性、元数据注入。"""

from __future__ import annotations

from graphrag.chunking import SemanticChunker
from graphrag.config import Settings
from graphrag.parsing import DocumentParser
from graphrag.text import expand_bilingual, char_coverage


MARKDOWN = """# 产品规格

概述段落用于说明产品定位。

## 1. 电气参数

| 参数 | 数值 | 单位 |
| --- | --- | --- |
| 容量 | 1024 | Wh |
| 循环寿命 | 3000 | 次 |

## 2. 认证要求

产品需要通过 UN38.3 认证与 CE 认证。
"""

PLAINTEXT = """海关报关单填写说明

1. 总则
本说明用于指导跨境卖家填写出口报关单。

2. 必填字段
申报价值按实际成交价格填写。
"""


def test_markdown_parse_keeps_heading_levels() -> None:
    parsed = DocumentParser().parse_bytes(MARKDOWN.encode("utf-8"), "spec.md")
    headings = [block for block in parsed.blocks if block.type == "heading"]
    assert parsed.title == "产品规格"
    assert [block.level for block in headings] == [1, 2, 2]
    assert any(block.type == "table" for block in parsed.blocks)


def test_plaintext_first_line_becomes_title() -> None:
    parsed = DocumentParser().parse_bytes(PLAINTEXT.encode("utf-8"), "报关单填写说明.txt")
    assert parsed.title == "海关报关单填写说明"
    assert parsed.blocks[0].type == "heading"
    assert parsed.blocks[0].level == 1


def test_semantic_chunking_section_path_and_table_atomicity(settings: Settings) -> None:
    parsed = DocumentParser().parse_bytes(MARKDOWN.encode("utf-8"), "spec.md")
    chunks = SemanticChunker(settings).chunk(parsed)
    assert chunks
    assert all(chunk.embedding_text == "" for chunk in chunks)  # 由索引阶段注入
    assert any("电气参数" in " / ".join(chunk.section_path) for chunk in chunks)
    table_chunks = [chunk for chunk in chunks if "| 参数 |" in chunk.text]
    assert table_chunks, "表格应作为独立切片保留"
    assert all("| 循环寿命 | 3000 | 次 |" in chunk.text for chunk in table_chunks)
    for previous, current in zip(chunks, chunks[1:]):
        assert previous.next_chunk_id == current.chunk_id
        assert current.prev_chunk_id == previous.chunk_id


def test_bilingual_expansion_and_char_coverage() -> None:
    expanded = expand_bilingual("Battery-X1 的循环寿命和充电时间是多少？")
    assert "cycle life" in expanded
    assert "charge time" in expanded
    # 中文覆盖度只统计有效汉字
    assert char_coverage("循环寿命", "额定循环寿命为 3000 次") == 1.0
    assert char_coverage("循环寿命", "完全无关的内容") < 0.2
