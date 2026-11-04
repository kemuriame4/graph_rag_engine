"""文档解析子系统：PDF / Word / Markdown / HTML / 纯文本 → 结构化块序列。

设计原则：
1. 重型依赖（PyMuPDF / python-docx）为可选，缺失时自动降级或给出明确提示；
2. 所有解析器输出统一的 ``Block`` 序列，下游切片、表格与公式处理与格式无关。
"""

from .parser import DocumentParser, UnsupportedDocumentError

__all__ = ["DocumentParser", "UnsupportedDocumentError"]
