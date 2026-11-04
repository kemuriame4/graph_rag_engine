"""解析调度器：按扩展名路由到对应解析器，统一编码与错误处理。"""

from __future__ import annotations

from pathlib import Path

from ..models import ParsedDocument
from .docx import parse_docx
from .html import parse_html
from .markdown import parse_markdown
from .pdf import parse_pdf
from .plaintext import parse_plaintext


class UnsupportedDocumentError(ValueError):
    """缺少可选解析依赖或格式不受支持。"""


class DocumentParser:
    supported_extensions = (".pdf", ".docx", ".md", ".markdown", ".txt", ".html", ".htm", ".csv")

    def parse_bytes(self, data: bytes, filename: str) -> ParsedDocument:
        suffix = Path(filename).suffix.lower()
        if suffix == ".pdf":
            try:
                return parse_pdf(data, filename)
            except ImportError as exc:
                raise UnsupportedDocumentError(str(exc)) from exc
        if suffix == ".docx":
            try:
                return parse_docx(data, filename)
            except ImportError as exc:
                raise UnsupportedDocumentError(
                    "解析 DOCX 需要安装 python-docx：pip install python-docx"
                ) from exc
        if suffix in {".md", ".markdown"}:
            return parse_markdown(self._decode(data), filename)
        if suffix in {".html", ".htm"}:
            return parse_html(self._decode(data), filename)
        if suffix == ".csv":
            return parse_plaintext(self._decode(data).replace(",", "\t"), filename)
        if suffix in {".txt", ""}:
            return parse_plaintext(self._decode(data), filename)
        raise UnsupportedDocumentError(
            f"不支持的文档格式：{suffix or 'unknown'}；当前支持 {', '.join(self.supported_extensions)}"
        )

    def parse_path(self, path: str | Path) -> ParsedDocument:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"文件不存在：{path}")
        return self.parse_bytes(path.read_bytes(), path.name)

    @staticmethod
    def _decode(data: bytes) -> str:
        for encoding in ("utf-8-sig", "utf-8", "gb18030", "big5"):
            try:
                return data.decode(encoding)
            except UnicodeDecodeError:
                continue
        return data.decode("utf-8", errors="ignore")
