"""服务容器：进程内单例装配索引 / 检索 / 引擎 / 评测 / 入库。"""

from __future__ import annotations

import threading

from .config import Settings, settings as default_settings
from .evaluation import Evaluator
from .indexing import SearchIndex
from .ingestion import IngestionService
from .observability import Tracer
from .parsing import DocumentParser
from .retrieval import HybridRetriever
from .selfrag import SelfRAGEngine, build_llm


class Container:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or default_settings
        self.settings.ensure_dirs()
        self.tracer = Tracer(self.settings.trace_dir)
        self.index = SearchIndex(self.settings)
        self.parser = DocumentParser()
        self.retriever = HybridRetriever(self.index, self.settings)
        self.llm = build_llm(self.settings, self.index.embedder)
        self.engine = SelfRAGEngine(
            self.index, self.settings, retriever=self.retriever, llm=self.llm, tracer=self.tracer
        )
        self.evaluator = Evaluator(self.engine, self.settings)
        self.ingestion = IngestionService(self.settings, self.index, tracer=self.tracer, parser=self.parser)


_container: Container | None = None
_lock = threading.Lock()


def get_container() -> Container:
    global _container
    if _container is None:
        with _lock:
            if _container is None:
                _container = Container()
    return _container


def reset_container(settings: Settings | None = None) -> Container:
    """测试用：重建容器。"""
    global _container
    with _lock:
        _container = Container(settings)
    return _container
