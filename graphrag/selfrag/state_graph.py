"""极简状态图执行器：节点 + 条件边 + 递归上限，语义对齐 LangGraph StateGraph。

之所以不直接依赖 langgraph：核心链路需要可在离线/受限环境稳定复现；
当需要接入 LangGraph 时，仅需把本文件的节点函数按同名注册即可。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

END = "__end__"
NodeFn = Callable[[dict[str, Any]], dict[str, Any] | None]
RouterFn = Callable[[dict[str, Any]], str]


@dataclass
class _Edge:
    router: RouterFn
    mapping: dict[str, str]


class StateGraph:
    def __init__(self) -> None:
        self.nodes: dict[str, NodeFn] = {}
        self.edges: dict[str, _Edge] = {}
        self.entry: str | None = None

    def add_node(self, name: str, fn: NodeFn) -> None:
        self.nodes[name] = fn

    def set_entry_point(self, name: str) -> None:
        self.entry = name

    def add_conditional_edges(self, source: str, router: RouterFn, mapping: dict[str, str]) -> None:
        self.edges[source] = _Edge(router=router, mapping=mapping)

    def compile(self) -> "CompiledGraph":
        if not self.entry:
            raise ValueError("StateGraph 缺少 entry point")
        return CompiledGraph(self.nodes, self.edges, self.entry)


@dataclass
class CompiledGraph:
    nodes: dict[str, NodeFn]
    edges: dict[str, _Edge]
    entry: str
    execution_path: list[str] = field(default_factory=list)

    def invoke(
        self,
        state: dict[str, Any],
        recursion_limit: int = 30,
        on_step: Callable[[str, dict[str, Any] | None, float], None] | None = None,
    ) -> dict[str, Any]:
        import time

        current = self.entry
        steps = 0
        self.execution_path = [current]
        while current != END:
            if steps >= recursion_limit:
                raise RuntimeError(f"状态图超过递归上限 {recursion_limit}，可能存在死循环")
            node = self.nodes.get(current)
            if node is None:
                raise KeyError(f"未注册的节点：{current}")
            started = time.perf_counter()
            patch = node(state)
            elapsed = (time.perf_counter() - started) * 1000
            if patch:
                state.update(patch)
            if on_step:
                on_step(current, patch, elapsed)
            edge = self.edges.get(current)
            if edge is None:
                break
            key = edge.router(state)
            nxt = edge.mapping.get(key)
            if nxt is None:
                raise KeyError(f"路由 {current} 返回未映射的键：{key}")
            current = nxt
            self.execution_path.append(current)
            steps += 1
        return state
