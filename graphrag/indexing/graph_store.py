"""知识图谱存储：NetworkX 多关系有向图，实体-关系-切片三层结构。

节点：
- ``ent:<normalized>``  实体节点（name/type/tenants）
- ``chunk:<chunk_id>``  切片节点（doc_id/section/tenant）
边：
- 实体间语义关系（is_a / includes / requires / depends_on / co_occurs ...）
- ``chunk -> ent`` 的 mentions 边（带权重），用于多跳召回时回落到原文切片。
"""

from __future__ import annotations

import json
from collections import deque
from pathlib import Path

import networkx as nx

from ..models import Chunk, ExtractedRelation, GraphEdge, GraphNode, SubGraph
from ..text import tokenize

ENTITY_PREFIX = "ent:"
CHUNK_PREFIX = "chunk:"


class KnowledgeGraph:
    def __init__(self) -> None:
        self.graph = nx.MultiDiGraph()

    def __len__(self) -> int:
        return self.graph.number_of_nodes()

    def clear(self) -> None:
        self.graph.clear()

    def add_chunk(self, chunk: Chunk) -> None:
        node_id = CHUNK_PREFIX + chunk.chunk_id
        self.graph.add_node(
            node_id,
            kind="chunk",
            label=chunk.section,
            doc_id=chunk.doc_id,
            doc_title=chunk.doc_title,
            tenant=chunk.tenant,
            section=" / ".join(chunk.section_path),
        )

    def add_entity(
        self,
        name: str,
        normalized: str,
        entity_type: str,
        chunk_id: str,
        doc_id: str,
        tenant: str,
    ) -> None:
        node_id = ENTITY_PREFIX + normalized
        if node_id not in self.graph:
            self.graph.add_node(
                node_id,
                kind="entity",
                label=name,
                type=entity_type,
                doc_ids=[],
                tenants=[],
                mentions=0,
            )
        data = self.graph.nodes[node_id]
        if doc_id not in data.setdefault("doc_ids", []):
            data["doc_ids"].append(doc_id)
        if tenant not in data.setdefault("tenants", []):
            data["tenants"].append(tenant)
        data["mentions"] = int(data.get("mentions", 0)) + 1
        self.graph.add_edge(
            CHUNK_PREFIX + chunk_id,
            node_id,
            relation="mentions",
            weight=1.0,
            doc_id=doc_id,
        )

    def add_relation(
        self,
        relation: ExtractedRelation,
        chunk_id: str,
        doc_id: str,
        tenant: str,
    ) -> None:
        source = ENTITY_PREFIX + relation.source
        target = ENTITY_PREFIX + relation.target
        if source not in self.graph or target not in self.graph or source == target:
            return
        key = self._edge_key(source, target, relation.relation)
        if key is not None:
            data = self.graph.edges[source, target, key]
            data["weight"] = round(float(data.get("weight", 1.0)) + relation.weight, 3)
            evidence = data.setdefault("evidence", [])
            if chunk_id not in evidence:
                evidence.append(chunk_id)
            return
        self.graph.add_edge(
            source,
            target,
            relation=relation.relation,
            weight=relation.weight,
            evidence=[chunk_id],
            doc_id=doc_id,
            tenant=tenant,
        )

    def _edge_key(self, source: str, target: str, relation: str):
        if not self.graph.has_edge(source, target):
            return None
        for key, data in self.graph.get_edge_data(source, target).items():
            if data.get("relation") == relation:
                return key
        return None

    # ---------------- 检索 ----------------
    def link_entities(self, query: str, tenant: str, limit: int = 8) -> list[tuple[str, float]]:
        """将 Query 链接到图谱实体：精确包含优先，其次关键词重叠。"""
        lowered = query.lower()
        query_tokens = set(tokenize(query))
        candidates: list[tuple[str, float]] = []
        for node_id, data in self.graph.nodes(data=True):
            if data.get("kind") != "entity":
                continue
            tenants = data.get("tenants") or []
            if tenants and tenant not in tenants:
                continue
            label = str(data.get("label", ""))
            normalized = node_id[len(ENTITY_PREFIX) :]
            if not label:
                continue
            label_lower = label.lower()
            if label_lower in lowered or normalized in lowered:
                score = 1.0 + min(len(label), 10) / 10
            else:
                overlap = len(query_tokens & set(tokenize(label)))
                if overlap == 0:
                    continue
                score = overlap / max(len(query_tokens), 1)
            candidates.append((node_id, score))
        candidates.sort(key=lambda item: item[1], reverse=True)
        return candidates[:limit]

    def chunk_hits_for_entities(
        self, entity_nodes: list[tuple[str, float]], tenant: str, max_hops: int = 2
    ) -> tuple[dict[str, float], list[tuple[str, list[str]]]]:
        """从实体节点多跳扩散，返回 (chunk_id -> score, 图谱路径样例)。"""
        scores: dict[str, float] = {}
        paths: list[tuple[str, list[str]]] = []
        visited: dict[str, int] = {}
        queue: deque[tuple[str, int, list[str], float]] = deque()

        for node_id, seed_score in entity_nodes:
            if node_id not in self.graph:
                continue
            queue.append((node_id, 0, [str(self.graph.nodes[node_id].get("label", node_id))], seed_score))
            visited[node_id] = 0

        while queue:
            node_id, hop, path, score = queue.popleft()

            # 反向：所有提到该实体的切片
            for chunk_node in self.graph.predecessors(node_id):
                if not chunk_node.startswith(CHUNK_PREFIX):
                    continue
                if self.graph.nodes[chunk_node].get("tenant") != tenant:
                    continue
                chunk_id = chunk_node[len(CHUNK_PREFIX) :]
                scores[chunk_id] = scores.get(chunk_id, 0.0) + score * (1.0 - 0.12 * hop)
                if len(paths) < 40:
                    paths.append((chunk_id, path))

            if hop >= max_hops:
                continue

            for neighbor in self.graph.successors(node_id):
                if not neighbor.startswith(ENTITY_PREFIX):
                    continue
                next_hop = hop + 1
                if visited.get(neighbor, 99) <= next_hop:
                    continue
                visited[neighbor] = next_hop
                label = self.graph.nodes[neighbor].get("label", neighbor)
                queue.append((neighbor, next_hop, path + [str(label)], score * 0.5))

        deduped_paths: list[tuple[str, list[str]]] = []
        seen: set[str] = set()
        for chunk_id, path in paths:
            if chunk_id in seen:
                continue
            seen.add(chunk_id)
            deduped_paths.append((chunk_id, path))
        return {cid: round(value, 4) for cid, value in scores.items()}, deduped_paths

    def subgraph(self, entity_names: list[str], hops: int = 2, limit: int = 80, tenant: str = "default") -> SubGraph:
        seeds: list[str] = []
        for name in entity_names:
            node_id = ENTITY_PREFIX + name.lower().replace(" ", "")
            if node_id in self.graph:
                seeds.append(node_id)
                continue
            seed_matches = [
                candidate
                for candidate, data in self.graph.nodes(data=True)
                if data.get("kind") == "entity" and name.lower() in str(data.get("label", "")).lower()
            ]
            seeds.extend(seed_matches)
        if not seeds and entity_names:
            matches = self.link_entities(" ".join(entity_names), tenant, limit=4)
            seeds = [node_id for node_id, _ in matches]

        nodes: set[str] = set()
        frontier = [(node_id, 0) for node_id in seeds]
        visited = set(seeds)
        while frontier:
            node_id, depth = frontier.pop(0)
            if node_id.startswith(CHUNK_PREFIX):
                continue
            nodes.add(node_id)
            if depth >= hops or len(nodes) >= limit:
                continue
            for neighbor in list(self.graph.successors(node_id)) + list(self.graph.predecessors(node_id)):
                if neighbor.startswith(CHUNK_PREFIX) or neighbor in visited:
                    continue
                visited.add(neighbor)
                frontier.append((neighbor, depth + 1))

        graph_nodes: list[GraphNode] = []
        for node_id in list(nodes)[:limit]:
            data = self.graph.nodes[node_id]
            graph_nodes.append(
                GraphNode(
                    id=node_id[len(ENTITY_PREFIX) :],
                    label=str(data.get("label", node_id)),
                    type=str(data.get("type", "term")),
                    degree=int(self.graph.degree(node_id)),
                    doc_ids=list(data.get("doc_ids", [])),
                )
            )
        allowed = {ENTITY_PREFIX + node.id for node in graph_nodes}
        edges: list[GraphEdge] = []
        for source, target, data in self.graph.edges(data=True):
            if source in allowed and target in allowed:
                edges.append(
                    GraphEdge(
                        source=source[len(ENTITY_PREFIX) :],
                        target=target[len(ENTITY_PREFIX) :],
                        relation=str(data.get("relation", "related_to")),
                        weight=round(float(data.get("weight", 1.0)), 2),
                    )
                )
        return SubGraph(seed_entities=entity_names, nodes=graph_nodes, edges=edges)

    def related_terms(self, text: str, tenant: str, limit: int = 6) -> list[str]:
        """用于 Query 重写：返回与 query 命中实体相邻的术语。"""
        terms: list[str] = []
        for node_id, _ in self.link_entities(text, tenant, limit=3):
            for neighbor in self.graph.successors(node_id):
                if neighbor.startswith(ENTITY_PREFIX):
                    label = str(self.graph.nodes[neighbor].get("label", ""))
                    if label and label not in terms:
                        terms.append(label)
                if len(terms) >= limit:
                    break
            if len(terms) >= limit:
                break
        return terms

    def stats(self) -> dict:
        entities = sum(1 for _, data in self.graph.nodes(data=True) if data.get("kind") == "entity")
        chunks = sum(1 for _, data in self.graph.nodes(data=True) if data.get("kind") == "chunk")
        relations = sum(
            1
            for source, _, data in self.graph.edges(data=True)
            if not source.startswith(CHUNK_PREFIX) and data.get("relation") != "mentions"
        )
        return {"entities": entities, "chunks": chunks, "relations": relations}

    # ---------------- 持久化 ----------------
    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        payload = {
            "nodes": [{"id": node_id, **data} for node_id, data in self.graph.nodes(data=True)],
            "edges": [{"source": s, "target": t, **data} for s, t, data in self.graph.edges(data=True)],
        }
        (directory / "graph.json").write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    def load(self, directory: Path) -> None:
        path = directory / "graph.json"
        if not path.exists():
            return
        payload = json.loads(path.read_text(encoding="utf-8"))
        self.graph.clear()
        for node in payload.get("nodes", []):
            node_id = node.pop("id")
            self.graph.add_node(node_id, **node)
        for edge in payload.get("edges", []):
            source = edge.pop("source")
            target = edge.pop("target")
            relation = edge.pop("relation", "related_to")
            self.graph.add_edge(source, target, relation=relation, **edge)

    def remove_document(self, doc_id: str) -> None:
        chunk_nodes = [
            node_id
            for node_id, data in self.graph.nodes(data=True)
            if data.get("kind") == "chunk" and data.get("doc_id") == doc_id
        ]
        for node_id in chunk_nodes:
            self.graph.remove_node(node_id)
        isolated = [
            node_id
            for node_id, data in self.graph.nodes(data=True)
            if data.get("kind") == "entity" and self.graph.degree(node_id) == 0
        ]
        self.graph.remove_nodes_from(isolated)
