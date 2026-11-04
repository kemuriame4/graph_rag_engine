"""轻量实体-关系抽取：规则 + 术语模式，可无缝替换为 LLM 抽取器。

无需模型即可复现「实体共现 + 定义式关系」的知识图谱构建流程；开启 LLM 后
可在同一接口下替换为 ``LLMEntityExtractor``（见 selfrag.llm）。
"""

from __future__ import annotations

import re

from ..models import ExtractedEntity, ExtractedRelation, ExtractionResult
from ..text import normalize_text, split_sentences, tokenize

QUOTED_RE = re.compile(r"[“「《\"]([^”」》\"\n]{2,30})[”」》\"]")
DEFINED_RE = re.compile(
    r"([\u4e00-\u9fffA-Za-z0-9·\-]{2,30})\s*[（(]\s*(?:以下)?简称\s*[：:]?\s*([^）)\n]{1,20})\s*[）)]"
)
ACRONYM_RE = re.compile(r"\b([A-Z][A-Z0-9]{1,7})\b")
MODEL_CODE_RE = re.compile(r"\b([A-Z]{2,6}[-_]?\d{2,}[A-Z0-9\-]*)\b")
EN_TERM_RE = re.compile(r"\b([A-Z][A-Za-z0-9]*(?:\s+[A-Z][A-Za-z0-9]*){0,3})\b")
CN_TERM_RE = re.compile(
    r"([\u4e00-\u9fff]{2,12}(?:认证|证书|标准|法规|政策|条款|税号|编码|平台|机构|公司|系统|"
    r"产品|类目|服务|协议|流程|规格|电池|关税|标签|发票|报关单))"
)

TYPE_RULES: list[tuple[str, str]] = [
    ("认证", "certification"), ("证书", "certification"), ("标准", "standard"),
    ("法规", "regulation"), ("政策", "policy"), ("条款", "clause"),
    ("税号", "identifier"), ("编码", "identifier"), ("平台", "platform"),
    ("机构", "organization"), ("公司", "organization"), ("系统", "system"),
    ("产品", "product"), ("类目", "category"), ("服务", "service"),
    ("协议", "agreement"), ("流程", "process"), ("规格", "spec"),
    ("电池", "component"), ("关税", "metric"), ("标签", "label"),
    ("发票", "document"), ("报关单", "document"),
]

RELATION_PATTERNS: list[tuple[str, tuple[str, ...]]] = [
    ("is_a", ("属于", "是一种", "是", "is a", "is an", "belongs to")),
    ("includes", ("包含", "包括", "涵盖", "includes", "contains", "covers")),
    ("requires", ("需要", "必须", "要求", "requires", "must", "needs")),
    ("depends_on", ("依赖", "基于", "依托", "depends on", "based on")),
    ("governed_by", ("依据", "按照", "受", "governed by", "according to")),
]

GENERIC_TERMS = {
    "文档", "内容", "信息", "说明", "要求", "规定", "情况", "相关", "进行", "可以",
    "the", "this", "that", "document", "content", "information", "requirements",
}


class RuleEntityExtractor:
    def __init__(self, max_entities: int = 14, max_pairs: int = 8) -> None:
        self.max_entities = max_entities
        self.max_pairs = max_pairs

    def extract(self, text: str) -> ExtractionResult:
        text = normalize_text(text)
        mentions: dict[str, dict] = {}

        def add(name: str, alias: str | None = None, weight: int = 1) -> str | None:
            name = self._clean(name)
            if not name or not self._valid(name):
                return None
            key = self._normalize(name)
            entry = mentions.setdefault(
                key, {"name": name, "aliases": set(), "count": 0, "type": self._infer_type(name)}
            )
            entry["count"] += weight
            if alias:
                cleaned = self._clean(alias)
                if cleaned and self._valid(cleaned) and self._normalize(cleaned) != key:
                    entry["aliases"].add(cleaned)
                    # 别名也建立独立条目，便于图谱链接
                    alias_key = self._normalize(cleaned)
                    mentions.setdefault(
                        alias_key,
                        {"name": cleaned, "aliases": {name}, "count": weight, "type": entry["type"], "alias_of": key},
                    )
            return key

        for quoted in QUOTED_RE.findall(text):
            add(quoted)

        for main, alias in DEFINED_RE.findall(text):
            key = add(main, alias, weight=3)
            if key:
                alias_key = self._normalize(self._clean(alias))
                if alias_key in mentions:
                    mentions[alias_key]["alias_of"] = key

        for acronym in ACRONYM_RE.findall(text):
            if len(acronym) <= 7:
                add(acronym)

        for code in MODEL_CODE_RE.findall(text):
            add(code, weight=2)

        for term in EN_TERM_RE.findall(text):
            if len(term) >= 4:
                add(term)

        for term in CN_TERM_RE.findall(text):
            add(term)

        for sentence in split_sentences(text):
            if "是指" in sentence or "指的是" in sentence or "定义为" in sentence:
                subject = sentence.split("是指")[0].split("指的是")[0].split("定义为")[0]
                add(subject.strip("，,。 "), weight=2)

        ordered = sorted(mentions.items(), key=lambda item: item[1]["count"], reverse=True)
        entities = [
            ExtractedEntity(
                name=data["name"],
                normalized=key,
                type=data["type"],
                aliases=sorted(data["aliases"]),
                mentions=data["count"],
            )
            for key, data in ordered[: self.max_entities]
            if self._valid(key)
        ]

        entity_names = [entity.name for entity in entities]
        relations: list[ExtractedRelation] = []
        seen: set[tuple[str, str, str]] = set()

        for entity in entities:
            alias_of = mentions.get(entity.normalized, {}).get("alias_of")
            if alias_of:
                relations.append(
                    ExtractedRelation(source=entity.normalized, target=alias_of, relation="alias_of", weight=1.5)
                )

        for sentence in split_sentences(text):
            present = [name for name in entity_names if name.lower() in sentence.lower()]
            present = present[: self.max_pairs]
            for i, left in enumerate(present):
                for right in present[i + 1 :]:
                    relation = self._relation_between(sentence, left, right)
                    source, target = self._normalize(left), self._normalize(right)
                    if relation == "co_occurs" and source == target:
                        continue
                    key = (source, target, relation)
                    if key in seen:
                        continue
                    seen.add(key)
                    relations.append(
                        ExtractedRelation(
                            source=source,
                            target=target,
                            relation=relation,
                            weight=1.0 if relation != "co_occurs" else 0.6,
                        )
                    )

        return ExtractionResult(entities=entities, relations=relations)

    def _relation_between(self, sentence: str, left: str, right: str) -> str:
        lower = sentence.lower()
        left_index = lower.find(left.lower())
        right_index = lower.find(right.lower(), left_index + 1)
        if left_index < 0 or right_index < 0:
            return "co_occurs"
        between = lower[left_index + len(left) : right_index]
        for relation, keywords in RELATION_PATTERNS:
            if any(keyword in between for keyword in keywords):
                return relation
        return "co_occurs"

    @staticmethod
    def _clean(name: str) -> str:
        name = normalize_text(name).strip("，,。.;；:：、（）()[]【】 ")
        name = re.sub(r"^(?:其中|此外|同时|并且|以及|以及该|该)+", "", name)
        return name.strip()

    @staticmethod
    def _normalize(name: str) -> str:
        return re.sub(r"\s+", "", name.lower())

    @staticmethod
    def _valid(name: str) -> bool:
        if len(name) < 2 or len(name) > 40:
            return False
        if name.lower() in GENERIC_TERMS:
            return False
        if not any(ch.isalnum() for ch in name):
            return False
        tokens = tokenize(name)
        return bool(tokens)

    @staticmethod
    def _infer_type(name: str) -> str:
        for suffix, entity_type in TYPE_RULES:
            if name.endswith(suffix):
                return entity_type
        if re.fullmatch(r"[A-Z0-9\-]{2,10}", name):
            return "acronym"
        if name[0].isupper() and name.replace(" ", "").isalpha():
            return "term"
        return "term"
