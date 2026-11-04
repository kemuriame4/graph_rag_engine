"""轻量文本工具：中英混合分词、句子切分、Token 估算、哈希 ID。"""

from __future__ import annotations

import hashlib
import re
from collections import Counter

CJK_RE = re.compile(r"[\u4e00-\u9fff]+")
WORD_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9_+\-]*|\d+(?:\.\d+)?")
SENTENCE_SPLIT_RE = re.compile(r"(?<=[。！？!?；;])\s*|\n{2,}|(?<=[.!?])\s+")
WHITESPACE_RE = re.compile(r"[ \t\u3000]+")
MULTI_NEWLINE_RE = re.compile(r"\n{3,}")

STOP_CHARS = set("的了和与及或在是有为对从到而并被把这那其之也就都很等吗呢啊嘛吧着过地得所")
PUNCT_CHARS = set("，。！？；：、（）()[]【】“”\"'·—…,.!?;:")
STOPWORDS = STOP_CHARS | set(
    "我 你 他 她 它 们 该 此 更 再 又 或 各 每 以 于 由 至 向 将 已 未 不 没 无 可 需 应 能 会 要 这 那 些 个 种 "
    "the a an of to in for on with and or is are was were be been being this that these those it its as at by from "
    "which what how why when where who whom whose does do did can could should would will shall may might must "
    "how many how much there their they we you i he she".split()
)

# 离线跨语言术语表：在无多语言 Embedding 时提供最小可用的查询扩展。
# 生产环境建议直接使用 bge-m3 / multilingual-e5 等多语言向量模型替代。
BILINGUAL_GLOSSARY: dict[str, str] = {
    "循环寿命": "cycle life",
    "寿命": "lifespan",
    "充电时间": "charge time",
    "充电": "charging",
    "电池": "battery",
    "认证": "certification",
    "关税": "tariff",
    "税率": "tax rate",
    "退货": "return",
    "退款": "refund",
    "报关单": "customs declaration",
    "申报": "declaration",
    "保证金": "deposit",
    "违规": "violation",
    "材料": "materials",
    "审核": "review",
    "流程": "process",
    "时限": "deadline",
    "参数": "specification",
    "规格": "specification",
    "容量": "capacity",
    "功率": "power",
    "保修": "warranty",
    "侵权": "infringement",
    "类目": "category",
    "入驻": "onboarding",
    "卖家": "seller",
    "平台": "platform",
    "安全": "safety",
    "发票": "invoice",
    "单证": "document",
    "海关": "customs",
    "查验": "inspection",
    "放行": "release",
}

_QUESTION_WORDS = {
    "什么", "如何", "怎么", "怎样", "为什么", "哪些", "哪个", "是否", "能否", "多少",
    "what", "how", "why", "which", "when", "where", "who", "does", "do", "can",
}


def normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\x00", "")
    text = WHITESPACE_RE.sub(" ", text)
    text = MULTI_NEWLINE_RE.sub("\n\n", text)
    return text.strip()


def tokenize(text: str) -> list[str]:
    """中英混合分词：英文词 + 中文二元组（+ 单字兜底），过滤停用词。"""
    if not text:
        return []
    lowered = text.lower()
    tokens: list[str] = [w for w in WORD_RE.findall(lowered) if w not in STOPWORDS]
    for seq in CJK_RE.findall(lowered):
        if len(seq) == 1:
            if seq not in STOPWORDS:
                tokens.append(seq)
            continue
        for i in range(len(seq) - 1):
            bigram = seq[i : i + 2]
            if bigram not in STOPWORDS:
                tokens.append(bigram)
    return tokens


def content_words(text: str) -> list[str]:
    """去掉疑问词后的关键词，用于覆盖度/相关性判断。"""
    return [t for t in tokenize(text) if t not in _QUESTION_WORDS and len(t) > 1]


def split_sentences(text: str, min_len: int = 4) -> list[str]:
    if not text:
        return []
    parts = [normalize_text(p) for p in SENTENCE_SPLIT_RE.split(text)]
    return [p for p in parts if len(p) >= min_len]


def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    cjk_chars = sum(len(s) for s in CJK_RE.findall(text))
    latin_words = len(WORD_RE.findall(text))
    return max(1, int(cjk_chars * 1.0 + latin_words * 1.3 + 1))


def sha1_id(*parts: str, size: int = 16) -> str:
    raw = "::".join(parts).encode("utf-8")
    return hashlib.sha1(raw).hexdigest()[:size]


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def keyword_overlap(a: str, b: str) -> float:
    """集合级 Jaccard，用于快速相关性近似。"""
    ta, tb = set(tokenize(a)), set(tokenize(b))
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def coverage(needles: str, haystack: str) -> float:
    """needles 的关键词被 haystack 覆盖的比例。"""
    ta, tb = set(content_words(needles)), set(tokenize(haystack))
    if not ta:
        return 0.0
    return len(ta & tb) / len(ta)


def char_coverage(needles: str, haystack: str) -> float:
    """中文字符级覆盖度：剔除停用字与标点，缓解分词误差。"""
    chars = [
        ch
        for ch in needles
        if "\u4e00" <= ch <= "\u9fff" and ch not in STOP_CHARS
    ]
    if not chars:
        return 0.0
    haystack_chars = set(haystack)
    return sum(1 for ch in chars if ch in haystack_chars) / len(chars)


def expand_bilingual(text: str) -> str:
    """按内置术语表做中英双向扩展，用于离线跨语言检索/选句。"""
    lowered = text.lower()
    remaining = text
    expansions: list[str] = []
    for zh, en in sorted(BILINGUAL_GLOSSARY.items(), key=lambda item: len(item[0]), reverse=True):
        if zh in remaining:
            expansions.append(en)
            remaining = remaining.replace(zh, " ")
        if en in lowered:
            expansions.append(zh)
    return " ".join(dict.fromkeys(expansions))


def top_keywords(text: str, limit: int = 8) -> list[str]:
    counter = Counter(tokenize(text))
    return [w for w, _ in counter.most_common(limit) if len(w) > 1]


def detect_language(text: str) -> str:
    cjk = sum(len(s) for s in CJK_RE.findall(text))
    latin = len(WORD_RE.findall(text))
    if cjk == 0 and latin == 0:
        return "unknown"
    if cjk >= latin * 3:
        return "zh"
    if latin >= cjk * 10:
        return "en"
    return "mixed"


def truncate(text: str, limit: int = 200, ellipsis: str = "…") -> str:
    text = normalize_text(text)
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + ellipsis


def highlight_terms(query: str, text: str) -> list[str]:
    """返回 query 中在 text 里出现的关键词，供前端高亮。"""
    terms: list[str] = []
    for word in content_words(query):
        if len(word) >= 2 and word in text.lower():
            terms.append(word)
    for raw in re.findall(r"[A-Za-z0-9][A-Za-z0-9\-_.]{2,}", query):
        if raw.lower() in text.lower():
            terms.append(raw)
    deduped: list[str] = []
    for term in sorted(terms, key=len, reverse=True):
        if not any(term.lower() in d.lower() for d in deduped):
            deduped.append(term)
    return deduped[:12]


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))
