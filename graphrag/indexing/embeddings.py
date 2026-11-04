"""Embedding 适配层：真实模型 / OpenAI 兼容 API / 确定性哈希降级。"""

from __future__ import annotations

import hashlib
import math

import numpy as np

from ..config import Settings
from ..text import CJK_RE, WORD_RE, tokenize


class BaseEmbedder:
    name: str = "base"
    dim: int = 256

    def embed(self, texts: list[str]) -> np.ndarray:
        raise NotImplementedError

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed([text])[0]

    @staticmethod
    def _normalize(matrix: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return matrix / norms


class HashingEmbedder(BaseEmbedder):
    """无外部依赖的特征哈希向量化。

    中文采用字级 2/3-gram，英文采用词级 + 字符 4-gram；使用 blake2b 做
    跨进程稳定的 signed hashing，保证索引与查询在不同进程下完全一致。
    """

    name = "hashing"

    def __init__(self, dim: int = 256) -> None:
        self.dim = dim

    def _features(self, text: str) -> list[tuple[str, float]]:
        lowered = text.lower()
        features: list[tuple[str, float]] = [(token, 1.0) for token in tokenize(lowered)]
        for seq in CJK_RE.findall(lowered):
            for n in (2, 3):
                for i in range(max(0, len(seq) - n + 1)):
                    features.append((seq[i : i + n], 1.25))
        for word in WORD_RE.findall(lowered):
            if len(word) < 4:
                continue
            padded = f"<{word}>"
            for i in range(len(padded) - 3):
                features.append((padded[i : i + 4], 0.6))
        return features

    def embed(self, texts: list[str]) -> np.ndarray:
        matrix = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for feature, weight in self._features(text):
                digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
                index = int.from_bytes(digest[:4], "little") % self.dim
                sign = 1.0 if digest[4] & 1 else -1.0
                matrix[row, index] += sign * weight
        return self._normalize(matrix)


class OpenAIEmbedder(BaseEmbedder):
    name = "openai"

    def __init__(self, settings: Settings) -> None:
        import httpx

        self._client = httpx.Client(
            base_url=settings.llm_base_url.rstrip("/"),
            headers={"Authorization": f"Bearer {settings.llm_api_key}"},
            timeout=settings.llm_timeout_seconds,
        )
        self.model = settings.llm_embed_model
        self.dim = 0

    def embed(self, texts: list[str]) -> np.ndarray:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), 64):
            batch = texts[start : start + 64]
            response = self._client.post("/embeddings", json={"model": self.model, "input": batch})
            response.raise_for_status()
            payload = response.json()
            vectors.extend(item["embedding"] for item in payload["data"])
        matrix = np.asarray(vectors, dtype=np.float32)
        if not self.dim:
            self.dim = int(matrix.shape[1])
        return self._normalize(matrix)


class SentenceTransformerEmbedder(BaseEmbedder):
    name = "sentence-transformers"

    def __init__(self, model_name: str) -> None:
        from sentence_transformers import SentenceTransformer  # type: ignore[import-not-found]

        self.model = SentenceTransformer(model_name)
        self.dim = int(self.model.get_sentence_embedding_dimension())

    def embed(self, texts: list[str]) -> np.ndarray:
        vectors = self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        return np.asarray(vectors, dtype=np.float32)


class ResilientEmbedder(BaseEmbedder):
    """主后端失败时自动切换到哈希后端，避免整条链路不可用。"""

    def __init__(self, primary: BaseEmbedder, fallback: BaseEmbedder) -> None:
        self.primary = primary
        self.fallback = fallback
        self.name = f"{primary.name}+fallback"
        self.dim = primary.dim or fallback.dim

    def embed(self, texts: list[str]) -> np.ndarray:
        try:
            matrix = self.primary.embed(texts)
            self.dim = int(matrix.shape[1])
            return matrix
        except Exception:  # noqa: BLE001 - 外部服务不可用时保证可用性
            self.name = "hashing"
            self.dim = self.fallback.dim
            return self.fallback.embed(texts)


def build_embedder(settings: Settings) -> BaseEmbedder:
    backend = settings.embedding_backend
    hashing = HashingEmbedder(settings.embedding_dim)
    if backend == "hashing":
        return hashing
    if backend == "openai":
        return OpenAIEmbedder(settings)
    if backend == "sentence-transformers":
        try:
            return SentenceTransformerEmbedder(settings.st_model)
        except Exception:  # noqa: BLE001
            return hashing
    # auto
    if settings.llm_api_key:
        primary = OpenAIEmbedder(settings)
        return ResilientEmbedder(primary, hashing)
    try:
        return SentenceTransformerEmbedder(settings.st_model)
    except Exception:  # noqa: BLE001
        return hashing


def cosine_scores(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    if matrix.size == 0:
        return np.zeros((0,), dtype=np.float32)
    denom = np.linalg.norm(matrix, axis=1) * float(np.linalg.norm(query) + 1e-9)
    return (matrix @ query) / np.maximum(denom, 1e-9)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b)) + 1e-9
    return float(np.dot(a, b) / denominator)


def cosine_similarity_text(embedder: BaseEmbedder, a: str, b: str) -> float:
    if not a.strip() or not b.strip():
        return 0.0
    vectors = embedder.embed([a, b])
    return cosine(vectors[0], vectors[1])


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))
