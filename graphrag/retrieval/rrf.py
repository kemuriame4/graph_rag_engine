"""Reciprocal Rank Fusion (RRF)：多路召回的无监督融合。"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class FusionHit:
    chunk_id: str
    rrf_score: float = 0.0
    channels: list[str] = field(default_factory=list)
    channel_scores: dict[str, float] = field(default_factory=dict)
    ranks: dict[str, int] = field(default_factory=dict)


def reciprocal_rank_fusion(
    channel_results: dict[str, list[tuple[str, float]]], k: int = 60
) -> list[FusionHit]:
    """score = Σ 1 / (k + rank)，rank 从 1 开始。

    不依赖各通道分数量纲，是异构召回（Dense / BM25 / Graph）最稳健的融合方式。
    """
    fused: dict[str, FusionHit] = {}
    for channel, results in channel_results.items():
        for rank, (chunk_id, score) in enumerate(results, start=1):
            hit = fused.setdefault(chunk_id, FusionHit(chunk_id=chunk_id))
            hit.rrf_score += 1.0 / (k + rank)
            hit.channels.append(channel)
            hit.channel_scores[channel] = float(score)
            hit.ranks[channel] = rank
    return sorted(fused.values(), key=lambda item: item.rrf_score, reverse=True)
