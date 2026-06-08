"""
Task 7 - Reranking module.

The README allows self-implemented rerankers such as MMR and RRF. This file
keeps a no-API default so the lab runs locally:
    - keyword reranker: combines original retrieval score with query-term overlap
    - RRF: combines multiple ranked lists
    - MMR: selects relevant but less-duplicated candidates when embeddings exist
"""

from __future__ import annotations

import math

from src.task4_chunking_indexing import _hashing_embedding, _tokenize


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)


def _keyword_relevance(query: str, content: str) -> float:
    query_tokens = set(_tokenize(query))
    if not query_tokens:
        return 0.0
    content_tokens = _tokenize(content)
    if not content_tokens:
        return 0.0
    matched = sum(1 for token in content_tokens if token in query_tokens)
    coverage = len(query_tokens.intersection(content_tokens)) / len(query_tokens)
    density = matched / len(content_tokens)
    return coverage + density


def rerank_keyword(query: str, candidates: list[dict], top_k: int = 5) -> list[dict]:
    """
    Lightweight local reranker.

    It preserves each candidate's original retrieval score but boosts passages
    whose text directly overlaps with the query. This is deterministic and works
    well enough for legal/news retrieval without calling an external reranker API.
    """
    reranked = []
    for rank, candidate in enumerate(candidates):
        item = dict(candidate)
        original_score = float(item.get("score", 0.0) or 0.0)
        keyword_score = _keyword_relevance(query, item.get("content", ""))
        rank_bonus = 1.0 / (rank + 1)
        item["score"] = 0.55 * original_score + 0.40 * keyword_score + 0.05 * rank_bonus
        metadata = dict(item.get("metadata", {}))
        metadata["reranker"] = "keyword_overlap"
        metadata["original_score"] = original_score
        metadata["keyword_score"] = keyword_score
        item["metadata"] = metadata
        reranked.append(item)

    reranked.sort(key=lambda result: result["score"], reverse=True)
    return reranked[: max(top_k, 0)]


def rerank_cross_encoder(query: str, candidates: list[dict], top_k: int = 5) -> list[dict]:
    """
    Cross-encoder compatible entry point.

    In this local lab environment no API key or heavy model is required; the
    function falls back to keyword reranking while keeping the same signature.
    """
    return rerank_keyword(query, candidates, top_k)


def rerank_mmr(
    query_embedding: list[float],
    candidates: list[dict],
    top_k: int = 5,
    lambda_param: float = 0.7,
) -> list[dict]:
    """
    Maximal Marginal Relevance:
        MMR = lambda * relevance - (1 - lambda) * similarity_to_selected
    """
    if top_k <= 0 or not candidates:
        return []

    prepared = []
    for candidate in candidates:
        item = dict(candidate)
        item["metadata"] = dict(candidate.get("metadata", {}))
        if not item.get("embedding"):
            item["embedding"] = _hashing_embedding(_tokenize(item.get("content", "")))
        prepared.append(item)

    selected_indices: list[int] = []
    remaining = set(range(len(prepared)))

    while remaining and len(selected_indices) < top_k:
        best_index = None
        best_score = float("-inf")

        for index in remaining:
            candidate = prepared[index]
            relevance = _cosine_similarity(query_embedding, candidate.get("embedding", []))
            max_similarity = 0.0
            for selected_index in selected_indices:
                max_similarity = max(
                    max_similarity,
                    _cosine_similarity(
                        candidate.get("embedding", []),
                        prepared[selected_index].get("embedding", []),
                    ),
                )
            mmr_score = lambda_param * relevance - (1 - lambda_param) * max_similarity
            if mmr_score > best_score:
                best_score = mmr_score
                best_index = index

        if best_index is None:
            break

        prepared[best_index]["score"] = float(best_score)
        prepared[best_index]["metadata"]["reranker"] = "mmr"
        selected_indices.append(best_index)
        remaining.remove(best_index)

    return [prepared[index] for index in selected_indices]


def rerank_rrf(ranked_lists: list[list[dict]], top_k: int = 5, k: int = 60) -> list[dict]:
    """
    Reciprocal Rank Fusion:
        RRF(d) = sum(1 / (k + rank_r(d)))
    """
    scores: dict[str, float] = {}
    items: dict[str, dict] = {}

    for ranked_list in ranked_lists:
        for rank, item in enumerate(ranked_list, 1):
            key = item.get("content", "")
            if not key:
                continue
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
            items[key] = item

    results = []
    for content, score in sorted(scores.items(), key=lambda entry: entry[1], reverse=True)[:top_k]:
        item = dict(items[content])
        metadata = dict(item.get("metadata", {}))
        metadata["reranker"] = "rrf"
        item["metadata"] = metadata
        item["score"] = float(score)
        results.append(item)
    return results


def rerank(
    query: str,
    candidates: list[dict],
    top_k: int = 5,
    method: str = "keyword",
) -> list[dict]:
    """
    Unified reranking interface.

    Args:
        query: User query.
        candidates: Results from retrieval.
        top_k: Number of results after reranking.
        method: keyword | cross_encoder | mmr
    """
    if top_k <= 0:
        return []
    if not candidates:
        return []

    if method in {"keyword", "cross_encoder"}:
        return rerank_keyword(query, candidates, top_k)
    if method == "mmr":
        query_embedding = _hashing_embedding(_tokenize(query))
        return rerank_mmr(query_embedding, candidates, top_k)
    raise ValueError(f"Unknown rerank method: {method}")


if __name__ == "__main__":
    dummy_candidates = [
        {"content": "Dieu 248: Toi tang tru trai phep chat ma tuy", "score": 0.8, "metadata": {}},
        {"content": "Nghe si bi bat vi su dung ma tuy", "score": 0.7, "metadata": {}},
        {"content": "Hinh phat tu 2-7 nam cho toi tang tru", "score": 0.6, "metadata": {}},
    ]
    for result in rerank("hinh phat tang tru ma tuy", dummy_candidates, top_k=2):
        print(f"[{result['score']:.3f}] {result['content']}")
