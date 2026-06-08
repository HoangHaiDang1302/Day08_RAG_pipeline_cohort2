"""
Task 5 - Semantic search over the vector index.

Query embeddings use the configured API embedding provider first and fall back
to local hashing only when the API is unavailable.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from src.task4_chunking_indexing import (
    CHUNKS_INDEX_PATH,
    _embed_texts_with_api,
    _hashing_embedding,
    _tokenize,
    run_pipeline,
)


def _load_index() -> list[dict]:
    if not CHUNKS_INDEX_PATH.exists():
        run_pipeline()

    payload = json.loads(CHUNKS_INDEX_PATH.read_text(encoding="utf-8"))
    chunks = payload.get("chunks", [])
    if not isinstance(chunks, list):
        return []
    return chunks


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)


def semantic_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Search chunks by vector similarity.

    Args:
        query: User query.
        top_k: Maximum number of results.

    Returns:
        List of {'content': str, 'score': float, 'metadata': dict}, sorted by
        score descending.
    """
    if top_k <= 0:
        return []

    chunks = _load_index()
    expected_dim = len(chunks[0].get("embedding", [])) if chunks else 0

    api_vectors, _, _ = _embed_texts_with_api([query])
    query_embedding = api_vectors[0] if api_vectors else _hashing_embedding(_tokenize(query))
    if expected_dim and len(query_embedding) != expected_dim:
        query_embedding = _hashing_embedding(_tokenize(query))
    results = []

    for chunk in chunks:
        chunk_embedding = chunk.get("embedding") or []
        score = _cosine_similarity(query_embedding, chunk_embedding)
        results.append(
            {
                "content": chunk.get("content", ""),
                "score": float(score),
                "metadata": chunk.get("metadata", {}),
            }
        )

    results.sort(key=lambda item: item["score"], reverse=True)
    return results[:top_k]


if __name__ == "__main__":
    for result in semantic_search("hinh phat cho toi tang tru ma tuy", top_k=5):
        print(f"[{result['score']:.3f}] {result['content'][:100]}...")
