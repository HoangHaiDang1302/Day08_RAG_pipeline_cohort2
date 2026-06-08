"""
Task 9 - Complete retrieval pipeline.

Pipeline:
    1. Run semantic search and lexical search.
    2. Merge both ranked lists with Reciprocal Rank Fusion (RRF).
    3. Rerank merged candidates.
    4. If the best hybrid score is below threshold, fallback to PageIndex.
"""

from __future__ import annotations

from src.task5_semantic_search import semantic_search
from src.task6_lexical_search import lexical_search
from src.task7_reranking import rerank, rerank_rrf
from src.task8_pageindex_vectorless import pageindex_search

SCORE_THRESHOLD = 0.0
DEFAULT_TOP_K = 5
RERANK_METHOD = "llm"


def _tag_results(results: list[dict], retriever: str) -> list[dict]:
    tagged = []
    for result in results:
        item = dict(result)
        metadata = dict(item.get("metadata", {}))
        metadata["retriever"] = retriever
        item["metadata"] = metadata
        tagged.append(item)
    return tagged


def _mark_hybrid(results: list[dict]) -> list[dict]:
    marked = []
    for result in results:
        item = dict(result)
        metadata = dict(item.get("metadata", {}))
        metadata["pipeline"] = "semantic+lexical+rerank"
        item["metadata"] = metadata
        item["source"] = "hybrid"
        marked.append(item)
    return marked


def retrieve(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    score_threshold: float = SCORE_THRESHOLD,
    use_reranking: bool = True,
) -> list[dict]:
    """
    Complete retrieval pipeline with fallback logic.

    Returns:
        List of {'content': str, 'score': float, 'metadata': dict,
        'source': 'hybrid' | 'pageindex'}.
    """
    if top_k <= 0:
        return []

    search_k = max(top_k * 5, 20)
    dense_results = _tag_results(semantic_search(query, top_k=search_k), "semantic")
    sparse_results = _tag_results(lexical_search(query, top_k=search_k), "lexical")

    merged = rerank_rrf([dense_results, sparse_results], top_k=search_k)
    merged = _mark_hybrid(merged)

    if use_reranking and merged:
        final_results = rerank(query, merged, top_k=top_k, method=RERANK_METHOD)
        final_results = _mark_hybrid(final_results)
    else:
        final_results = merged[:top_k]

    best_score = final_results[0]["score"] if final_results else 0.0
    if not final_results or best_score < score_threshold:
        return pageindex_search(query, top_k=top_k)

    return final_results[:top_k]


if __name__ == "__main__":
    test_queries = [
        "Hinh phat cho toi tang tru trai phep chat ma tuy",
        "Nghe si nao bi bat vi su dung ma tuy nam 2024",
        "Luat phong chong ma tuy 2021 quy dinh gi ve cai nghien",
    ]

    for question in test_queries:
        print(f"\nQuery: {question}")
        print("-" * 60)
        for index, result in enumerate(retrieve(question, top_k=3), 1):
            print(f"  {index}. [{result['score']:.3f}] [{result['source']}] {result['content'][:80]}...")
