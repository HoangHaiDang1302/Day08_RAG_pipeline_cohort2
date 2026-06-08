"""
Task 7 - Reranking module.

Default reranking uses the configured OpenAI-compatible LLM provider
(Groq/OpenRouter/OpenAI). Local keyword/MMR rerankers are kept only as fallbacks
so the lab remains runnable when the API is unavailable or rate-limited.
"""

from __future__ import annotations

import math
import json
import os
import re

from src.task4_chunking_indexing import _hashing_embedding, _tokenize

try:
    from dotenv import load_dotenv

    load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env"))
except Exception:
    pass

DEFAULT_RERANK_PROVIDER = os.getenv("RERANK_PROVIDER") or os.getenv("LLM_PROVIDER", "groq")
DEFAULT_RERANK_PROVIDER = DEFAULT_RERANK_PROVIDER.lower()
DEFAULT_RERANK_CANDIDATES = 20


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


def _llm_config() -> tuple[str, str, str | None, str | None]:
    provider = DEFAULT_RERANK_PROVIDER
    if provider == "openrouter":
        return (
            provider,
            os.getenv("OPENROUTER_MODEL", "meta-llama/llama-3.3-70b-instruct:free"),
            os.getenv("OPENROUTER_API_KEY"),
            os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
        )
    if provider == "openai":
        return (
            provider,
            os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            os.getenv("OPENAI_API_KEY"),
            os.getenv("OPENAI_BASE_URL"),
        )
    return (
        "groq",
        os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
        os.getenv("GROQ_API_KEY"),
        os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1"),
    )


def _extract_json(text: str) -> dict | list | None:
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.IGNORECASE | re.MULTILINE)
    for candidate in (
        cleaned,
        cleaned[cleaned.find("{") : cleaned.rfind("}") + 1] if "{" in cleaned and "}" in cleaned else "",
        cleaned[cleaned.find("[") : cleaned.rfind("]") + 1] if "[" in cleaned and "]" in cleaned else "",
    ):
        if not candidate:
            continue
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    return None


def _candidate_prompt(query: str, candidates: list[dict]) -> str:
    rows = []
    for index, item in enumerate(candidates):
        metadata = item.get("metadata", {}) or {}
        source = metadata.get("source") or metadata.get("path") or "unknown"
        content = re.sub(r"\s+", " ", item.get("content", "")).strip()
        rows.append(
            {
                "id": index,
                "source": source,
                "type": metadata.get("type", "unknown"),
                "retrieval_score": float(item.get("score", 0.0) or 0.0),
                "content": content[:1200],
            }
        )

    return (
        "You are reranking retrieval chunks for a Vietnamese RAG system.\n"
        "Rank chunks by usefulness for answering the question. Prefer exact file/title/topic matches, "
        "legal document catalog entries for dataset-level questions, and passages that directly support citations.\n"
        'Return strict JSON only: {"ranked":[{"id":0,"score":0.99,"reason":"short reason"}]}.\n\n'
        f"Question: {query}\n\n"
        f"Candidates:\n{json.dumps(rows, ensure_ascii=False, indent=2)}"
    )


def rerank_llm(query: str, candidates: list[dict], top_k: int = 5) -> list[dict]:
    """
    Rerank candidates with the configured OpenAI-compatible LLM provider.

    If the provider is missing or fails, this returns the keyword fallback with
    error metadata instead of raising.
    """
    provider, model, api_key, base_url = _llm_config()
    if not api_key:
        fallback = rerank_keyword(query, candidates, top_k)
        for item in fallback:
            item["metadata"]["llm_rerank_error"] = f"{provider} API key is missing"
        return fallback

    try:
        from openai import OpenAI
    except Exception as exc:
        fallback = rerank_keyword(query, candidates, top_k)
        for item in fallback:
            item["metadata"]["llm_rerank_error"] = f"OpenAI SDK unavailable: {type(exc).__name__}: {exc}"
        return fallback

    pool = candidates[: max(DEFAULT_RERANK_CANDIDATES, top_k)]
    try:
        client = OpenAI(api_key=api_key, base_url=base_url) if base_url else OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "Return only valid JSON. Do not include markdown fences.",
                },
                {"role": "user", "content": _candidate_prompt(query, pool)},
            ],
            temperature=0,
        )
        parsed = _extract_json(response.choices[0].message.content or "")
    except Exception as exc:
        fallback = rerank_keyword(query, candidates, top_k)
        for item in fallback:
            item["metadata"]["llm_rerank_error"] = f"{type(exc).__name__}: {exc}"
        return fallback

    ranked = parsed.get("ranked", parsed) if isinstance(parsed, dict) else parsed
    if not isinstance(ranked, list):
        fallback = rerank_keyword(query, candidates, top_k)
        for item in fallback:
            item["metadata"]["llm_rerank_error"] = "LLM returned invalid ranking JSON"
        return fallback

    results = []
    used_ids: set[int] = set()
    for rank, row in enumerate(ranked):
        if not isinstance(row, dict):
            continue
        try:
            candidate_id = int(row.get("id"))
        except (TypeError, ValueError):
            continue
        if candidate_id < 0 or candidate_id >= len(pool) or candidate_id in used_ids:
            continue
        item = dict(pool[candidate_id])
        metadata = dict(item.get("metadata", {}))
        metadata["reranker"] = f"llm_{provider}"
        metadata["reranker_model"] = model
        metadata["llm_reason"] = str(row.get("reason", ""))[:240]
        metadata["original_score"] = float(item.get("score", 0.0) or 0.0)
        item["metadata"] = metadata
        try:
            item["score"] = float(row.get("score"))
        except (TypeError, ValueError):
            item["score"] = 1.0 - (rank * 0.01)
        results.append(item)
        used_ids.add(candidate_id)
        if len(results) >= top_k:
            break

    if len(results) < top_k:
        for item in rerank_keyword(query, candidates, top_k=len(candidates)):
            if item.get("content") in {result.get("content") for result in results}:
                continue
            metadata = dict(item.get("metadata", {}))
            metadata["reranker"] = metadata.get("reranker", "keyword_overlap")
            metadata["llm_rerank_fill"] = True
            item["metadata"] = metadata
            results.append(item)
            if len(results) >= top_k:
                break

    return results[:top_k]


def rerank_cross_encoder(query: str, candidates: list[dict], top_k: int = 5) -> list[dict]:
    """
    Cross-encoder compatible entry point.

    This lab uses an OpenAI-compatible LLM as the cross-encoder-style reranker.
    """
    return rerank_llm(query, candidates, top_k)


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
    method: str = "llm",
) -> list[dict]:
    """
    Unified reranking interface.

    Args:
        query: User query.
        candidates: Results from retrieval.
        top_k: Number of results after reranking.
        method: llm | keyword | cross_encoder | mmr
    """
    if top_k <= 0:
        return []
    if not candidates:
        return []

    if method == "llm":
        return rerank_llm(query, candidates, top_k)
    if method == "keyword":
        return rerank_keyword(query, candidates, top_k)
    if method == "cross_encoder":
        return rerank_cross_encoder(query, candidates, top_k)
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
