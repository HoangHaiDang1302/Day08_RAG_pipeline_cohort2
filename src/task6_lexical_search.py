"""
Task 6 - Lexical search with BM25.

BM25 ranks chunks by exact keyword overlap while normalizing document length.
This file uses the local chunk index from Task 4, so it does not need a running
database. If rank-bm25 is installed, the same interface can be swapped in later;
the implementation below keeps the lab runnable without extra packages.
"""

from __future__ import annotations

import json
import math
from collections import Counter

from src.task4_chunking_indexing import CHUNKS_INDEX_PATH, _tokenize, run_pipeline

K1 = 1.5
B = 0.75


def _load_corpus() -> list[dict]:
    if not CHUNKS_INDEX_PATH.exists():
        run_pipeline()
    payload = json.loads(CHUNKS_INDEX_PATH.read_text(encoding="utf-8"))
    chunks = payload.get("chunks", [])
    if not isinstance(chunks, list):
        return []
    return [
        {
            "content": chunk.get("content", ""),
            "metadata": chunk.get("metadata", {}),
        }
        for chunk in chunks
    ]


CORPUS: list[dict] = _load_corpus()


class SimpleBM25:
    """Small BM25 Okapi implementation for deterministic local lexical search."""

    def __init__(self, tokenized_corpus: list[list[str]]) -> None:
        self.tokenized_corpus = tokenized_corpus
        self.doc_freqs = [Counter(doc) for doc in tokenized_corpus]
        self.doc_lengths = [len(doc) for doc in tokenized_corpus]
        self.avg_doc_length = (
            sum(self.doc_lengths) / len(self.doc_lengths) if self.doc_lengths else 0.0
        )
        self.idf = self._compute_idf()

    def _compute_idf(self) -> dict[str, float]:
        doc_count = len(self.tokenized_corpus)
        document_frequency: Counter[str] = Counter()
        for doc in self.tokenized_corpus:
            document_frequency.update(set(doc))

        idf = {}
        for term, freq in document_frequency.items():
            idf[term] = math.log(1 + (doc_count - freq + 0.5) / (freq + 0.5))
        return idf

    def get_scores(self, query_tokens: list[str]) -> list[float]:
        scores = []
        for doc_freq, doc_length in zip(self.doc_freqs, self.doc_lengths):
            score = 0.0
            for term in query_tokens:
                term_frequency = doc_freq.get(term, 0)
                if term_frequency == 0:
                    continue
                idf = self.idf.get(term, 0.0)
                denominator = term_frequency + K1 * (
                    1 - B + B * doc_length / (self.avg_doc_length or 1.0)
                )
                score += idf * (term_frequency * (K1 + 1)) / denominator
            scores.append(score)
        return scores


def build_bm25_index(corpus: list[dict]) -> SimpleBM25:
    """
    Build a BM25 index from a list of {'content': str, 'metadata': dict}.
    """
    tokenized_corpus = [_tokenize(doc.get("content", "")) for doc in corpus]
    return SimpleBM25(tokenized_corpus)


def lexical_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Search chunks by BM25 keyword matching.

    Returns:
        List of {'content': str, 'score': float, 'metadata': dict}, sorted by
        score descending.
    """
    if top_k <= 0:
        return []

    corpus = CORPUS or _load_corpus()
    if not corpus:
        return []

    bm25 = build_bm25_index(corpus)
    query_tokens = _tokenize(query)
    scores = bm25.get_scores(query_tokens)

    ranked = sorted(enumerate(scores), key=lambda item: item[1], reverse=True)
    results = []
    for index, score in ranked[:top_k]:
        results.append(
            {
                "content": corpus[index]["content"],
                "score": float(score),
                "metadata": corpus[index].get("metadata", {}),
            }
        )
    return results


if __name__ == "__main__":
    for result in lexical_search("Dieu 248 tang tru trai phep chat ma tuy", top_k=5):
        print(f"[{result['score']:.3f}] {result['content'][:100]}...")
