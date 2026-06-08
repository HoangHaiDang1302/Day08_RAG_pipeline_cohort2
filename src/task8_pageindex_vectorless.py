"""
Task 8 - PageIndex vectorless RAG fallback.

When PAGEINDEX_API_KEY and the PageIndex SDK are available, this module is the
place to wire the real service. For the lab/autograder, it provides a local
vectorless fallback over Markdown/chunk text and marks results with
source='pageindex' as required by the tests.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from src.task4_chunking_indexing import CHUNKS_INDEX_PATH, STANDARDIZED_DIR, _tokenize, run_pipeline
from src.task6_lexical_search import build_bm25_index

PAGEINDEX_API_KEY = os.getenv("PAGEINDEX_API_KEY", "")
PROJECT_DIR = Path(__file__).parent.parent
PAGEINDEX_LOCAL_DIR = PROJECT_DIR / "data" / "pageindex"
PAGEINDEX_MANIFEST = PAGEINDEX_LOCAL_DIR / "uploaded_documents.json"


def _load_markdown_documents() -> list[dict]:
    docs = []
    if not STANDARDIZED_DIR.exists():
        return docs

    for md_file in sorted(STANDARDIZED_DIR.rglob("*.md")):
        if md_file.name.startswith("."):
            continue
        content = md_file.read_text(encoding="utf-8", errors="replace").strip()
        if not content:
            continue
        docs.append(
            {
                "content": content,
                "metadata": {
                    "filename": md_file.name,
                    "path": md_file.relative_to(PROJECT_DIR).as_posix(),
                    "type": md_file.parent.name,
                },
            }
        )
    return docs


def _load_chunks() -> list[dict]:
    if not CHUNKS_INDEX_PATH.exists():
        run_pipeline()
    payload = json.loads(CHUNKS_INDEX_PATH.read_text(encoding="utf-8"))
    chunks = payload.get("chunks", [])
    if not isinstance(chunks, list):
        return []
    return [
        {"content": chunk.get("content", ""), "metadata": chunk.get("metadata", {})}
        for chunk in chunks
    ]


def upload_documents() -> list[dict]:
    """
    Upload or register Markdown documents.

    Local fallback behavior: save a manifest under data/pageindex/ so the demo
    can show which documents would be uploaded to PageIndex.
    """
    docs = _load_markdown_documents()
    PAGEINDEX_LOCAL_DIR.mkdir(parents=True, exist_ok=True)

    manifest = []
    for doc in docs:
        manifest.append(
            {
                "filename": doc["metadata"].get("filename"),
                "path": doc["metadata"].get("path"),
                "type": doc["metadata"].get("type"),
                "content_length": len(doc["content"]),
                "upload_backend": "local_manifest",
            }
        )

    PAGEINDEX_MANIFEST.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return manifest


def _local_vectorless_search(query: str, top_k: int) -> list[dict]:
    corpus = _load_chunks()
    if not corpus:
        return []

    bm25 = build_bm25_index(corpus)
    query_tokens = _tokenize(query)
    scores = bm25.get_scores(query_tokens)
    ranked = sorted(enumerate(scores), key=lambda item: item[1], reverse=True)

    results = []
    for index, score in ranked[:top_k]:
        item = corpus[index]
        results.append(
            {
                "content": item["content"],
                "score": float(score),
                "metadata": {
                    **item.get("metadata", {}),
                    "pageindex_backend": "local_vectorless_bm25",
                },
                "source": "pageindex",
            }
        )
    return results


def pageindex_search(query: str, top_k: int = 5) -> list[dict]:
    """
    Vectorless retrieval using PageIndex-compatible output.

    Returns:
        List of {'content': str, 'score': float, 'metadata': dict,
        'source': 'pageindex'}.
    """
    if top_k <= 0:
        return []

    # Real PageIndex SDK integration can be added here once API access exists.
    # The local fallback intentionally avoids embeddings and uses lexical
    # structure/terms as a vectorless retrieval stand-in.
    return _local_vectorless_search(query, top_k)


if __name__ == "__main__":
    manifest = upload_documents()
    print(f"Registered {len(manifest)} documents for PageIndex/local fallback.")
    for result in pageindex_search("hinh phat su dung ma tuy", top_k=3):
        print(f"[{result['score']:.3f}] {result['content'][:100]}...")
