"""
Task 4 - Chunking and indexing for the RAG corpus.

Design choices:
    - Chunking: recursive character splitting, because the corpus mixes legal
      Markdown and crawled news Markdown. This strategy is robust even when
      headings are inconsistent.
    - Chunk size: 500 chars with 80 chars overlap. Legal Vietnamese text often
      has long clauses, so this keeps chunks small enough for retrieval tests
      while preserving nearby context across boundaries.
    - Embedding: configured OpenAI-compatible embedding API first. Local hashing
      embedding is only a fallback when no embedding key/provider is available.
    - Vector store: local JSON index in data/index/. This keeps Task 4 runnable
      offline and gives Tasks 5-9 a stable source of chunks.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from pathlib import Path
from typing import Iterable

try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).parent.parent / ".env")
except Exception:
    pass

PROJECT_DIR = Path(__file__).parent.parent
STANDARDIZED_DIR = PROJECT_DIR / "data" / "standardized"
INDEX_DIR = PROJECT_DIR / "data" / "index"
CHUNKS_INDEX_PATH = INDEX_DIR / "chunks.json"

# Recursive chunking is the safest default for mixed legal/news markdown.
CHUNK_SIZE = 500
CHUNK_OVERLAP = 80
CHUNKING_METHOD = "recursive"

EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", os.getenv("LLM_PROVIDER", "groq")).lower()
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
LOCAL_EMBEDDING_MODEL = "local-hashing-embedding"
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "384"))

# A local JSON store is enough for tests and can feed later tasks.
VECTOR_STORE = "local_json"


def _doc_type_from_path(path: Path) -> str:
    parts = {part.lower() for part in path.parts}
    if "legal" in parts:
        return "legal"
    if "news" in parts:
        return "news"
    return "unknown"


def load_documents() -> list[dict]:
    """
    Load all Markdown files from data/standardized/.

    Returns:
        List of {'content': str, 'metadata': {'source': str, 'type': str, ...}}
    """
    documents: list[dict] = []
    if not STANDARDIZED_DIR.exists():
        return documents

    for md_file in sorted(STANDARDIZED_DIR.rglob("*.md")):
        if md_file.name.startswith("."):
            continue
        content = md_file.read_text(encoding="utf-8", errors="replace").strip()
        if not content:
            continue
        relative_path = md_file.relative_to(PROJECT_DIR).as_posix()
        documents.append(
            {
                "content": content,
                "metadata": {
                    "source": md_file.name,
                    "path": relative_path,
                    "type": _doc_type_from_path(md_file),
                },
            }
        )
    return documents


def _split_long_text(text: str) -> list[str]:
    """Dependency-free recursive-ish splitter with overlap."""
    separators = ["\n\n", "\n", ". ", "; ", ", ", " "]
    chunks: list[str] = []

    def split_piece(piece: str) -> None:
        piece = piece.strip()
        if not piece:
            return
        if len(piece) <= CHUNK_SIZE:
            chunks.append(piece)
            return

        for sep in separators:
            cut = piece.rfind(sep, 0, CHUNK_SIZE + 1)
            if cut > int(CHUNK_SIZE * 0.45):
                end = cut + len(sep)
                chunks.append(piece[:end].strip())
                next_start = max(0, end - CHUNK_OVERLAP)
                split_piece(piece[next_start:])
                return

        chunks.append(piece[:CHUNK_SIZE].strip())
        next_start = max(0, CHUNK_SIZE - CHUNK_OVERLAP)
        split_piece(piece[next_start:])

    split_piece(text)
    return [chunk for chunk in chunks if chunk]


def _split_with_langchain(text: str) -> list[str] | None:
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
    except Exception:
        return None

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", "; ", ", ", " ", ""],
    )
    return [chunk.strip() for chunk in splitter.split_text(text) if chunk.strip()]


def chunk_documents(documents: list[dict]) -> list[dict]:
    """
    Chunk documents using recursive character splitting.

    Returns:
        List of {'content': str, 'metadata': dict}
    """
    chunks: list[dict] = []
    for doc_index, doc in enumerate(documents):
        content = doc.get("content", "")
        splits = _split_with_langchain(content) or _split_long_text(content)
        for chunk_index, chunk_text in enumerate(splits):
            chunks.append(
                {
                    "content": chunk_text,
                    "metadata": {
                        **doc.get("metadata", {}),
                        "doc_index": doc_index,
                        "chunk_index": chunk_index,
                        "chunking_method": CHUNKING_METHOD,
                        "chunk_size": CHUNK_SIZE,
                        "chunk_overlap": CHUNK_OVERLAP,
                    },
                }
            )
    return chunks


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower(), flags=re.UNICODE)


def _hashing_embedding(tokens: Iterable[str]) -> list[float]:
    vector = [0.0] * EMBEDDING_DIM
    for token in tokens:
        digest = hashlib.md5(token.encode("utf-8")).hexdigest()
        index = int(digest[:8], 16) % EMBEDDING_DIM
        sign = 1.0 if int(digest[8:10], 16) % 2 == 0 else -1.0
        vector[index] += sign

    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        return vector
    return [value / norm for value in vector]


def _embedding_api_config() -> tuple[str, str, str | None, str | None]:
    if EMBEDDING_PROVIDER == "openrouter":
        return (
            "openrouter",
            os.getenv("EMBEDDING_MODEL", os.getenv("OPENROUTER_EMBEDDING_MODEL", EMBEDDING_MODEL)),
            os.getenv("OPENROUTER_API_KEY"),
            os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
        )
    if EMBEDDING_PROVIDER == "openai":
        return (
            "openai",
            os.getenv("EMBEDDING_MODEL", os.getenv("OPENAI_EMBEDDING_MODEL", EMBEDDING_MODEL)),
            os.getenv("OPENAI_API_KEY"),
            os.getenv("OPENAI_BASE_URL"),
        )
    if EMBEDDING_PROVIDER == "groq":
        return (
            "groq",
            os.getenv("EMBEDDING_MODEL", EMBEDDING_MODEL),
            os.getenv("GROQ_API_KEY"),
            os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1"),
        )
    return (
        EMBEDDING_PROVIDER,
        os.getenv("EMBEDDING_MODEL", EMBEDDING_MODEL),
        os.getenv("EMBEDDING_API_KEY"),
        os.getenv("EMBEDDING_BASE_URL"),
    )


def _embed_texts_with_api(texts: list[str]) -> tuple[list[list[float]] | None, str | None, str | None]:
    provider, model, api_key, base_url = _embedding_api_config()
    if not api_key:
        return None, model, f"{provider} embedding API key is missing"

    try:
        from openai import OpenAI
    except Exception as exc:
        return None, model, f"OpenAI SDK unavailable: {type(exc).__name__}: {exc}"

    try:
        client = OpenAI(api_key=api_key, base_url=base_url) if base_url else OpenAI(api_key=api_key)
        response = client.embeddings.create(model=model, input=texts)
        vectors = [item.embedding for item in response.data]
    except Exception as exc:
        return None, model, f"{type(exc).__name__}: {exc}"

    return vectors, model, None


def embed_chunks(chunks: list[dict]) -> list[dict]:
    """
    Add API embeddings to chunks, falling back to deterministic local embeddings.

    Returns:
        Each chunk dict with an added 'embedding': list[float].
    """
    api_vectors, api_model, api_error = _embed_texts_with_api([chunk.get("content", "") for chunk in chunks])
    embedded: list[dict] = []
    for index, chunk in enumerate(chunks):
        item = dict(chunk)
        item["metadata"] = dict(chunk.get("metadata", {}))
        if api_vectors is not None:
            item["embedding"] = api_vectors[index]
            item["metadata"]["embedding_provider"] = EMBEDDING_PROVIDER
            item["metadata"]["embedding_model"] = api_model
            item["metadata"]["embedding_dim"] = len(api_vectors[index])
        else:
            item["embedding"] = _hashing_embedding(_tokenize(chunk.get("content", "")))
            item["metadata"]["embedding_provider"] = "local_fallback"
            item["metadata"]["embedding_model"] = LOCAL_EMBEDDING_MODEL
            item["metadata"]["embedding_dim"] = EMBEDDING_DIM
            item["metadata"]["embedding_error"] = api_error
        embedded.append(item)
    return embedded


def index_to_vectorstore(chunks: list[dict]) -> Path:
    """
    Save chunks into a local JSON vector store.

    The JSON store is intentionally simple: it keeps content, metadata and the
    local embedding so later tasks can implement semantic, lexical and hybrid
    retrieval without a running Weaviate server.
    """
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "vector_store": VECTOR_STORE,
        "embedding_provider": EMBEDDING_PROVIDER,
        "embedding_model": chunks[0]["metadata"].get("embedding_model") if chunks else EMBEDDING_MODEL,
        "embedding_dim": len(chunks[0].get("embedding", [])) if chunks else EMBEDDING_DIM,
        "chunking_method": CHUNKING_METHOD,
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "chunks": chunks,
    }
    CHUNKS_INDEX_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return CHUNKS_INDEX_PATH


def run_pipeline() -> Path:
    """Run the full pipeline: load -> chunk -> embed -> index."""
    print("=" * 50)
    print("Task 4: Chunking & Indexing")
    print(f"  Chunking: {CHUNKING_METHOD} (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    print(f"  Embedding provider: {EMBEDDING_PROVIDER} (model={EMBEDDING_MODEL})")
    print(f"  Vector Store: {VECTOR_STORE}")
    print("=" * 50)

    docs = load_documents()
    print(f"Loaded {len(docs)} documents")

    chunks = chunk_documents(docs)
    print(f"Created {len(chunks)} chunks")

    embedded_chunks = embed_chunks(chunks)
    print(f"Embedded {len(embedded_chunks)} chunks")

    index_path = index_to_vectorstore(embedded_chunks)
    print(f"Indexed to: {index_path}")
    return index_path


if __name__ == "__main__":
    run_pipeline()
