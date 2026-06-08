"""
Task 4 - Chunking and local indexing for the RAG corpus.

Design choices:
    - Chunking: recursive character splitting, because the corpus mixes legal
      Markdown and crawled news Markdown. This strategy is robust even when
      headings are inconsistent.
    - Chunk size: 500 chars with 80 chars overlap. Legal Vietnamese text often
      has long clauses, so this keeps chunks small enough for retrieval tests
      while preserving nearby context across boundaries.
    - Embedding: local hashing embedding, 384 dimensions. It is lightweight and
      deterministic, so the lab can run before sentence-transformers or external
      vector stores are installed. It can later be swapped for BAAI/bge-m3.
    - Vector store: local JSON index in data/index/. This keeps Task 4 runnable
      offline and gives Tasks 5-9 a stable source of chunks.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Iterable

PROJECT_DIR = Path(__file__).parent.parent
STANDARDIZED_DIR = PROJECT_DIR / "data" / "standardized"
INDEX_DIR = PROJECT_DIR / "data" / "index"
CHUNKS_INDEX_PATH = INDEX_DIR / "chunks.json"

# Recursive chunking is the safest default for mixed legal/news markdown.
CHUNK_SIZE = 500
CHUNK_OVERLAP = 80
CHUNKING_METHOD = "recursive"

# Local deterministic embedding keeps the lab runnable without model downloads.
EMBEDDING_MODEL = "local-hashing-embedding"
EMBEDDING_DIM = 384

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
    return re.findall(r"[\wÀ-ỹ]+", text.lower(), flags=re.UNICODE)


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


def embed_chunks(chunks: list[dict]) -> list[dict]:
    """
    Add a deterministic local embedding to each chunk.

    Returns:
        Each chunk dict with an added 'embedding': list[float].
    """
    embedded: list[dict] = []
    for chunk in chunks:
        item = dict(chunk)
        item["metadata"] = dict(chunk.get("metadata", {}))
        item["metadata"]["embedding_model"] = EMBEDDING_MODEL
        item["metadata"]["embedding_dim"] = EMBEDDING_DIM
        item["embedding"] = _hashing_embedding(_tokenize(chunk.get("content", "")))
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
        "embedding_model": EMBEDDING_MODEL,
        "embedding_dim": EMBEDDING_DIM,
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
    print(f"  Embedding: {EMBEDDING_MODEL} (dim={EMBEDDING_DIM})")
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
