"""
Task 10 - Generation with citations.

This local implementation does not require an LLM API key. It retrieves context,
reorders chunks to reduce "lost in the middle", formats source labels, and
builds a grounded Vietnamese answer with citations from the retrieved chunks.

When an OpenAI/Gemini key is available, the answer-construction part can be
replaced with an LLM call while keeping reorder_for_llm() and format_context().
"""

from __future__ import annotations

import os
import re

from src.task9_retrieval_pipeline import retrieve

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    pass

# top_k=5 gives enough evidence without making the prompt too long.
TOP_K = 5

# top_p=0.9 would be used for API generation: diverse but still controlled.
TOP_P = 0.9

# temperature=0.3 is appropriate for factual RAG because it reduces randomness.
TEMPERATURE = 0.3

SYSTEM_PROMPT = """Answer the following question in Vietnamese.
Use only the provided context.
Every factual claim must include a source citation.
If the context is insufficient, say: I cannot verify this information."""

DEFAULT_LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq").lower()
DEFAULT_LLM_MODEL = os.getenv("GROQ_MODEL") or os.getenv("OPENAI_MODEL") or "llama-3.3-70b-versatile"
GROQ_BASE_URL = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")


def reorder_for_llm(chunks: list[dict]) -> list[dict]:
    """
    Reorder chunks to reduce "lost in the middle".

    Input sorted by score: [1, 2, 3, 4, 5]
    Output pattern:        [1, 3, 5, 4, 2]

    The best chunk stays first, the second-best is moved near the end, and lower
    confidence chunks stay closer to the middle.
    """
    if len(chunks) <= 2:
        return chunks

    front = chunks[0::2]
    back = list(reversed(chunks[1::2]))
    return front + back


def _source_label(chunk: dict, index: int) -> str:
    metadata = chunk.get("metadata", {}) or {}
    source = (
        metadata.get("source")
        or metadata.get("filename")
        or metadata.get("path")
        or f"Source {index}"
    )
    return str(source)


def format_context(chunks: list[dict]) -> str:
    """
    Format chunks into a context string with source labels for citation.
    """
    context_parts = []
    for index, chunk in enumerate(chunks, 1):
        metadata = chunk.get("metadata", {}) or {}
        source = _source_label(chunk, index)
        doc_type = metadata.get("type", "unknown")
        score = float(chunk.get("score", 0.0) or 0.0)
        context_parts.append(
            f"[Document {index} | Source: {source} | Type: {doc_type} | Score: {score:.3f}]\n"
            f"{chunk.get('content', '').strip()}"
        )
    return "\n\n---\n\n".join(context_parts)


def _sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?。])\s+|\n+", text.strip())
    return [part.strip() for part in parts if len(part.strip()) > 40]


def _build_local_answer(query: str, chunks: list[dict]) -> str:
    if not chunks:
        return "I cannot verify this information"

    answer_lines = [
        f"Dựa trên các nguồn đã truy xuất, câu hỏi '{query}' có thể được trả lời như sau:"
    ]

    used = 0
    for index, chunk in enumerate(chunks, 1):
        source = _source_label(chunk, index)
        sentence_pool = _sentences(chunk.get("content", ""))
        if not sentence_pool:
            continue
        sentence = sentence_pool[0]
        if len(sentence) > 320:
            sentence = sentence[:317].rstrip() + "..."
        answer_lines.append(f"- {sentence} [{source}]")
        used += 1
        if used >= 3:
            break

    if used == 0:
        return "I cannot verify this information"

    answer_lines.append(
        "Các nhận định trên chỉ dựa trên những đoạn context được truy xuất; "
        "những thông tin không xuất hiện trong context cần được kiểm chứng thêm. "
        "[retrieval_context]"
    )
    return "\n".join(answer_lines)


def _generate_with_openai_compatible(query: str, context: str) -> tuple[str | None, str | None]:
    """
    Call an LLM for true abstractive generation when OPENAI_API_KEY is available.

    Returns (answer, error). Answer is None when SDK/key/network/quota is
    unavailable so callers can use the local extractive fallback.
    """
    if DEFAULT_LLM_PROVIDER == "groq":
        api_key = os.getenv("GROQ_API_KEY")
        base_url = GROQ_BASE_URL
        missing_key_message = "GROQ_API_KEY is missing"
    else:
        api_key = os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("OPENAI_BASE_URL")
        missing_key_message = "OPENAI_API_KEY is missing"

    if not api_key:
        return None, missing_key_message

    try:
        from openai import OpenAI
    except Exception as exc:
        return None, f"OpenAI SDK unavailable: {type(exc).__name__}: {exc}"

    user_prompt = f"""Context:
{context}

---

Question: {query}

Please answer in Vietnamese. Cite every factual claim using the Source labels
shown in the context, for example [luat-phong-chong-ma-tuy-2021.md]. If the
context does not contain enough evidence, say: I cannot verify this information."""

    try:
        client = OpenAI(api_key=api_key)
        if base_url:
            client = OpenAI(api_key=api_key, base_url=base_url)
        response = client.chat.completions.create(
            model=DEFAULT_LLM_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            top_p=TOP_P,
        )
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"

    return response.choices[0].message.content or None, None


def generate_with_citation(query: str, top_k: int = TOP_K, use_llm: bool | None = None) -> dict:
    """
    End-to-end RAG generation with citation.

    Returns:
        {
            'answer': str,
            'sources': list[dict],
            'retrieval_source': str
        }
    """
    if use_llm is None:
        use_llm = os.getenv("ENABLE_LLM_GENERATION", "").lower() in {"1", "true", "yes"}

    chunks = retrieve(query, top_k=top_k)
    reordered = reorder_for_llm(chunks)
    context = format_context(reordered)
    llm_answer, llm_error = _generate_with_openai_compatible(query, context) if use_llm else (None, None)
    answer = llm_answer or _build_local_answer(query, reordered)

    return {
        "answer": answer,
        "sources": reordered,
        "retrieval_source": chunks[0].get("source", "none") if chunks else "none",
        "context": context,
        "generation_config": {
            "top_k": top_k,
            "top_p": TOP_P,
            "temperature": TEMPERATURE,
            "mode": f"{DEFAULT_LLM_PROVIDER}_chat_completion" if llm_answer else "local_extractive_fallback",
            "provider": DEFAULT_LLM_PROVIDER if llm_answer else None,
            "model": DEFAULT_LLM_MODEL if llm_answer else None,
            "llm_error": llm_error,
        },
    }


if __name__ == "__main__":
    questions = [
        "Hinh phat cho toi tang tru trai phep chat ma tuy theo phap luat Viet Nam?",
        "Nhung nghe si nao da bi bat vi lien quan toi ma tuy?",
        "Quy trinh cai nghien bat buoc theo Luat Phong chong ma tuy 2021?",
    ]

    for question in questions:
        print(f"\n{'=' * 70}")
        print(f"Q: {question}")
        print("=" * 70)
        result = generate_with_citation(question)
        print(f"\nA: {result['answer']}")
        print(f"\n[Sources: {len(result['sources'])} chunks | via {result['retrieval_source']}]")
