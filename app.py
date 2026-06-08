"""Streamlit UI for the Day 8 RAG chatbot."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

PROJECT_DIR = Path(__file__).resolve().parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from src.task10_generation import generate_with_citation


def _source_label(chunk: dict, index: int) -> str:
    metadata = chunk.get("metadata", {}) or {}
    return str(metadata.get("source") or metadata.get("path") or f"Source {index}")


def _render_sources(sources: list[dict]) -> None:
    if not sources:
        st.caption("Không có source chunks.")
        return

    with st.expander(f"Sources used ({len(sources)})", expanded=False):
        for index, chunk in enumerate(sources, 1):
            source = _source_label(chunk, index)
            score = float(chunk.get("score", 0.0) or 0.0)
            retrieval_source = chunk.get("source", "hybrid")
            metadata = chunk.get("metadata", {}) or {}

            st.markdown(f"**{index}. {source}**")
            st.caption(
                f"retrieval={retrieval_source} | score={score:.3f} | "
                f"type={metadata.get('type', 'unknown')}"
            )
            st.write(chunk.get("content", "")[:1200])
            st.divider()


st.set_page_config(page_title="Drug Law RAG Chatbot", page_icon="🔎", layout="wide")

st.title("Drug Law RAG Chatbot")
st.caption("Hỏi đáp về pháp luật ma túy và tin tức nghệ sĩ liên quan. Mặc định dùng Groq để sinh câu trả lời có citation.")

with st.sidebar:
    st.header("Settings")
    top_k = st.slider("Top K context chunks", min_value=1, max_value=10, value=5, step=1)
    use_groq = st.toggle("Use Groq generation", value=True)
    st.caption("Nếu Groq lỗi hoặc thiếu key, app tự fallback sang câu trả lời local có citation.")

    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            _render_sources(message.get("sources", []))

prompt = st.chat_input("Nhập câu hỏi của bạn...")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving context and generating answer..."):
            result = generate_with_citation(prompt, top_k=top_k, use_llm=use_groq)

        answer = result["answer"]
        st.markdown(answer)
        _render_sources(result.get("sources", []))

        config = result.get("generation_config", {})
        if config.get("llm_error"):
            st.warning(f"Groq/API fallback reason: {config['llm_error']}")

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "sources": result.get("sources", []),
        }
    )

