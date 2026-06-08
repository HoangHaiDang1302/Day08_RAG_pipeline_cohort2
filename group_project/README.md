# Bài Tập Nhóm — Drug Law RAG Chatbot & Evaluation Pipeline

> Nguyễn Trường Phúc - 2A202600767
> 
> Trần Minh Anh - 2A202600706
> 
> Hoàng Hải Đăng - 2A202600916
> 
> Nguyễn Huyền San - 2A202600835
> 
> Vũ Đăng Khiêm -2A20260072
> 
> Lê Dương Hiếu - 2A202600635

---

## Tổng Quan Sản Phẩm

Nhóm lựa chọn **cả 2 yêu cầu**: xây dựng RAG Chatbot hoàn chỉnh **và** Evaluation Pipeline đánh giá chất lượng hệ thống.

**Domain:** Pháp luật Việt Nam về ma tuý & tin tức nghệ sĩ liên quan.

---

## Kiến Trúc Hệ Thống

```
Người dùng
    │
    ▼
┌─────────────────────────────────────────────────────┐
│              FastAPI Web App (app.py)                │
│  ┌──────────────┐    ┌──────────────────────────┐   │
│  │  Guardrail   │    │   Conversation Memory    │   │
│  │ (LLM judge)  │    │  (session → history[])   │   │
│  └──────┬───────┘    └──────────────────────────┘   │
│         │ ON_TOPIC                                   │
│         ▼                                            │
│  ┌──────────────────────────────────────────────┐   │
│  │           Retrieval Pipeline (Task 9)        │   │
│  │                                              │   │
│  │  Query ──┬── Semantic Search (Task 5)  ──┐   │   │
│  │          └── Lexical BM25  (Task 6)   ──┴──▶ │   │
│  │                  RRF Merge                   │   │
│  │                     │                        │   │
│  │          Cross-Encoder Reranker (Task 7)     │   │
│  │          AITeamVN/Vietnamese_Reranker        │   │
│  │                     │                        │   │
│  │          score < threshold → PageIndex (T8)  │   │
│  └──────────────────────────────────────────────┘   │
│         │                                            │
│         ▼                                            │
│  ┌──────────────────────────────────────────────┐   │
│  │         Generation (Task 10)                 │   │
│  │  Lost-in-middle reorder → Format context     │   │
│  │  → GPT-4o-mini (OpenRouter) → Answer + cite  │   │
│  └──────────────────────────────────────────────┘   │
│         │                                            │
│         ▼                                            │
│   Answer + Source Pills UI (static/index.html)       │
└─────────────────────────────────────────────────────┘

Vector Store: ChromaDB (cosine, persistent)
Embedding:    AITeamVN/Vietnamese_Embedding (1024-dim, Kaggle GPU)
              ↳ fallback: OpenAI text-embedding-3-small (1536-dim)
Reranker:     AITeamVN/Vietnamese_Reranker (Kaggle GPU)
              ↳ fallback: Jina Reranker API → RRF score
LLM:          GPT-4o-mini via OpenRouter (hoặc OpenAI trực tiếp)
```

---

## Tech Stack

| Layer          | Công nghệ                                                          |
| -------------- | ------------------------------------------------------------------ |
| Web framework  | FastAPI + Uvicorn                                                  |
| Frontend       | Vanilla JS + CSS (AI20K dark theme)                                |
| Vector DB      | ChromaDB (persistent, cosine similarity)                           |
| Embedding      | `AITeamVN/Vietnamese_Embedding` (Kaggle T4 GPU) / OpenAI fallback  |
| Lexical search | BM25 (rank-bm25)                                                   |
| Reranker       | `AITeamVN/Vietnamese_Reranker` (Kaggle T4 GPU) / Jina API fallback |
| LLM            | GPT-4o-mini qua OpenRouter API                                     |
| Evaluation     | RAGAS-style LLM-as-judge (self-contained, không cần ragas package) |

---

## Cấu Trúc Thư Mục

```
Day08_RAG_pipeline_cohort2/
├── app.py                          # FastAPI backend + guardrail + chat endpoint
├── src/
│   ├── task1_collect_legal_docs.py # Thu thập văn bản pháp luật
│   ├── task2_crawl_news.py         # Crawl tin tức nghệ sĩ
│   ├── task3_convert_markdown.py   # Chuẩn hoá sang Markdown
│   ├── task4_chunking_indexing.py  # Chunk + index vào ChromaDB
│   ├── task5_semantic_search.py    # Dense vector search
│   ├── task6_lexical_search.py     # BM25 sparse search
│   ├── task7_reranking.py          # Cross-encoder reranking
│   ├── task8_pageindex_vectorless.py # PageIndex fallback
│   ├── task9_retrieval_pipeline.py # Hybrid pipeline + fallback logic
│   ├── task10_generation.py        # LLM generation + citation
│   └── llm_client.py              # OpenRouter / OpenAI client factory
├── static/
│   ├── index.html                  # Chat UI
│   ├── style.css                   # AI20K dark/light theme
│   └── app.js                      # Frontend logic + modal sources
├── group_project/
│   └── evaluation/
│       ├── golden_dataset.json     # 15 Q&A pairs
│       ├── eval_pipeline.py        # RAGAS-style evaluation script
│       └── results.md              # Kết quả A/B evaluation
└── vietnamese_model_server.ipynb   # Kaggle notebook serving 2 Vietnamese models
```

---

## Hướng Dẫn Chạy

### 1. Cài đặt

```bash
pip install -r requirements.txt
```

### 2. Cấu hình `.env`

```env
# LLM (chọn 1 trong 2)
OPENROUTER_API_KEY=sk-or-...   # ưu tiên nếu có
OPENAI_API_KEY=sk-...          # fallback

# Kaggle GPU servers (optional — nếu đang serve Vietnamese models)
KAGGLE_EMBED_URL=https://xxxx.ngrok-free.app
KAGGLE_RERANK_URL=https://yyyy.ngrok-free.app
```

### 3. Index dữ liệu (lần đầu)

```bash
python -m src.task4_chunking_indexing
```

### 4. Chạy webapp

```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
# Mở http://localhost:8000
```

### 5. Chạy Evaluation Pipeline

```bash
python -m group_project.evaluation.eval_pipeline
# Kết quả ghi vào group_project/evaluation/results.md
```

---

## Evaluation Pipeline

### Framework

**RAGAS-style LLM-as-judge** — implement 4 metrics bằng prompt engineering trực tiếp, không phụ thuộc ragas package (tránh dependency conflict với langchain-community).

### Dataset

`golden_dataset.json` — 15 cặp Q&A bao gồm:

- Điều luật hình sự (tàng trữ, vận chuyển, mua bán, trồng cây ma tuý)
- Luật Phòng chống ma tuý 2021 (cai nghiện, tiền chất, trường học)
- Nghị định 105/2021/NĐ-CP
- Tin tức nghệ sĩ liên quan ma tuý

### Kết Quả A/B

| Metric            | Config A — Hybrid + Rerank | Config B — Dense-only |      Δ      |
| ----------------- | :------------------------: | :-------------------: | :---------: |
| Faithfulness      |           0.6000           |        0.5667         |   +0.0333   |
| Answer Relevancy  |           0.6000           |        0.5667         |   +0.0333   |
| Context Recall    |           0.5333           |        0.4333         |   +0.1000   |
| Context Precision |           0.3933           |        0.3600         |   +0.0333   |
| **Average**       |         **0.5317**         |      **0.4817**       | **+0.0500** |

**Kết luận:** Config A (Hybrid + Rerank) vượt trội hơn ở tất cả 4 metrics, đặc biệt Context Recall (+0.10) — hybrid search bắt được các từ khoá pháp lý cụ thể (số điều, tên văn bản) mà dense-only bỏ sót.

### Top 3 Cải Tiến Đề Xuất

1. **Chunk theo điều/khoản** — thay vì fixed-size, giữ nguyên ranh giới điều luật → tăng context_precision
2. **Query expansion** — LLM sinh 2-3 biến thể query trước khi retrieve → tăng context_recall
3. **Fine-tune reranker** — tạo training set từ golden_dataset, fine-tune `AITeamVN/Vietnamese_Reranker` → tăng faithfulness

---

## Tính Năng Nổi Bật

- **Guardrail thông minh** — LLM classifier phân biệt on-topic/off-topic, bao gồm kiến thức nền về ma tuý (không chỉ pháp luật)
- **Hybrid search** — kết hợp ChromaDB cosine + BM25, merge RRF → tốt hơn pure dense trên văn bản pháp lý
- **Lost-in-middle reorder** — sắp xếp lại chunks theo Liu et al. 2023 trước khi đưa vào LLM
- **Dual GPU serving** — `AITeamVN/Vietnamese_Embedding` trên cuda:0, `AITeamVN/Vietnamese_Reranker` trên cuda:1 (Kaggle T4)
- **Source pills** — mỗi response hiển thị pill tài liệu tham khảo, click mở modal full content với markdown render
- **Dark/Light mode** — toggle persistent qua localStorage
- **Conversation memory** — giữ 6 turns gần nhất, follow-up questions hoạt động đúng

---

## Phân Công Công Việc

| Thành viên         | MSSV        | Nhiệm vụ                                                          | Trạng thái    |
| ------------------ | ----------- | ----------------------------------------------------------------- | ------------- |
| Nguyễn Trường Phúc | 2A202600767 | Implement full pipeline (Task 1–10)e | Hoàn thành |
| Trần Minh Anh | 2A202600706 | Serving custom embedding/reranker models on server | Hoàn thành |
| Hoàng Hải Đăng | 2A202600916 | Implement RAG Chatbot UI | Hoàn thành |
| Nguyễn Huyền San | 2A202600835 | Add fallback OpenAI/Jina when custom model error | Hoàn thành |
| Vũ Đăng Khiêm | 2A202600727 | Run Eval pipeline with RAGAS | Hoàn thành |
| Lê Dương Hiếu | 2A202600635 | Add gruadrail, out-of-scope filtering | Hoàn thành |

---

> **Lưu ý:** Giữ lại repo này nếu học Track 3 giai đoạn 2 — sẽ phát triển tiếp lên Knowledge Graph để xử lý các câu hỏi phức tạp đa hop.
