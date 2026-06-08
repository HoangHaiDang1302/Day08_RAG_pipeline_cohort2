# Bài Tập Nhóm - RAG Evaluation Pipeline

## Mục Tiêu

Nhóm chọn hướng **RAG Evaluation Pipeline** để đánh giá hệ thống RAG trả lời câu hỏi về:

- Pháp luật Việt Nam liên quan đến ma túy và chất cấm.
- Tin tức về nghệ sĩ Việt Nam liên quan đến ma túy.

Pipeline cá nhân từ Task 1-10 được tích hợp làm hệ thống RAG nền:

```text
data/landing
  -> data/standardized
  -> Task 4 chunking + local index
  -> Task 5 semantic search
  -> Task 6 lexical BM25
  -> Task 7 reranking
  -> Task 8 PageIndex fallback
  -> Task 9 retrieval pipeline
  -> Task 10 generation with citation
  -> group_project/evaluation
```

## Kiến Trúc Hệ Thống

```text
User Question
  |
  v
Task 9 Retrieval Pipeline
  |-- Semantic Search
  |-- Lexical Search / BM25
  |-- RRF Merge
  |-- Reranking
  |-- PageIndex fallback
  v
Task 10 Generation with Citation
  |
  v
Evaluation Pipeline
  |-- Golden Dataset
  |-- Local Metrics
  |-- DeepEval Metrics
  |-- A/B Comparison
  v
results.md
```

## Deliverables

| File | Mô tả | Trạng thái |
|------|------|------------|
| `group_project/evaluation/golden_dataset.json` | 15 cặp Q&A gồm question, expected_answer, expected_context | Done |
| `group_project/evaluation/eval_pipeline.py` | Script chạy evaluation local và DeepEval | Done |
| `group_project/evaluation/results.md` | Báo cáo điểm, A/B comparison, worst performers, recommendations | Done |
| `group_project/evaluation/deepeval_results.json` | Raw output khi chạy DeepEval | Sinh tự động |

## Golden Dataset

Dataset có 15 câu hỏi, chia thành 3 nhóm:

- Legal: câu hỏi về Luật Phòng, chống ma túy 2021 và các nghị định liên quan.
- News: câu hỏi về Hữu Tín, Nhikolai Đinh, Chi Dân, An Tây, Nguyễn Công Trí và các bài báo đã crawl.
- Hybrid: câu hỏi kiểm tra khả năng kết hợp legal/news và citation metadata.

## Metrics

Script hỗ trợ 4 metrics theo yêu cầu:

| Metric | Ý nghĩa |
|--------|---------|
| Faithfulness | Câu trả lời có bám vào retrieved context và có citation không |
| Answer Relevance | Câu trả lời có liên quan tới question và expected answer không |
| Context Recall | Retrieved context có bao phủ expected answer/context không |
| Context Precision | Tỷ lệ chunks hữu ích trong context truy xuất |

## A/B Comparison

Script so sánh 2 cấu hình:

| Config | Mô tả |
|--------|------|
| Config A | Hybrid retrieval + RRF merge + reranking + generation |
| Config B | Hybrid retrieval + RRF merge, bỏ reranking |

Mục tiêu là đo tác động của reranking trong pipeline.

## Hướng Dẫn Chạy

Cài dependencies:

```powershell
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Chạy local evaluation:

```powershell
.venv\Scripts\python.exe group_project\evaluation\eval_pipeline.py
```

Chạy DeepEval:

```powershell
.venv\Scripts\python.exe group_project\evaluation\eval_pipeline.py --framework deepeval
```

Chạy thử DeepEval trên 1 case:

```powershell
.venv\Scripts\python.exe group_project\evaluation\eval_pipeline.py --framework deepeval --limit 1
```

Chạy cả local và DeepEval:

```powershell
.venv\Scripts\python.exe group_project\evaluation\eval_pipeline.py --framework both
```

## Lưu Ý DeepEval

DeepEval thường dùng LLM judge, nên cần cấu hình provider/API key phù hợp trong `.env`. Nhóm dùng OpenRouter làm judge mặc định cho DeepEval vì hỗ trợ OpenAI-compatible API và có nhiều model `:free`.

Khuyến nghị:

```env
DEEPEVAL_PROVIDER=openrouter
OPENROUTER_API_KEY=sk-or-v1-xxx
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
OPENROUTER_MODEL=meta-llama/llama-3.3-70b-instruct:free
```

`meta-llama/llama-3.3-70b-instruct:free` được chọn vì là model lớn, multilingual tốt, phù hợp làm LLM judge cho câu hỏi tiếng Việt. Không dùng `openrouter/free` cho evaluation chính vì router random có thể làm kết quả không ổn định giữa các lần chạy.

Nếu LLM judge thiếu quota hoặc lỗi provider, script vẫn ghi lỗi vào `deepeval_results.json` và giữ báo cáo local trong `results.md`.

## Phân Công Công Việc

| Thành viên | MSSV | Nhiệm vụ | Trạng thái |
|-----------|------|----------|------------|
| Hoàng Hải Đăng | N/A | Tích hợp Task 1-10, tạo local RAG pipeline | Done |
| Hoàng Hải Đăng | N/A | Xây dựng golden dataset 15 Q&A | Done |
| Hoàng Hải Đăng | N/A | Viết evaluation pipeline local + DeepEval | Done |
| Hoàng Hải Đăng | N/A | Viết báo cáo kết quả và hướng chạy | Done |

## Kết Quả Hiện Tại

Local evaluation đã chạy được và sinh `results.md`. Kết quả gần nhất:

```text
Loaded 15 test cases
Config A average: 0.769
Config B average: 0.822
```

Điểm còn hạn chế chính: một số PDF pháp luật hiện được convert bằng metadata fallback, nên câu hỏi pháp luật chi tiết sẽ cải thiện rõ nếu cài MarkItDown và extract full text từ PDF.
