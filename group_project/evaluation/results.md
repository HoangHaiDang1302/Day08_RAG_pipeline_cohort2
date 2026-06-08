# RAG Evaluation Results

## Framework

Primary framework: lightweight local evaluator inspired by RAGAS/DeepEval.

Optional framework: DeepEval via `--framework deepeval` using Faithfulness, Answer Relevancy, Contextual Recall, and Contextual Precision.

## Dataset

- Golden Q&A pairs: 15
- Domain: Vietnamese drug law documents and news about Vietnamese artists related to drug cases.

## Overall Scores

| Metric | Config A (hybrid + rerank) | Config B (hybrid no rerank) | Delta A-B |
|--------|-----------------------------|------------------------------|-----------|
| Faithfulness | 0.758 | 0.911 | -0.153 |
| Answer Relevance | 0.76 | 0.794 | -0.034 |
| Context Recall | 0.597 | 0.623 | -0.026 |
| Context Precision | 0.96 | 0.96 | 0.0 |
| **Average** | **0.769** | **0.822** | **-0.053** |

## A/B Comparison Analysis

**Config A: hybrid + rerank**

Runs semantic search + lexical search, merges with RRF, reranks candidates, then generates citation-grounded answers.

**Config B: hybrid no rerank**

Runs semantic search + lexical search and RRF merge, but skips reranking to measure the impact of the reranker.

**Conclusion:**

Config A is the demo pipeline because it keeps the full retrieval flow and citation output. Some legal-law questions still need better full-text extraction from PDFs to improve recall.

## Worst Performers (Bottom 3 - Config A)

| # | Question | Avg | Faithfulness | Relevance | Recall | Precision | Top Sources | Root Cause |
|---|----------|-----|--------------|-----------|--------|-----------|-------------|------------|
| 1 | Corpus kết hợp những loại nguồn nào cho RAG? | 0.550 | 0.647 | 0.633 | 0.522 | 0.4 | nghe-si-bi-bat-vietnamnet-2025.md, nghi-dinh-116-2021.md, luat-phong-chong-ma-tuy-2021.md | Expected context is not fully present in retrieved chunks or legal PDFs are metadata fallback only. |
| 2 | Nghị định 105/2021/NĐ-CP trong corpus liên quan đến nội dung gì? | 0.652 | 0.665 | 0.648 | 0.296 | 1.0 | nghi-dinh-105-2021.md, nghi-dinh-116-2021.md, nhikolai-dinh-tuoitre-2024.md | Expected context is not fully present in retrieved chunks or legal PDFs are metadata fallback only. |
| 3 | Luật Phòng, chống ma túy 2021 là văn bản nào trong bộ dữ liệu? | 0.715 | 0.723 | 0.736 | 0.4 | 1.0 | chi-dan-vov-2026.md, nhikolai-dinh-tuoitre-2024.md, chi-dan-vov-2026.md | Expected context is not fully present in retrieved chunks or legal PDFs are metadata fallback only. |

## DeepEval Run

- Status: `completed`
- Case count: `1`
- Metrics: `FaithfulnessMetric, AnswerRelevancyMetric, ContextualRecallMetric, ContextualPrecisionMetric`
- Raw output: `group_project/evaluation/deepeval_results.json`

## Recommendations

### Improvement 1

**Action:** Install MarkItDown and reconvert legal PDFs to extract full legal text.

**Expected impact:** Improve context recall for legal article questions.

### Improvement 2

**Action:** Add Vietnamese tokenization with underthesea or pyvi for BM25 and reranking.

**Expected impact:** Improve precision for accented/unaccented Vietnamese legal queries.

### Improvement 3

**Action:** Use Groq Cloud for real generation in the demo with `GROQ_API_KEY` and `use_llm=True`.

**Expected impact:** More natural answers while preserving citation grounding.

## Per-case Results (Config A)

| # | Question | Avg | Retrieval | Top Sources |
|---|----------|-----|-----------|-------------|
| 1 | Luật Phòng, chống ma túy 2021 là văn bản nào trong bộ dữ liệu? | 0.715 | pageindex | chi-dan-vov-2026.md, nhikolai-dinh-tuoitre-2024.md, chi-dan-vov-2026.md |
| 2 | Nghị định 105/2021/NĐ-CP trong corpus liên quan đến nội dung gì? | 0.652 | pageindex | nghi-dinh-105-2021.md, nghi-dinh-116-2021.md, nhikolai-dinh-tuoitre-2024.md |
| 3 | Nghị định nào trong dữ liệu nói về danh mục chất ma túy và tiền chất? | 0.752 | hybrid | chi-dan-vov-2026.md, chi-dan-vov-2026.md, nhikolai-dinh-tuoitre-2024.md |
| 4 | Bộ dữ liệu có văn bản nào liên quan đến cai nghiện ma túy và quản lý sau cai nghiện? | 0.767 | pageindex | nhikolai-dinh-tuoitre-2024.md, chi-dan-vov-2026.md, chi-dan-vov-2026.md |
| 5 | Các văn bản pháp luật trong corpus thuộc nhóm tài liệu nào? | 0.747 | hybrid | chi-dan-an-tay-plo-2024.md, chi-dan-an-tay-plo-2024.md, nhikolai-dinh-tuoitre-2024.md |
| 6 | Bài báo nào trong dữ liệu nhắc đến diễn viên hài Hữu Tín? | 0.870 | hybrid | huu-tin-vnexpress-2022.md, chi-dan-an-tay-plo-2024.md, chi-dan-an-tay-plo-2024.md |
| 7 | Hữu Tín được mô tả là ai trong dữ liệu? | 0.828 | hybrid | chi-dan-an-tay-plo-2024.md, huu-tin-vnexpress-2022.md, chi-dan-an-tay-plo-2024.md |
| 8 | Bài Tuổi Trẻ trong dữ liệu nói về người mẫu nào bị bắt trong chuyên án ma túy? | 0.873 | hybrid | nhikolai-dinh-tuoitre-2024.md, nhikolai-dinh-tuoitre-2024.md, nhikolai-dinh-tuoitre-2024.md |
| 9 | Nhikolai Đinh từng tham gia chương trình nào theo dữ liệu? | 0.868 | pageindex | nhikolai-dinh-tuoitre-2024.md, nhikolai-dinh-tuoitre-2024.md, chi-dan-an-tay-plo-2024.md |
| 10 | Bài Pháp Luật TP.HCM tổng hợp những nghệ sĩ nào liên quan đến ma túy? | 0.849 | hybrid | chi-dan-an-tay-plo-2024.md, nghe-si-bi-bat-vietnamnet-2025.md, chi-dan-an-tay-plo-2024.md |
| 11 | VietnamNet điểm lại những nghệ sĩ nào từng bị bắt hoặc xử lý vì ma túy? | 0.754 | hybrid | nghe-si-bi-bat-vietnamnet-2025.md, chi-dan-an-tay-plo-2024.md, chi-dan-vov-2026.md |
| 12 | Bài VOV trong dữ liệu viết về ca sĩ nào? | 0.792 | pageindex | nhikolai-dinh-tuoitre-2024.md, chi-dan-vov-2026.md, chi-dan-vov-2026.md |
| 13 | Chi Dân được dữ liệu mô tả với tên thật là gì? | 0.781 | hybrid | chi-dan-an-tay-plo-2024.md, chi-dan-vov-2026.md, chi-dan-vov-2026.md |
| 14 | Corpus kết hợp những loại nguồn nào cho RAG? | 0.550 | pageindex | nghe-si-bi-bat-vietnamnet-2025.md, nghi-dinh-116-2021.md, luat-phong-chong-ma-tuy-2021.md |
| 15 | Khi trả lời câu hỏi, pipeline cần hiển thị thông tin gì để hỗ trợ citation? | 0.734 | pageindex | nhikolai-dinh-tuoitre-2024.md, nghe-si-bi-bat-vietnamnet-2025.md, nghe-si-bi-bat-vietnamnet-2025.md |
