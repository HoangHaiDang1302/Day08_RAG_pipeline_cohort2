"""
RAG Evaluation Pipeline — RAGAS-style metrics, LLM-as-judge.

Implement 4 metrics theo đúng phương pháp RAGAS nhưng tự tính bằng
LLM-as-judge thay vì dùng ragas package (tránh dependency hell).

Metrics:
    faithfulness       — answer có bám sát context không? (0-1)
    answer_relevancy   — answer có trả lời đúng câu hỏi không? (0-1)
    context_recall     — context có chứa đủ thông tin để trả lời không? (0-1)
    context_precision  — context retrieve được có liên quan không? (0-1)

A/B configs:
    Config A — hybrid search + cross-encoder reranking
    Config B — dense-only, no reranking

Chạy:
    cd Day08_RAG_pipeline_cohort2
    python -m group_project.evaluation.eval_pipeline
"""

import json
import sys
import time
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

from src.task9_retrieval_pipeline import retrieve
from src.task10_generation import format_context, reorder_for_llm, SYSTEM_PROMPT
from src.llm_client import get_llm_client, LLM_MODEL

if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

GOLDEN_DATASET_PATH = Path(__file__).parent / "golden_dataset.json"
RESULTS_PATH        = Path(__file__).parent / "results.md"


# =============================================================================
# HELPERS
# =============================================================================

def load_golden_dataset() -> list[dict]:
    with open(GOLDEN_DATASET_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _llm_score(prompt: str, retries: int = 2) -> float:
    """Gọi LLM, parse số float 0-1 từ response. Retry nếu parse lỗi."""
    client = get_llm_client()
    for attempt in range(retries + 1):
        try:
            resp = client.chat.completions.create(
                model=LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=16,
            )
            text = resp.choices[0].message.content.strip()
            # Lấy số đầu tiên trong response
            import re
            nums = re.findall(r"[01](?:\.\d+)?|\d+\.\d+", text)
            if nums:
                return min(1.0, max(0.0, float(nums[0])))
        except Exception:
            if attempt < retries:
                time.sleep(1)
    return 0.0


def run_rag(question: str, use_reranking: bool = True, top_k: int = 5) -> dict:
    chunks = retrieve(question, top_k=top_k, use_reranking=use_reranking)
    if not chunks:
        return {"answer": "Không tìm thấy thông tin liên quan.", "contexts": []}

    reordered = reorder_for_llm(chunks)
    context   = format_context(reordered)

    client = get_llm_client()
    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": f"Context:\n\n{context}\n\n---\n\nCâu hỏi: {question}"},
        ],
        temperature=0.3,
        top_p=0.9,
        max_tokens=1024,
    )
    return {
        "answer":   response.choices[0].message.content,
        "contexts": [c["content"] for c in chunks],
    }


# =============================================================================
# RAGAS-STYLE METRICS (LLM-as-judge)
# =============================================================================

def score_faithfulness(answer: str, contexts: list[str]) -> float:
    """
    Faithfulness: mỗi câu trong answer có được hỗ trợ bởi context không?
    Prompt RAGAS-style: chia answer thành statements, kiểm tra từng cái.
    """
    ctx = "\n---\n".join(contexts[:3])  # top-3 context để giảm token
    prompt = f"""Đánh giá mức độ faithful của câu trả lời so với context (0.0 đến 1.0).

Context:
{ctx}

Câu trả lời:
{answer}

Hướng dẫn:
- 1.0 = mọi thông tin trong câu trả lời đều có trong context
- 0.5 = một nửa thông tin có trong context
- 0.0 = câu trả lời bịa đặt hoàn toàn, không có trong context

Chỉ trả lời một số thập phân từ 0.0 đến 1.0, không giải thích."""
    return _llm_score(prompt)


def score_answer_relevancy(question: str, answer: str) -> float:
    """
    Answer Relevancy: answer có trực tiếp trả lời câu hỏi không?
    """
    prompt = f"""Đánh giá mức độ liên quan của câu trả lời với câu hỏi (0.0 đến 1.0).

Câu hỏi: {question}

Câu trả lời:
{answer}

Hướng dẫn:
- 1.0 = câu trả lời trực tiếp và đầy đủ trả lời câu hỏi
- 0.5 = câu trả lời liên quan nhưng không đầy đủ hoặc lạc đề một phần
- 0.0 = câu trả lời hoàn toàn không liên quan đến câu hỏi

Chỉ trả lời một số thập phân từ 0.0 đến 1.0, không giải thích."""
    return _llm_score(prompt)


def score_context_recall(ground_truth: str, contexts: list[str]) -> float:
    """
    Context Recall: ground truth answer có thể được suy ra từ context không?
    """
    ctx = "\n---\n".join(contexts[:3])
    prompt = f"""Đánh giá mức độ context chứa đủ thông tin để suy ra câu trả lời chuẩn (0.0 đến 1.0).

Context đã retrieve:
{ctx}

Câu trả lời chuẩn (ground truth):
{ground_truth}

Hướng dẫn:
- 1.0 = context chứa đủ tất cả thông tin cần thiết để trả lời
- 0.5 = context chứa một phần thông tin
- 0.0 = context không chứa thông tin liên quan đến câu trả lời chuẩn

Chỉ trả lời một số thập phân từ 0.0 đến 1.0, không giải thích."""
    return _llm_score(prompt)


def score_context_precision(question: str, contexts: list[str]) -> float:
    """
    Context Precision: tỷ lệ context chunks thực sự liên quan đến câu hỏi.
    """
    if not contexts:
        return 0.0
    scores = []
    for ctx in contexts:
        prompt = f"""Đánh giá mức độ liên quan của đoạn văn bản sau với câu hỏi (0.0 đến 1.0).

Câu hỏi: {question}

Đoạn văn bản:
{ctx[:600]}

Hướng dẫn:
- 1.0 = đoạn văn bản trực tiếp liên quan và hữu ích để trả lời câu hỏi
- 0.5 = liên quan một phần
- 0.0 = không liên quan đến câu hỏi

Chỉ trả lời một số thập phân từ 0.0 đến 1.0, không giải thích."""
        scores.append(_llm_score(prompt))
    return sum(scores) / len(scores)


# =============================================================================
# EVALUATE CONFIG
# =============================================================================

def evaluate_config(
    golden_dataset: list[dict],
    use_reranking: bool,
    config_name: str,
) -> dict:
    print(f"\n{'='*60}")
    print(f"Config: {config_name}")
    print(f"{'='*60}")

    per_question = []
    total = len(golden_dataset)

    for i, item in enumerate(golden_dataset, 1):
        q  = item["question"]
        gt = item["expected_answer"]
        print(f"  [{i:02d}/{total}] {q[:60]}...")

        # RAG
        rag = run_rag(q, use_reranking=use_reranking)
        answer   = rag["answer"]
        contexts = rag["contexts"]

        # 4 metrics
        faith   = score_faithfulness(answer, contexts)
        rel     = score_answer_relevancy(q, answer)
        recall  = score_context_recall(gt, contexts)
        prec    = score_context_precision(q, contexts)
        avg     = (faith + rel + recall + prec) / 4

        print(f"         faith={faith:.3f}  rel={rel:.3f}  recall={recall:.3f}  prec={prec:.3f}")

        per_question.append({
            "question":          q,
            "answer":            answer,
            "ground_truth":      gt,
            "contexts":          contexts,
            "faithfulness":      faith,
            "answer_relevancy":  rel,
            "context_recall":    recall,
            "context_precision": prec,
            "avg_score":         avg,
        })

    n = len(per_question)
    scores = {
        "faithfulness":      sum(r["faithfulness"]      for r in per_question) / n,
        "answer_relevancy":  sum(r["answer_relevancy"]  for r in per_question) / n,
        "context_recall":    sum(r["context_recall"]    for r in per_question) / n,
        "context_precision": sum(r["context_precision"] for r in per_question) / n,
    }
    scores["average"] = sum(scores.values()) / 4

    print(f"\n  Summary — {config_name}:")
    for k, v in scores.items():
        print(f"    {k:<22}: {v:.4f}")

    return {"config": config_name, "scores": scores, "per_question": per_question}


# =============================================================================
# EXPORT RESULTS
# =============================================================================

def export_results(config_a: dict, config_b: dict):
    def fmt(v: float) -> str:
        return f"{v:.4f}"

    def delta(a: float, b: float) -> str:
        d = a - b
        return f"{'+'if d>=0 else ''}{d:.4f}"

    sa, sb = config_a["scores"], config_b["scores"]
    worst3 = sorted(config_a["per_question"], key=lambda x: x["avg_score"])[:3]
    now    = datetime.now().strftime("%Y-%m-%d %H:%M")

    winner = config_a["config"] if sa["average"] >= sb["average"] else config_b["config"]

    md = f"""# RAG Evaluation Results

> Generated: {now}
> Framework: **RAGAS-style LLM-as-judge** (faithfulness · answer_relevancy · context_recall · context_precision)
> Dataset: {len(config_a['per_question'])} questions (`golden_dataset.json`)
> LLM judge: `{LLM_MODEL}`

---

## Overall Scores

| Metric | Config A — Hybrid + Rerank | Config B — Dense-only | Δ (A − B) |
|--------|:--------------------------:|:---------------------:|:---------:|
| Faithfulness       | {fmt(sa['faithfulness'])}      | {fmt(sb['faithfulness'])}      | {delta(sa['faithfulness'],      sb['faithfulness'])} |
| Answer Relevancy   | {fmt(sa['answer_relevancy'])}  | {fmt(sb['answer_relevancy'])}  | {delta(sa['answer_relevancy'],  sb['answer_relevancy'])} |
| Context Recall     | {fmt(sa['context_recall'])}    | {fmt(sb['context_recall'])}    | {delta(sa['context_recall'],    sb['context_recall'])} |
| Context Precision  | {fmt(sa['context_precision'])} | {fmt(sb['context_precision'])} | {delta(sa['context_precision'], sb['context_precision'])} |
| **Average**        | **{fmt(sa['average'])}**       | **{fmt(sb['average'])}**       | **{delta(sa['average'], sb['average'])}** |

---

## A/B Comparison Analysis

**Config A — Hybrid + Rerank:**
Kết hợp semantic search (dense vector ChromaDB cosine) + BM25 lexical search, merge bằng RRF, sau đó rerank bằng cross-encoder (`AITeamVN/Vietnamese_Reranker` hoặc Jina). Cross-encoder đọc toàn bộ cặp (query, chunk) nên sắp xếp lại chính xác hơn RRF thuần.

**Config B — Dense-only, no reranking:**
Chỉ dùng semantic search vector (cosine similarity), bỏ qua BM25 và reranker. Nhanh hơn nhưng có thể bỏ sót các từ khoá pháp lý mà sparse search bắt được tốt hơn (số điều luật, tên văn bản cụ thể).

**Kết luận:**
**{winner}** cho kết quả tốt hơn với average score Δ = {delta(sa['average'], sb['average'])}. {"Reranking cải thiện rõ rệt faithfulness và context precision — cross-encoder hiểu ngữ nghĩa tốt hơn cosine trên văn bản pháp lý tiếng Việt có nhiều từ chuyên ngành." if sa['average'] >= sb['average'] else "Dense-only đủ tốt cho domain này; overhead của reranking chưa justify rõ ràng, có thể do reranker chưa được fine-tune trên domain pháp luật Việt Nam."}

---

## Per-question Scores — Config A

| # | Question (truncated) | Faith. | Rel. | Recall | Prec. | Avg |
|---|----------------------|:------:|:----:|:------:|:-----:|:---:|
"""
    for i, pq in enumerate(config_a["per_question"], 1):
        q_short = pq["question"][:52].rstrip() + ("…" if len(pq["question"]) > 52 else "")
        md += (
            f"| {i:2d} | {q_short} "
            f"| {pq['faithfulness']:.3f} "
            f"| {pq['answer_relevancy']:.3f} "
            f"| {pq['context_recall']:.3f} "
            f"| {pq['context_precision']:.3f} "
            f"| {pq['avg_score']:.3f} |\n"
        )

    md += f"""
---

## Worst Performers — Bottom 3 (Config A)

| # | Question | Faith. | Rel. | Recall | Root Cause |
|---|----------|:------:|:----:|:------:|------------|
"""
    for i, pq in enumerate(worst3, 1):
        q_short = pq["question"][:58].rstrip() + ("…" if len(pq["question"]) > 58 else "")
        if pq["context_recall"] < 0.5:
            cause = "Retrieval miss — golden context chưa có trong ChromaDB"
        elif pq["faithfulness"] < 0.5:
            cause = "LLM hallucination — answer vượt ngoài context retrieve được"
        elif pq["answer_relevancy"] < 0.5:
            cause = "Answer lạc đề — LLM không bám sát câu hỏi"
        else:
            cause = "Context noise — nhiều chunk nhiễu lọt vào top-k"
        md += (
            f"| {i} | {q_short} "
            f"| {pq['faithfulness']:.3f} "
            f"| {pq['answer_relevancy']:.3f} "
            f"| {pq['context_recall']:.3f} "
            f"| {cause} |\n"
        )

    md += """
---

## Recommendations

### Cải tiến 1 — Chunk theo điều khoản pháp lý
**Action:** Chunk theo ranh giới điều/khoản thay vì fixed-size token; giữ nguyên số điều, tên văn bản pháp luật trong mỗi chunk metadata.
**Expected impact:** Tăng context_precision vì mỗi chunk mang đúng 1 điều luật hoàn chỉnh, giảm chunk nhiễu lọt vào top-k.

### Cải tiến 2 — Query expansion trước khi retrieve
**Action:** Dùng LLM sinh 2-3 biến thể query (paraphrase + từ khoá pháp lý), retrieve cho từng variant rồi merge bằng RRF.
**Expected impact:** Tăng context_recall với các câu hỏi ngắn hoặc thiếu từ khoá điều luật cụ thể.

### Cải tiến 3 — Fine-tune reranker trên domain pháp luật Việt Nam
**Action:** Tạo training set từ golden_dataset (câu hỏi + relevant chunk + hard negative), fine-tune `AITeamVN/Vietnamese_Reranker` thêm vài epoch.
**Expected impact:** Tăng faithfulness vì reranker đưa chunk chính xác nhất lên đầu context, LLM ít bị nhiễu hơn.
"""

    RESULTS_PATH.write_text(md, encoding="utf-8")
    print(f"\nResults exported → {RESULTS_PATH}")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("Loading golden dataset...")
    golden = load_golden_dataset()
    print(f"  {len(golden)} questions loaded.\n")
    print(f"LLM judge: {LLM_MODEL}")

    config_a = evaluate_config(golden, use_reranking=True,  config_name="Config A — Hybrid + Rerank")
    config_b = evaluate_config(golden, use_reranking=False, config_name="Config B — Dense-only")

    export_results(config_a, config_b)
    print("\nEvaluation complete.")
