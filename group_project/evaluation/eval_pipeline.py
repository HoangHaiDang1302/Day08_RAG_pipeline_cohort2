"""
Group project - RAG evaluation pipeline.

Run local evaluator:
    python group_project/evaluation/eval_pipeline.py

Run DeepEval evaluator:
    python group_project/evaluation/eval_pipeline.py --framework deepeval

Run both:
    python group_project/evaluation/eval_pipeline.py --framework both

DeepEval uses an LLM judge, so provider/API configuration must be available in
the environment. The local evaluator remains as a deterministic fallback for
offline classroom demos.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    pass

PROJECT_DIR = Path(__file__).resolve().parents[2]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from src.task4_chunking_indexing import _tokenize
from src.task9_retrieval_pipeline import retrieve
from src.task10_generation import generate_with_citation

EVAL_DIR = Path(__file__).parent
GOLDEN_DATASET_PATH = EVAL_DIR / "golden_dataset.json"
RESULTS_PATH = EVAL_DIR / "results.md"
DEEPEVAL_RESULTS_PATH = EVAL_DIR / "deepeval_results.json"


class OpenAICompatibleDeepEvalJudge:
    """DeepEval custom judge model backed by OpenAI-compatible providers."""

    def __init__(self) -> None:
        from deepeval.models import DeepEvalBaseLLM
        from openai import OpenAI

        class _CompatibleModel(DeepEvalBaseLLM):
            def __init__(self) -> None:
                provider = os.getenv("DEEPEVAL_PROVIDER") or os.getenv("LLM_PROVIDER", "openrouter")
                self.provider = provider.lower()
                if self.provider == "openrouter":
                    self.api_key = os.getenv("OPENROUTER_API_KEY")
                    self.base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
                    self.model_name = os.getenv(
                        "OPENROUTER_MODEL",
                        "meta-llama/llama-3.3-70b-instruct:free",
                    )
                elif self.provider == "groq":
                    self.api_key = os.getenv("GROQ_API_KEY")
                    self.base_url = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
                    self.model_name = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
                else:
                    self.api_key = os.getenv("OPENAI_API_KEY")
                    self.base_url = os.getenv("OPENAI_BASE_URL")
                    self.model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
                if not self.api_key:
                    raise RuntimeError(f"API key is missing for provider {self.provider}")
                super().__init__(model=self.model_name)

            def load_model(self):
                default_headers = {}
                if self.provider == "openrouter":
                    default_headers = {
                        "HTTP-Referer": os.getenv("OPENROUTER_SITE_URL", "http://localhost"),
                        "X-Title": os.getenv("OPENROUTER_APP_NAME", "Day08 RAG Evaluation"),
                    }
                kwargs = {"api_key": self.api_key}
                if self.base_url:
                    kwargs["base_url"] = self.base_url
                if default_headers:
                    kwargs["default_headers"] = default_headers
                return OpenAI(**kwargs)

            def get_model_name(self) -> str:
                return f"{self.provider}/{self.model_name}"

            def supports_temperature(self) -> bool:
                return True

            def _complete(self, prompt: str) -> str:
                client = self.load_model()
                response = client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0,
                )
                return response.choices[0].message.content or ""

            def generate(self, prompt: str, schema=None):
                if schema is not None:
                    schema_prompt = (
                        f"{prompt}\n\n"
                        "Return ONLY valid JSON matching this Pydantic schema. "
                        f"Schema name: {schema.__name__}. "
                        f"JSON schema: {schema.model_json_schema()}"
                    )
                    output = self._complete(schema_prompt)
                    try:
                        parsed = json.loads(_extract_json_object(output))
                    except Exception:
                        parsed = {}
                    return schema.model_validate(parsed)
                return self._complete(prompt)

            async def a_generate(self, prompt: str, schema=None):
                return await asyncio.to_thread(self.generate, prompt, schema)

        self.model = _CompatibleModel()


def _extract_json_object(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:].strip()
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start : end + 1]
    return text


@dataclass
class EvalCaseResult:
    config: str
    question: str
    expected_answer: str
    expected_context: str
    answer: str
    faithfulness: float
    answer_relevance: float
    context_recall: float
    context_precision: float
    retrieval_source: str
    top_sources: list[str]

    @property
    def average(self) -> float:
        return statistics.mean(
            [
                self.faithfulness,
                self.answer_relevance,
                self.context_recall,
                self.context_precision,
            ]
        )


def load_golden_dataset() -> list[dict]:
    with open(GOLDEN_DATASET_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


def _token_set(text: str) -> set[str]:
    return {token for token in _tokenize(text) if len(token) > 1}


def _overlap_score(reference: str, candidate: str) -> float:
    reference_tokens = _token_set(reference)
    if not reference_tokens:
        return 0.0
    candidate_tokens = _token_set(candidate)
    return len(reference_tokens & candidate_tokens) / len(reference_tokens)


def _source_label(chunk: dict, fallback: str) -> str:
    metadata = chunk.get("metadata", {}) or {}
    return str(metadata.get("source") or metadata.get("path") or fallback)


def _make_extractive_answer(question: str, chunks: list[dict]) -> str:
    if not chunks:
        return "I cannot verify this information"

    lines = [f"Cau hoi: {question}"]
    for index, chunk in enumerate(chunks[:3], 1):
        source = _source_label(chunk, f"source_{index}")
        content = " ".join(chunk.get("content", "").split())
        lines.append(f"- {content[:260]} [{source}]")
    return "\n".join(lines)


def _score_case(
    question: str,
    expected_answer: str,
    expected_context: str,
    answer: str,
    chunks: list[dict],
) -> dict:
    context_text = "\n".join(chunk.get("content", "") for chunk in chunks)
    expected_signal = f"{expected_answer} {expected_context}"
    question_signal = f"{question} {expected_answer}"

    citation_bonus = 0.15 if "[" in answer and "]" in answer else 0.0
    faithfulness = min(1.0, 0.85 * _overlap_score(answer, context_text) + citation_bonus)
    answer_relevance = min(
        1.0,
        0.45 * _overlap_score(question, answer)
        + 0.55 * _overlap_score(expected_answer, answer),
    )
    context_recall = _overlap_score(expected_signal, context_text)

    if not chunks:
        context_precision = 0.0
    else:
        useful_chunks = sum(
            1
            for chunk in chunks
            if _overlap_score(question_signal, chunk.get("content", "")) >= 0.08
        )
        context_precision = useful_chunks / len(chunks)

    return {
        "faithfulness": round(faithfulness, 3),
        "answer_relevance": round(answer_relevance, 3),
        "context_recall": round(context_recall, 3),
        "context_precision": round(context_precision, 3),
    }


def run_config(config_name: str, item: dict) -> EvalCaseResult:
    question = item["question"]
    expected_answer = item["expected_answer"]
    expected_context = item["expected_context"]

    if config_name == "A_hybrid_rerank":
        result = generate_with_citation(question, top_k=5, use_llm=False)
        answer = result["answer"]
        chunks = result["sources"]
        retrieval_source = result.get("retrieval_source", "hybrid")
    elif config_name == "B_hybrid_no_rerank":
        chunks = retrieve(question, top_k=5, use_reranking=False)
        answer = _make_extractive_answer(question, chunks)
        retrieval_source = chunks[0].get("source", "none") if chunks else "none"
    else:
        raise ValueError(f"Unknown config: {config_name}")

    scores = _score_case(question, expected_answer, expected_context, answer, chunks)
    return EvalCaseResult(
        config=config_name,
        question=question,
        expected_answer=expected_answer,
        expected_context=expected_context,
        answer=answer,
        faithfulness=scores["faithfulness"],
        answer_relevance=scores["answer_relevance"],
        context_recall=scores["context_recall"],
        context_precision=scores["context_precision"],
        retrieval_source=retrieval_source,
        top_sources=[
            _source_label(chunk, f"source_{index}")
            for index, chunk in enumerate(chunks[:3], 1)
        ],
    )


def summarize(results: list[EvalCaseResult]) -> dict:
    return {
        "faithfulness": round(statistics.mean(item.faithfulness for item in results), 3),
        "answer_relevance": round(statistics.mean(item.answer_relevance for item in results), 3),
        "context_recall": round(statistics.mean(item.context_recall for item in results), 3),
        "context_precision": round(statistics.mean(item.context_precision for item in results), 3),
        "average": round(statistics.mean(item.average for item in results), 3),
    }


def compare_configs(golden_dataset: list[dict]) -> tuple[dict[str, list[EvalCaseResult]], dict[str, dict]]:
    by_config: dict[str, list[EvalCaseResult]] = {}
    summary: dict[str, dict] = {}

    for config in ["A_hybrid_rerank", "B_hybrid_no_rerank"]:
        case_results = [run_config(config, item) for item in golden_dataset]
        by_config[config] = case_results
        summary[config] = summarize(case_results)

    return by_config, summary


def evaluate_with_deepeval(golden_dataset: list[dict], limit: int | None = None) -> dict:
    """
    Run DeepEval with the 4 required RAG metrics from the README.

    Metrics:
        - FaithfulnessMetric
        - AnswerRelevancyMetric
        - ContextualRecallMetric
        - ContextualPrecisionMetric
    """
    try:
        from deepeval import evaluate
        from deepeval.evaluate.configs import AsyncConfig
        from deepeval.metrics import (
            AnswerRelevancyMetric,
            ContextualPrecisionMetric,
            ContextualRecallMetric,
            FaithfulnessMetric,
        )
        from deepeval.test_case import LLMTestCase
    except Exception as exc:
        return {
            "framework": "deepeval",
            "status": "import_failed",
            "error": f"{type(exc).__name__}: {exc}",
        }

    dataset = golden_dataset[:limit] if limit else golden_dataset
    test_cases = []
    case_payloads = []

    for item in dataset:
        result = generate_with_citation(item["question"], top_k=3, use_llm=False)
        contexts = [chunk.get("content", "")[:1200] for chunk in result.get("sources", [])[:3]]
        test_cases.append(
            LLMTestCase(
                input=item["question"],
                actual_output=result["answer"],
                expected_output=item["expected_answer"],
                retrieval_context=contexts,
            )
        )
        case_payloads.append(
            {
                "id": item.get("id"),
                "question": item["question"],
                "expected_context": item["expected_context"],
                "retrieval_source": result.get("retrieval_source"),
                "source_count": len(contexts),
            }
        )

    judge_model = None
    if os.getenv("OPENROUTER_API_KEY") or os.getenv("GROQ_API_KEY") or os.getenv("OPENAI_API_KEY"):
        try:
            judge_model = OpenAICompatibleDeepEvalJudge().model
        except Exception as exc:
            return {
                "framework": "deepeval",
                "status": "judge_setup_failed",
                "error": f"{type(exc).__name__}: {exc}",
                "case_count": len(test_cases),
            }

    metric_kwargs = {"model": judge_model} if judge_model is not None else {}
    metrics = [
        FaithfulnessMetric(threshold=0.7, **metric_kwargs),
        AnswerRelevancyMetric(threshold=0.7, **metric_kwargs),
        ContextualRecallMetric(threshold=0.7, **metric_kwargs),
        ContextualPrecisionMetric(threshold=0.7, **metric_kwargs),
    ]

    run_results = []
    errors = []
    for index, test_case in enumerate(test_cases):
        last_error = None
        for attempt in range(3):
            try:
                evaluation_result = evaluate(
                    test_cases=[test_case],
                    metrics=metrics,
                    async_config=AsyncConfig(run_async=False, throttle_value=1, max_concurrent=1),
                )
                run_results.append(
                    {
                        "case": case_payloads[index],
                        "status": "completed",
                        "result_repr": repr(evaluation_result),
                    }
                )
                last_error = None
                break
            except Exception as exc:
                last_error = exc
                if "rate" in str(exc).lower() or "429" in str(exc):
                    time.sleep(8 * (attempt + 1))
                else:
                    break
        if last_error is not None:
            errors.append(
                {
                    "case": case_payloads[index],
                    "error": f"{type(last_error).__name__}: {last_error}",
                }
            )
        time.sleep(3)

    payload = {
        "framework": "deepeval",
        "status": "completed" if not errors else "completed_with_errors",
        "case_count": len(test_cases),
        "judge_model": judge_model.get_model_name() if judge_model else "deepeval_default",
        "metrics": [metric.__class__.__name__ for metric in metrics],
        "completed_cases": len(run_results),
        "failed_cases": len(errors),
        "results": run_results,
        "errors": errors,
        "cases": case_payloads,
    }
    DEEPEVAL_RESULTS_PATH.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return payload


def _delta(summary: dict, metric: str) -> float:
    return round(summary["A_hybrid_rerank"][metric] - summary["B_hybrid_no_rerank"][metric], 3)


def export_results(
    by_config: dict[str, list[EvalCaseResult]],
    summary: dict[str, dict],
    deepeval_result: dict | None = None,
) -> None:
    config_a = summary["A_hybrid_rerank"]
    config_b = summary["B_hybrid_no_rerank"]
    worst = sorted(by_config["A_hybrid_rerank"], key=lambda item: item.average)[:3]

    lines = [
        "# RAG Evaluation Results",
        "",
        "## Framework",
        "",
        "Primary framework: lightweight local evaluator inspired by RAGAS/DeepEval.",
        "",
        "Optional framework: DeepEval via `--framework deepeval` using Faithfulness, Answer Relevancy, Contextual Recall, and Contextual Precision.",
        "",
        "## Dataset",
        "",
        f"- Golden Q&A pairs: {len(by_config['A_hybrid_rerank'])}",
        "- Domain: Vietnamese drug law documents and news about Vietnamese artists related to drug cases.",
        "",
        "## Overall Scores",
        "",
        "| Metric | Config A (hybrid + rerank) | Config B (hybrid no rerank) | Delta A-B |",
        "|--------|-----------------------------|------------------------------|-----------|",
        f"| Faithfulness | {config_a['faithfulness']} | {config_b['faithfulness']} | {_delta(summary, 'faithfulness')} |",
        f"| Answer Relevance | {config_a['answer_relevance']} | {config_b['answer_relevance']} | {_delta(summary, 'answer_relevance')} |",
        f"| Context Recall | {config_a['context_recall']} | {config_b['context_recall']} | {_delta(summary, 'context_recall')} |",
        f"| Context Precision | {config_a['context_precision']} | {config_b['context_precision']} | {_delta(summary, 'context_precision')} |",
        f"| **Average** | **{config_a['average']}** | **{config_b['average']}** | **{_delta(summary, 'average')}** |",
        "",
        "## A/B Comparison Analysis",
        "",
        "**Config A: hybrid + rerank**",
        "",
        "Runs semantic search + lexical search, merges with RRF, reranks candidates, then generates citation-grounded answers.",
        "",
        "**Config B: hybrid no rerank**",
        "",
        "Runs semantic search + lexical search and RRF merge, but skips reranking to measure the impact of the reranker.",
        "",
        "**Conclusion:**",
        "",
        "Config A is the demo pipeline because it keeps the full retrieval flow and citation output. Some legal-law questions still need better full-text extraction from PDFs to improve recall.",
        "",
        "## Worst Performers (Bottom 3 - Config A)",
        "",
        "| # | Question | Avg | Faithfulness | Relevance | Recall | Precision | Top Sources | Root Cause |",
        "|---|----------|-----|--------------|-----------|--------|-----------|-------------|------------|",
    ]

    for index, item in enumerate(worst, 1):
        source_text = ", ".join(item.top_sources)
        root_cause = "Expected context is not fully present in retrieved chunks or legal PDFs are metadata fallback only."
        question = item.question.replace("|", " ")
        lines.append(
            f"| {index} | {question} | {item.average:.3f} | {item.faithfulness} | "
            f"{item.answer_relevance} | {item.context_recall} | {item.context_precision} | "
            f"{source_text} | {root_cause} |"
        )

    lines.extend(
        [
            "",
            "## DeepEval Run",
            "",
        ]
    )
    if deepeval_result:
        lines.extend(
            [
                f"- Status: `{deepeval_result.get('status')}`",
                f"- Case count: `{deepeval_result.get('case_count', 0)}`",
                f"- Metrics: `{', '.join(deepeval_result.get('metrics', []))}`",
                f"- Raw output: `group_project/evaluation/{DEEPEVAL_RESULTS_PATH.name}`",
            ]
        )
        if deepeval_result.get("error"):
            lines.extend(["", f"**Error:** `{deepeval_result['error']}`"])
    else:
        lines.append("- Not run in this execution. Use `--framework deepeval` or `--framework both`.")

    lines.extend(
        [
            "",
            "## Recommendations",
            "",
            "### Improvement 1",
            "",
            "**Action:** Install MarkItDown and reconvert legal PDFs to extract full legal text.",
            "",
            "**Expected impact:** Improve context recall for legal article questions.",
            "",
            "### Improvement 2",
            "",
            "**Action:** Add Vietnamese tokenization with underthesea or pyvi for BM25 and reranking.",
            "",
            "**Expected impact:** Improve precision for accented/unaccented Vietnamese legal queries.",
            "",
            "### Improvement 3",
            "",
            "**Action:** Use Groq Cloud for real generation in the demo with `GROQ_API_KEY` and `use_llm=True`.",
            "",
            "**Expected impact:** More natural answers while preserving citation grounding.",
            "",
            "## Per-case Results (Config A)",
            "",
            "| # | Question | Avg | Retrieval | Top Sources |",
            "|---|----------|-----|-----------|-------------|",
        ]
    )

    for index, item in enumerate(by_config["A_hybrid_rerank"], 1):
        question = item.question.replace("|", " ")
        sources = ", ".join(item.top_sources)
        lines.append(f"| {index} | {question} | {item.average:.3f} | {item.retrieval_source} | {sources} |")

    RESULTS_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run group RAG evaluation.")
    parser.add_argument(
        "--framework",
        choices=["local", "deepeval", "both"],
        default="local",
        help="local is dependency-free; deepeval uses LLM judge metrics.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional number of golden cases to run with DeepEval while testing.",
    )
    args = parser.parse_args()

    golden_dataset = load_golden_dataset()
    if len(golden_dataset) < 15:
        raise ValueError(f"Golden dataset must contain at least 15 Q&A pairs, got {len(golden_dataset)}")

    by_config, summary = compare_configs(golden_dataset)
    deepeval_result = None
    if args.framework in {"deepeval", "both"}:
        deepeval_result = evaluate_with_deepeval(golden_dataset, limit=args.limit)

    export_results(by_config, summary, deepeval_result)
    print(f"Loaded {len(golden_dataset)} test cases")
    print(f"Config A average: {summary['A_hybrid_rerank']['average']}")
    print(f"Config B average: {summary['B_hybrid_no_rerank']['average']}")
    if deepeval_result:
        print(f"DeepEval status: {deepeval_result.get('status')}")
    print(f"Wrote: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
