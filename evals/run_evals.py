"""Run all 8 core RAGAS evaluations and generate HTML report with structured MetricsRAGAS logging."""
import os
import sys
import json
import logging
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "rag"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "function_app"))
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

# Install dashboard metrics handler
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "dashboard"))
try:
    from metrics_handler import install as install_dashboard
    install_dashboard()
except ImportError:
    pass  # Dashboard not available, metrics still go to stdout

from faithfulness import evaluate_faithfulness
from groundedness import evaluate_groundedness
from context_precision import evaluate_context_precision
from context_recall import evaluate_context_recall
from answer_relevance import evaluate_answer_relevance
from answer_correctness import evaluate_answer_correctness
from answer_similarity import evaluate_answer_similarity
from aspect_critique import evaluate_aspect_critique
from chain import rag_query

# Structured logger for MetricsRAGAS
logger = logging.getLogger("MetricsRAGAS")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(name)s | %(message)s"))
    logger.addHandler(handler)

# Test questions with ground truth answers (needed for correctness & similarity)
TEST_CASES = None  # Loaded from banking_test_cases.json


def _load_test_cases() -> list[dict]:
    """Load banking domain test cases from JSON file."""
    json_path = os.path.join(os.path.dirname(__file__), "..", "test_data", "banking_test_cases.json")
    if os.path.exists(json_path):
        with open(json_path) as f:
            return json.load(f)
    # Fallback if file not found
    return [
        {"question": "What is this document about?", "ground_truth": "This document covers banking policies."},
    ]

METRIC_NAMES = [
    "Faithfulness", "Groundedness", "Context Precision", "Context Recall",
    "Answer Relevance", "Answer Correctness", "Answer Similarity", "Aspect Critique",
]


def run_evaluation(test_cases: list[dict] | None = None) -> list[dict]:
    test_cases = test_cases or _load_test_cases()
    results = []

    for tc in test_cases:
        q = tc["question"]
        ground_truth = tc.get("ground_truth", "")
        print(f"\nEvaluating: {q}")
        response = rag_query(q)

        context = response["context_used"]
        answer = response["answer"]

        # --- All 8 Core RAGAS Metrics ---
        faithfulness = evaluate_faithfulness(q, answer, context)
        groundedness = evaluate_groundedness(q, answer, context)
        context_precision = evaluate_context_precision(q, answer, [context])
        context_recall = evaluate_context_recall(q, answer, context)
        answer_relevance = evaluate_answer_relevance(q, answer)
        answer_correctness = evaluate_answer_correctness(q, answer, ground_truth) if ground_truth else 0.0
        answer_similarity = evaluate_answer_similarity(q, answer, ground_truth) if ground_truth else 0.0
        aspect_result = evaluate_aspect_critique(q, answer, context)

        scores = {
            "Faithfulness": round(faithfulness, 4),
            "Groundedness": round(groundedness, 4),
            "Context Precision": round(context_precision, 4),
            "Context Recall": round(context_recall, 4),
            "Answer Relevance": round(answer_relevance, 4),
            "Answer Correctness": round(answer_correctness, 4),
            "Answer Similarity": round(answer_similarity, 4),
            "Aspect Critique": round(aspect_result["overall"], 4),
        }

        # --- Structured MetricsRAGAS log ---
        log_entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "question": q,
            "ground_truth_provided": bool(ground_truth),
            "rewritten_query": response["rewritten_query"],
            "sources": response["sources"],
            "retrieved_chunks": response["retrieved_chunks"],
            "scores": scores,
            "aspect_details": {k: v for k, v in aspect_result.items() if k != "overall"},
        }
        logger.info(json.dumps(log_entry))

        # Console output
        for name in METRIC_NAMES:
            print(f"  {name:25s} {scores[name]}")

        result = {
            "question": q,
            "rewritten_query": response["rewritten_query"],
            "answer": answer,
            "sources": response["sources"],
            "scores": scores,
            "aspect_details": {k: v for k, v in aspect_result.items() if k != "overall"},
        }
        results.append(result)

    # Summary
    avgs = {}
    for name in METRIC_NAMES:
        avgs[name] = round(sum(r["scores"][name] for r in results) / len(results), 4)

    summary = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "type": "summary",
        "total_questions": len(results),
        "avg_scores": avgs,
    }
    logger.info(json.dumps(summary))

    overall = sum(avgs.values()) / len(avgs)
    print(f"\n{'='*55}")
    print(f"  MetricsRAGAS Summary ({len(results)} questions)")
    print(f"{'='*55}")
    for name in METRIC_NAMES:
        print(f"  {name:25s} {avgs[name]:.4f}")
    print(f"  {'Overall':25s} {overall:.4f}")

    generate_html_report(results, avgs)
    return results


def generate_html_report(results: list[dict], avgs: dict) -> None:
    rows = ""
    for r in results:
        s = r["scores"]
        avg = sum(s.values()) / len(s)
        aspect = r.get("aspect_details", {})
        aspect_str = ", ".join(f"{k}:{'✓' if v == 1.0 else '✗'}" for k, v in aspect.items())
        rows += f"""
        <tr>
            <td>{r['question']}</td>
            <td class="answer">{r['answer'][:150]}...</td>
            <td>{', '.join(r['sources']) or 'N/A'}</td>
            <td class="score">{s['Faithfulness']:.3f}</td>
            <td class="score">{s['Groundedness']:.3f}</td>
            <td class="score">{s['Context Precision']:.3f}</td>
            <td class="score">{s['Context Recall']:.3f}</td>
            <td class="score">{s['Answer Relevance']:.3f}</td>
            <td class="score">{s['Answer Correctness']:.3f}</td>
            <td class="score">{s['Answer Similarity']:.3f}</td>
            <td class="score" title="{aspect_str}">{s['Aspect Critique']:.3f}</td>
            <td class="score">{avg:.3f}</td>
        </tr>"""

    overall = sum(avgs.values()) / len(avgs)

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>RAGAS Evaluation Report</title>
<style>
body {{ font-family: -apple-system, sans-serif; margin: 2rem; background: #f5f5f5; }}
h1 {{ color: #232f3e; }}
.summary {{ display: flex; gap: 0.8rem; margin: 1rem 0; flex-wrap: wrap; }}
.card {{ background: white; padding: 1.2rem; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); min-width: 120px; text-align: center; }}
.card h3 {{ margin: 0; color: #666; font-size: 0.75rem; }}
.card .value {{ font-size: 1.6rem; font-weight: bold; color: #232f3e; }}
table {{ width: 100%; border-collapse: collapse; background: white; border-radius: 8px; overflow: hidden; box-shadow: 0 2px 4px rgba(0,0,0,0.1); font-size: 0.85rem; }}
th {{ background: #232f3e; color: white; padding: 10px; text-align: left; white-space: nowrap; }}
td {{ padding: 8px 10px; border-bottom: 1px solid #eee; }}
.score {{ text-align: center; font-weight: bold; }}
.answer {{ max-width: 250px; font-size: 0.8rem; }}
tr:hover {{ background: #fff8e1; }}
</style></head><body>
<h1>RAGAS Evaluation Report — All 8 Core Metrics (AWS)</h1>
<p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
<div class="summary">
  <div class="card"><h3>Faithfulness</h3><div class="value">{avgs['Faithfulness']:.3f}</div></div>
  <div class="card"><h3>Groundedness</h3><div class="value">{avgs['Groundedness']:.3f}</div></div>
  <div class="card"><h3>Ctx Precision</h3><div class="value">{avgs['Context Precision']:.3f}</div></div>
  <div class="card"><h3>Ctx Recall</h3><div class="value">{avgs['Context Recall']:.3f}</div></div>
  <div class="card"><h3>Ans Relevance</h3><div class="value">{avgs['Answer Relevance']:.3f}</div></div>
  <div class="card"><h3>Ans Correctness</h3><div class="value">{avgs['Answer Correctness']:.3f}</div></div>
  <div class="card"><h3>Ans Similarity</h3><div class="value">{avgs['Answer Similarity']:.3f}</div></div>
  <div class="card"><h3>Aspect Critique</h3><div class="value">{avgs['Aspect Critique']:.3f}</div></div>
  <div class="card" style="background:#232f3e;"><h3 style="color:#ff9900;">Overall</h3><div class="value" style="color:white;">{overall:.3f}</div></div>
</div>
<table>
<tr><th>Question</th><th>Answer</th><th>Sources</th><th>Faith.</th><th>Grnd.</th><th>Ctx P.</th><th>Ctx R.</th><th>Ans R.</th><th>Ans C.</th><th>Ans S.</th><th>Aspect</th><th>Avg</th></tr>
{rows}
</table></body></html>"""

    report_path = os.path.join(os.path.dirname(__file__), "report", "eval_report.html")
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w") as f:
        f.write(html)
    print(f"\nReport saved to: {report_path}")


if __name__ == "__main__":
    run_evaluation()
