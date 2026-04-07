"""RAGAS Answer Correctness — measures factual overlap between generated answer and ground truth."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "rag"))
from bedrock_llm import chat_completion

EXTRACT_STATEMENTS_PROMPT = """Extract all factual statements from the following text as a numbered list.
Each statement should be a single, atomic fact.

Text: {text}

Statements:"""

CLASSIFY_PROMPT = """Given the following statement from the generated answer, classify it as:
- TP (True Positive): statement is present in both generated answer AND ground truth
- FP (False Positive): statement is in generated answer but NOT in ground truth
- FN (False Negative): statement is in ground truth but NOT in generated answer

Ground truth statements:
{ground_truth_statements}

Statement to classify: {statement}

Return ONLY one of: TP, FP, FN"""


def evaluate_answer_correctness(
    question: str, answer: str, ground_truth: str, f1_weight: float = 0.5
) -> float:
    """RAGAS Answer Correctness: weighted F1 between generated answer and ground truth.

    1. Extract atomic statements from both answer and ground truth.
    2. Classify each answer statement as TP or FP against ground truth.
    3. Classify each ground truth statement as TP or FN against answer.
    4. Score = weighted F1 score.
    """
    # Extract statements from answer
    answer_stmts_text = chat_completion(
        system_prompt="You are a statement extraction assistant.",
        user_prompt=EXTRACT_STATEMENTS_PROMPT.format(text=answer),
        temperature=0.0,
        max_tokens=500,
    )
    answer_stmts = _parse_statements(answer_stmts_text)

    # Extract statements from ground truth
    gt_stmts_text = chat_completion(
        system_prompt="You are a statement extraction assistant.",
        user_prompt=EXTRACT_STATEMENTS_PROMPT.format(text=ground_truth),
        temperature=0.0,
        max_tokens=500,
    )
    gt_stmts = _parse_statements(gt_stmts_text)

    if not answer_stmts and not gt_stmts:
        return 1.0
    if not answer_stmts or not gt_stmts:
        return 0.0

    gt_joined = "\n".join(f"- {s}" for s in gt_stmts)

    # Classify answer statements
    tp = 0
    fp = 0
    for stmt in answer_stmts:
        verdict = chat_completion(
            system_prompt="You are a classification assistant. Return ONLY TP, FP, or FN.",
            user_prompt=CLASSIFY_PROMPT.format(
                ground_truth_statements=gt_joined, statement=stmt
            ),
            temperature=0.0,
            max_tokens=5,
        ).strip().upper()
        if verdict.startswith("TP"):
            tp += 1
        else:
            fp += 1

    # Count FN: ground truth statements not covered by answer
    answer_joined = "\n".join(f"- {s}" for s in answer_stmts)
    fn = 0
    for stmt in gt_stmts:
        verdict = chat_completion(
            system_prompt="You are a classification assistant. Return ONLY TP or FN.",
            user_prompt=f"""Is the following ground truth statement covered by any of the answer statements?

Answer statements:
{answer_joined}

Ground truth statement: {stmt}

Return ONLY: TP (if covered) or FN (if not covered)""",
            temperature=0.0,
            max_tokens=5,
        ).strip().upper()
        if verdict.startswith("FN"):
            fn += 1

    # F1 score
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    if precision + recall == 0:
        return 0.0

    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def _parse_statements(text: str) -> list[str]:
    return [
        line.strip().lstrip("0123456789.-) ").strip()
        for line in text.strip().split("\n")
        if line.strip() and any(c.isalpha() for c in line)
    ]
