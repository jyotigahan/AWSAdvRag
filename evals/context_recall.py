"""RAGAS Context Recall — measures if the context contains all info needed to produce the ground truth answer."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "rag"))
from bedrock_llm import chat_completion

STATEMENT_EXTRACTION_PROMPT = """Break the following ground truth answer into individual, atomic statements.
Return a numbered list.

Ground truth answer: {ground_truth}

Statements:"""

ATTRIBUTION_PROMPT = """Can the following statement be attributed to (i.e. inferred from) the given context?
Answer ONLY "yes" or "no".

Context: {context}
Statement: {statement}

Attributable:"""


def evaluate_context_recall(
    question: str, ground_truth: str, context: str
) -> float:
    """RAGAS Context Recall: fraction of ground truth statements attributable to the context.

    1. Decompose the ground truth answer into atomic statements.
    2. For each statement, check if it can be attributed to the retrieved context.
    3. Score = attributed_statements / total_statements.
    """
    # Step 1: Extract statements from ground truth
    statements_text = chat_completion(
        system_prompt="You are a statement extraction assistant.",
        user_prompt=STATEMENT_EXTRACTION_PROMPT.format(ground_truth=ground_truth),
        temperature=0.0,
        max_tokens=500,
    )

    statements = [
        line.strip().lstrip("0123456789.-) ").strip()
        for line in statements_text.strip().split("\n")
        if line.strip() and any(c.isalpha() for c in line)
    ]

    if not statements:
        return 1.0  # Nothing to verify

    # Step 2: Check attribution for each statement
    attributed = 0
    for stmt in statements:
        verdict = chat_completion(
            system_prompt="You are a fact attribution assistant. Answer ONLY yes or no.",
            user_prompt=ATTRIBUTION_PROMPT.format(context=context, statement=stmt),
            temperature=0.0,
            max_tokens=5,
        )
        if verdict.strip().lower().startswith("yes"):
            attributed += 1

    return attributed / len(statements)
