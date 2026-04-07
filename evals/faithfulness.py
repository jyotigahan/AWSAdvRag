"""RAGAS Faithfulness — measures if each claim in the answer can be inferred from the context."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "rag"))
from bedrock_llm import chat_completion

CLAIM_EXTRACTION_PROMPT = """Extract all factual claims from the following answer as a numbered list.
Each claim should be a single, atomic statement.

Answer: {answer}

Claims:"""

CLAIM_VERIFICATION_PROMPT = """Given the context below, determine if the following claim can be inferred from it.
Answer ONLY "yes" or "no".

Context: {context}
Claim: {claim}

Verdict:"""


def evaluate_faithfulness(question: str, answer: str, context: str) -> float:
    """RAGAS Faithfulness: fraction of answer claims supported by the context.

    1. Extract atomic claims from the answer.
    2. Verify each claim against the context.
    3. Score = supported_claims / total_claims.
    """
    # Step 1: Extract claims
    claims_text = chat_completion(
        system_prompt="You are a claim extraction assistant.",
        user_prompt=CLAIM_EXTRACTION_PROMPT.format(answer=answer),
        temperature=0.0,
        max_tokens=500,
    )

    claims = [
        line.strip().lstrip("0123456789.-) ").strip()
        for line in claims_text.strip().split("\n")
        if line.strip() and any(c.isalpha() for c in line)
    ]

    if not claims:
        return 1.0  # No claims to verify

    # Step 2: Verify each claim
    supported = 0
    for claim in claims:
        verdict = chat_completion(
            system_prompt="You are a fact verification assistant. Answer ONLY yes or no.",
            user_prompt=CLAIM_VERIFICATION_PROMPT.format(context=context, claim=claim),
            temperature=0.0,
            max_tokens=5,
        )
        if verdict.strip().lower().startswith("yes"):
            supported += 1

    # Step 3: Score
    return supported / len(claims)
