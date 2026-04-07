"""RAGAS Context Precision — measures if relevant chunks are ranked higher than irrelevant ones."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "rag"))
from bedrock_llm import chat_completion

PRECISION_PROMPT = """Given the question and the expected answer, determine if the following retrieved chunk
is useful for answering the question.
Answer ONLY "yes" or "no".

Question: {question}
Expected answer: {answer}
Retrieved chunk: {chunk}

Is this chunk useful?"""


def evaluate_context_precision(
    question: str, answer: str, retrieved_chunks: list[str]
) -> float:
    """RAGAS Context Precision: measures ranking quality of retrieved context.

    For each position k, checks if the chunk is relevant, then computes
    precision@k weighted by relevance. This rewards relevant chunks appearing
    earlier in the ranked list.

    Score = (1/total_relevant) * sum(precision@k * relevance@k) for k=1..n
    """
    if not retrieved_chunks:
        return 0.0

    relevance = []
    for chunk in retrieved_chunks:
        verdict = chat_completion(
            system_prompt="You are a relevance judge. Answer ONLY yes or no.",
            user_prompt=PRECISION_PROMPT.format(
                question=question, answer=answer, chunk=chunk[:1000]
            ),
            temperature=0.0,
            max_tokens=5,
        )
        relevance.append(1 if verdict.strip().lower().startswith("yes") else 0)

    total_relevant = sum(relevance)
    if total_relevant == 0:
        return 0.0

    # Weighted precision: reward relevant chunks appearing earlier
    score = 0.0
    running_relevant = 0
    for k, rel in enumerate(relevance, start=1):
        running_relevant += rel
        if rel == 1:
            precision_at_k = running_relevant / k
            score += precision_at_k

    return score / total_relevant
