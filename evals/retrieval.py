"""Retrieval evaluation — checks if retrieved chunks are relevant to the question."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "rag"))
from bedrock_llm import chat_completion

RETRIEVAL_PROMPT = """Score how relevant the retrieved document chunk is to the question.

Question: {question}
Retrieved chunk: {chunk}

Score from 1-5:
1 = Completely irrelevant
2 = Mostly irrelevant
3 = Somewhat relevant
4 = Mostly relevant
5 = Highly relevant

Return ONLY the numeric score."""


def evaluate_retrieval(question: str, chunks: list[str]) -> float:
    """Average retrieval relevance across all chunks."""
    if not chunks:
        return 0.0

    scores = []
    for chunk in chunks:
        response = chat_completion(
            system_prompt="You are an evaluation assistant. Return ONLY a number 1-5.",
            user_prompt=RETRIEVAL_PROMPT.format(question=question, chunk=chunk[:1000]),
            temperature=0.0,
            max_tokens=5,
        )
        try:
            scores.append(float(response.strip()))
        except ValueError:
            scores.append(0.0)

    return sum(scores) / len(scores) if scores else 0.0
