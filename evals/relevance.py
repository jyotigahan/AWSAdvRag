"""Relevance evaluation — checks if the answer is relevant to the question."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "rag"))
from bedrock_llm import chat_completion

RELEVANCE_PROMPT = """Score how relevant the answer is to the question.

Question: {question}
Answer: {answer}

Score from 1-5:
1 = Completely irrelevant
2 = Mostly irrelevant
3 = Partially relevant
4 = Mostly relevant
5 = Perfectly relevant and addresses the question

Return ONLY the numeric score."""


def evaluate_relevance(question: str, answer: str) -> float:
    response = chat_completion(
        system_prompt="You are an evaluation assistant. Return ONLY a number 1-5.",
        user_prompt=RELEVANCE_PROMPT.format(question=question, answer=answer),
        temperature=0.0,
        max_tokens=5,
    )
    try:
        return float(response.strip())
    except ValueError:
        return 0.0
