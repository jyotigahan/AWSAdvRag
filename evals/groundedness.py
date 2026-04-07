"""Groundedness evaluation — checks if the answer is grounded in the context."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "rag"))
from bedrock_llm import chat_completion

GROUNDEDNESS_PROMPT = """Score how well the answer is grounded in the provided context.

Context: {context}
Question: {question}
Answer: {answer}

Score from 1-5:
1 = Completely ungrounded, fabricated information
2 = Mostly ungrounded, some fabrication
3 = Partially grounded, mix of grounded and ungrounded
4 = Mostly grounded, minor unsupported claims
5 = Fully grounded in the context

Return ONLY the numeric score."""


def evaluate_groundedness(question: str, answer: str, context: str) -> float:
    response = chat_completion(
        system_prompt="You are an evaluation assistant. Return ONLY a number 1-5.",
        user_prompt=GROUNDEDNESS_PROMPT.format(context=context, question=question, answer=answer),
        temperature=0.0,
        max_tokens=5,
    )
    try:
        return float(response.strip())
    except ValueError:
        return 0.0
