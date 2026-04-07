"""RAGAS Answer Similarity — measures semantic similarity between generated answer and ground truth."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "rag"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "function_app"))
from bedrock_llm import chat_completion
from embedder import get_embeddings


def evaluate_answer_similarity(
    question: str, answer: str, ground_truth: str, method: str = "embedding"
) -> float:
    """RAGAS Answer Similarity: semantic similarity between generated and reference answer.

    Two methods:
    - 'embedding': cosine similarity of Titan embeddings (fast, cheap)
    - 'llm': LLM judge score (slower, more nuanced)
    """
    if method == "embedding":
        return _embedding_similarity(answer, ground_truth)
    else:
        return _llm_similarity(answer, ground_truth)


def _embedding_similarity(text_a: str, text_b: str) -> float:
    """Cosine similarity between two text embeddings."""
    embeddings = get_embeddings([text_a, text_b])
    vec_a = embeddings[0]
    vec_b = embeddings[1]

    dot_product = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = sum(a * a for a in vec_a) ** 0.5
    norm_b = sum(b * b for b in vec_b) ** 0.5

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return dot_product / (norm_a * norm_b)


SIMILARITY_PROMPT = """Rate the semantic similarity between the following two answers on a scale of 0.0 to 1.0.
Consider meaning, not exact wording. Two answers that convey the same information should score high.

Answer A: {answer_a}
Answer B: {answer_b}

Return ONLY the numeric score (0.0 to 1.0):"""


def _llm_similarity(text_a: str, text_b: str) -> float:
    """LLM-judged semantic similarity."""
    response = chat_completion(
        system_prompt="You are a semantic similarity scorer. Return ONLY a number between 0.0 and 1.0.",
        user_prompt=SIMILARITY_PROMPT.format(answer_a=text_a, answer_b=text_b),
        temperature=0.0,
        max_tokens=5,
    )
    try:
        score = float(response.strip())
        return max(0.0, min(1.0, score))
    except ValueError:
        return 0.0
