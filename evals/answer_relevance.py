"""RAGAS Answer Relevance — measures if the answer addresses the question."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "rag"))
from bedrock_llm import chat_completion

GENERATE_QUESTIONS_PROMPT = """Given the following answer, generate {n} questions that this answer could be responding to.
Return ONLY the questions, one per line, numbered.

Answer: {answer}

Questions:"""

SIMILARITY_PROMPT = """Rate the semantic similarity between these two questions on a scale of 0.0 to 1.0.
Return ONLY the numeric score.

Question 1: {q1}
Question 2: {q2}

Similarity score:"""


def evaluate_answer_relevance(question: str, answer: str, n_generated: int = 3) -> float:
    """RAGAS Answer Relevance: measures how well the answer addresses the original question.

    1. Generate N questions that the answer could plausibly respond to.
    2. Compute semantic similarity between each generated question and the original.
    3. Score = average similarity.
    """
    # Step 1: Generate questions from the answer
    gen_text = chat_completion(
        system_prompt="You are a question generation assistant.",
        user_prompt=GENERATE_QUESTIONS_PROMPT.format(answer=answer, n=n_generated),
        temperature=0.3,
        max_tokens=300,
    )

    generated_questions = [
        line.strip().lstrip("0123456789.-) ").strip()
        for line in gen_text.strip().split("\n")
        if line.strip() and any(c.isalpha() for c in line)
    ][:n_generated]

    if not generated_questions:
        return 0.0

    # Step 2: Score similarity of each generated question to the original
    similarities = []
    for gq in generated_questions:
        sim_text = chat_completion(
            system_prompt="You are a semantic similarity scorer. Return ONLY a number between 0.0 and 1.0.",
            user_prompt=SIMILARITY_PROMPT.format(q1=question, q2=gq),
            temperature=0.0,
            max_tokens=5,
        )
        try:
            sim = float(sim_text.strip())
            similarities.append(max(0.0, min(1.0, sim)))
        except ValueError:
            similarities.append(0.0)

    return sum(similarities) / len(similarities)
