"""Reranking: Cross-encoder or LLM-based (Bedrock Claude) reranking."""
from bedrock_llm import chat_completion

RERANK_PROMPT = """Score the relevance of the following document chunk to the query on a scale of 0-10.
Return ONLY the numeric score.

Query: {query}
Document: {document}
Relevance score:"""


def llm_rerank(query: str, documents: list[dict], top_k: int = 5) -> list[dict]:
    """Rerank documents using Bedrock Claude as a judge."""
    scored = []
    for doc in documents:
        response = chat_completion(
            system_prompt="You are a relevance scoring assistant. Return ONLY a number 0-10.",
            user_prompt=RERANK_PROMPT.format(query=query, document=doc["content"][:1000]),
            temperature=0.0,
            max_tokens=5,
        )
        try:
            score = float(response.strip())
        except ValueError:
            score = 0.0
        doc["rerank_score"] = score
        scored.append(doc)

    scored.sort(key=lambda x: x["rerank_score"], reverse=True)
    return scored[:top_k]


def cross_encoder_rerank(query: str, documents: list[dict], top_k: int = 5) -> list[dict]:
    """Rerank using a cross-encoder model (sentence-transformers)."""
    from sentence_transformers import CrossEncoder

    model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    pairs = [(query, doc["content"][:512]) for doc in documents]
    scores = model.predict(pairs)

    for doc, score in zip(documents, scores):
        doc["rerank_score"] = float(score)

    documents.sort(key=lambda x: x["rerank_score"], reverse=True)
    return documents[:top_k]
