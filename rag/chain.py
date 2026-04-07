"""Full RAG chain orchestrator."""
from query_rewriter import rewrite_query
from retriever import hybrid_search
from reranker import llm_rerank, cross_encoder_rerank
from context_manager import build_context
from access_control import get_access_filter
from metadata_filter import build_filter, combine_filters
from bedrock_llm import chat_completion

SYSTEM_PROMPT = """You are a helpful assistant that answers questions based on the provided context.
Always cite the source document when possible.
If the context doesn't contain enough information to answer, say so clearly.
Do not make up information."""

RAG_PROMPT = """Context:
{context}

Question: {question}

Answer based on the context above:"""


def rag_query(
    question: str,
    user_role: str = "employee",
    user_department: str | None = None,
    metadata_criteria: dict | None = None,
    reranker: str = "llm",
    top_k: int = 5,
    max_context_tokens: int = 4000,
) -> dict:
    """Execute the full RAG pipeline."""
    # 1. Query rewriting
    rewritten = rewrite_query(question)

    # 2. Build filters (access control + metadata)
    access_filter = get_access_filter(user_role, user_department)
    meta_filter = build_filter(metadata_criteria) if metadata_criteria else None
    combined_filter = combine_filters(access_filter, meta_filter)

    # 3. Hybrid retrieval (vector kNN + BM25)
    retrieved = hybrid_search(rewritten, top_k=top_k * 4, filters=combined_filter)

    # 4. Reranking
    if reranker == "cross-encoder":
        ranked = cross_encoder_rerank(rewritten, retrieved, top_k=top_k)
    else:
        ranked = llm_rerank(rewritten, retrieved, top_k=top_k)

    # 5. Context management (windowing + token budget)
    context_result = build_context(ranked, max_tokens=max_context_tokens, window_size=1)
    context_text = context_result["text"]

    # 6. Generate answer via Bedrock
    answer = chat_completion(
        system_prompt=SYSTEM_PROMPT,
        user_prompt=RAG_PROMPT.format(context=context_text, question=question),
        temperature=0.1,
        max_tokens=1000,
    )

    sources = list({doc.get("source", "") for doc in ranked if doc.get("source")})

    return {
        "question": question,
        "rewritten_query": rewritten,
        "answer": answer,
        "sources": sources,
        "retrieved_chunks": len(ranked),
        "context_used": context_text[:500] + "..." if len(context_text) > 500 else context_text,
    }
