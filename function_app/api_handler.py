"""AWS Lambda handler for HTTP APIs — Upload URL + Queue Processing + RAG Query with MetricsRetrieval."""
import os
import sys
import json
import logging
import time
from datetime import datetime, timezone

import boto3

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "rag"))

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Install dashboard metrics handler
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "dashboard"))
try:
    from metrics_handler import install as install_dashboard
    install_dashboard()
except ImportError:
    pass

# Structured retrieval metrics logger
metrics_logger = logging.getLogger("MetricsRetrieval")
metrics_logger.setLevel(logging.INFO)
if not metrics_logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(name)s | %(message)s"))
    metrics_logger.addHandler(handler)

s3_client = boto3.client("s3")
sqs_client = boto3.client("sqs")


def _ms_since(start: float) -> float:
    return round((time.time() - start) * 1000, 2)


def _response(status_code: int, body: dict) -> dict:
    return {
        "statusCode": status_code,
        "headers": {
            "Content-Type": "application/json",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "POST,OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type",
        },
        "body": json.dumps(body),
    }


def get_upload_url_handler(event, context):
    """Generate a presigned S3 URL for direct file upload."""
    try:
        body = json.loads(event.get("body", "{}"))
        filename = body["filename"]
        bucket = os.environ.get("S3_BUCKET_NAME", "rag-documents")
        expiry = int(os.environ.get("S3_PRESIGNED_EXPIRY", 1800))

        url = s3_client.generate_presigned_url(
            "put_object",
            Params={
                "Bucket": bucket,
                "Key": filename,
                "ContentType": body.get("content_type", "application/octet-stream"),
            },
            ExpiresIn=expiry,
        )
        return _response(200, {"upload_url": url})
    except Exception as e:
        logger.error(f"get-upload-url error: {e}")
        return _response(500, {"error": str(e)})


def queue_processing_handler(event, context):
    """Send a message to SQS to queue document processing."""
    try:
        body = json.loads(event.get("body", "{}"))
        queue_url = os.environ["SQS_QUEUE_URL"]

        sqs_client.send_message(
            QueueUrl=queue_url,
            MessageBody=json.dumps(body),
        )
        return _response(200, {"status": "queued", "blob_name": body.get("blob_name")})
    except Exception as e:
        logger.error(f"queue-processing error: {e}")
        return _response(500, {"error": str(e)})


def query_handler(event, context):
    """Execute the full RAG query pipeline with MetricsRetrieval logging."""
    retrieval_start = time.time()
    metrics = {"timestamp": datetime.now(timezone.utc).isoformat()}

    try:
        from query_rewriter import rewrite_query
        from retriever import hybrid_search
        from reranker import llm_rerank, cross_encoder_rerank
        from context_manager import build_context
        from access_control import get_access_filter
        from metadata_filter import build_filter, combine_filters
        from bedrock_llm import chat_completion_with_usage

        body = json.loads(event.get("body", "{}"))
        question = body["question"]
        user_role = body.get("user_role", "employee")
        user_department = body.get("user_department")
        reranker_type = body.get("reranker", "llm")
        metadata_criteria = body.get("metadata_criteria")

        metrics["user_role"] = user_role
        metrics["user_department"] = user_department

        # ── 1. Query Rewriting ──
        rewrite_start = time.time()
        rewritten = rewrite_query(question)
        metrics["query_rewrite"] = {
            "duration_ms": _ms_since(rewrite_start),
            "original_query": question,
            "rewritten_query": rewritten,
        }

        # ── 2. Build Filters ──
        access_filter = get_access_filter(user_role, user_department)
        meta_filter = build_filter(metadata_criteria) if metadata_criteria else None
        combined_filter = combine_filters(access_filter, meta_filter)
        metrics["filter_applied"] = combined_filter is not None

        # ── 3. Hybrid Retrieval (embedding + search) ──
        embed_start = time.time()
        from embedder import get_embeddings
        query_vector = get_embeddings([rewritten])[0]
        metrics["query_embedding"] = {
            "duration_ms": _ms_since(embed_start),
        }

        search_start = time.time()
        retrieved = hybrid_search(rewritten, top_k=20, filters=combined_filter)
        search_scores = [doc.get("score", 0) for doc in retrieved]

        metrics["search"] = {
            "duration_ms": _ms_since(search_start),
            "total_results_returned": len(retrieved),
            "max_score": round(max(search_scores), 4) if search_scores else 0,
            "min_score": round(min(search_scores), 4) if search_scores else 0,
        }

        # ── 4. Reranking ──
        rerank_start = time.time()
        if reranker_type == "cross-encoder":
            ranked = cross_encoder_rerank(rewritten, retrieved, top_k=5)
        else:
            ranked = llm_rerank(rewritten, retrieved, top_k=5)

        rerank_scores = [doc.get("rerank_score", 0) for doc in ranked]
        metrics["rerank"] = {
            "duration_ms": _ms_since(rerank_start),
            "reranker_type": reranker_type,
            "docs_before_rerank": len(retrieved),
            "docs_after_rerank": len(ranked),
            "top_rerank_score": round(max(rerank_scores), 4) if rerank_scores else 0,
            "min_rerank_score": round(min(rerank_scores), 4) if rerank_scores else 0,
            "rerank_score_spread": round(
                max(rerank_scores) - min(rerank_scores), 4
            ) if rerank_scores else 0,
        }

        # ── 5. Context Management ──
        context_start = time.time()
        context_result = build_context(ranked, max_tokens=4000, window_size=1)
        context_text = context_result["text"]

        unique_sources = list({doc.get("source", "") for doc in ranked if doc.get("source")})

        metrics["context"] = {
            "duration_ms": _ms_since(context_start),
            "token_count": context_result["token_count"],
            "chunks_used": context_result["chunks_used"],
            "truncated": context_result["truncated"],
            "unique_sources": len(unique_sources),
        }

        # ── 6. Generate Answer (with token/cost tracking) ──
        gen_start = time.time()
        system_prompt = (
            "You are a helpful assistant that answers questions based on the provided context. "
            "Always cite the source document when possible. "
            "If the context doesn't contain enough information, say so clearly."
        )
        user_prompt = f"Context:\n{context_text}\n\nQuestion: {question}"
        gen_result = chat_completion_with_usage(
            system_prompt, user_prompt, temperature=0.1, max_tokens=1000, caller="query_generation"
        )
        answer = gen_result["text"]

        metrics["generation"] = {
            "duration_ms": _ms_since(gen_start),
            "input_tokens": gen_result["input_tokens"],
            "output_tokens": gen_result["output_tokens"],
            "total_tokens": gen_result["total_tokens"],
            "cost_usd": gen_result["cost_usd"],
            "model_id": gen_result["model_id"],
        }

        # ── Overall ──
        metrics["total_retrieval_duration_ms"] = _ms_since(retrieval_start)
        metrics["status"] = "success"
        metrics_logger.info(json.dumps(metrics))

        return _response(200, {
            "question": question,
            "rewritten_query": rewritten,
            "answer": answer,
            "sources": unique_sources,
            "retrieved_chunks": len(ranked),
            "usage": {
                "generation_tokens": gen_result["total_tokens"],
                "generation_cost_usd": gen_result["cost_usd"],
            },
        })

    except Exception as e:
        metrics["status"] = "error"
        metrics["error"] = str(e)
        metrics["total_retrieval_duration_ms"] = _ms_since(retrieval_start)
        metrics_logger.info(json.dumps(metrics))
        logger.error(f"query error: {e}")
        return _response(500, {"error": str(e)})
