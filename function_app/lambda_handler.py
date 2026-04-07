"""AWS Lambda handlers — Ingest pipeline (SQS trigger) with MetricsIngestion logging."""
import os
import sys
import json
import logging
import time
from datetime import datetime, timezone

import boto3

# Add parent paths
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "rag"))

from parser import parse_document
from chunker import chunk_text
from embedder import get_embeddings
from metadata_extractor import extract_metadata
from indexer import index_chunks

# Install dashboard metrics handler
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "dashboard"))
try:
    from metrics_handler import install as install_dashboard
    install_dashboard()
except ImportError:
    pass

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Structured ingestion metrics logger
metrics_logger = logging.getLogger("MetricsIngestion")
metrics_logger.setLevel(logging.INFO)
if not metrics_logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(name)s | %(message)s"))
    metrics_logger.addHandler(handler)

s3_client = boto3.client("s3")


def _ms_since(start: float) -> float:
    return round((time.time() - start) * 1000, 2)


# ─── SQS Trigger: Process uploaded documents ───

def process_document_handler(event, context):
    """Lambda handler triggered by SQS — processes documents from the queue."""
    for record in event.get("Records", []):
        ingestion_start = time.time()
        metrics = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "document": None,
            "file_type": None,
            "document_size_bytes": 0,
        }

        try:
            message = json.loads(record["body"])
            blob_name = message["blob_name"]
            metadata = message.get("metadata", {})
            file_type = blob_name.rsplit(".", 1)[-1].lower() if "." in blob_name else "unknown"

            metrics["document"] = blob_name
            metrics["file_type"] = file_type
            logger.info(f"Processing: {blob_name}")

            # ── Download from S3 ──
            bucket = os.environ.get("S3_BUCKET_NAME", "rag-documents")
            response = s3_client.get_object(Bucket=bucket, Key=blob_name)
            content = response["Body"].read()
            metrics["document_size_bytes"] = len(content)

            # ── Parse ──
            parse_start = time.time()
            parse_success = True
            try:
                text = parse_document(content, blob_name)
            except Exception as e:
                parse_success = False
                text = ""
                logger.error(f"Parse failed for {blob_name}: {e}")

            metrics["parse"] = {
                "duration_ms": _ms_since(parse_start),
                "success": parse_success,
                "char_count": len(text),
            }

            if not parse_success or len(text) == 0:
                metrics["parse"]["warning"] = "empty_or_failed"
                metrics["total_ingestion_duration_ms"] = _ms_since(ingestion_start)
                metrics_logger.info(json.dumps(metrics))
                continue

            # ── Chunk ──
            chunk_start = time.time()
            chunks = chunk_text(text, chunk_size=512, chunk_overlap=128)
            empty_chunks = sum(1 for c in chunks if not c["content"].strip())
            avg_tokens = (
                round(sum(c["token_count"] for c in chunks) / len(chunks), 1)
                if chunks else 0
            )

            metrics["chunk"] = {
                "duration_ms": _ms_since(chunk_start),
                "chunk_count": len(chunks),
                "avg_chunk_tokens": avg_tokens,
                "empty_chunk_count": empty_chunks,
            }

            # ── Embed (batch) ──
            embed_start = time.time()
            texts = [c["content"] for c in chunks]
            batch_size = 16
            all_embeddings = []
            embed_batch_count = 0

            for i in range(0, len(texts), batch_size):
                batch = texts[i : i + batch_size]
                all_embeddings.extend(get_embeddings(batch))
                embed_batch_count += 1

            embedding_dim = len(all_embeddings[0]) if all_embeddings else 0

            metrics["embed"] = {
                "duration_ms": _ms_since(embed_start),
                "embedding_dimension": embedding_dim,
                "batch_count": embed_batch_count,
                "total_embeddings": len(all_embeddings),
            }

            # ── Extract metadata ──
            doc_metadata = extract_metadata(blob_name, metadata)

            # ── Index to OpenSearch ──
            index_start = time.time()
            index_result = index_chunks(chunks, all_embeddings, doc_metadata)
            index_succeeded = index_result["succeeded"]
            index_failed = index_result["failed"]

            metrics["index"] = {
                "duration_ms": _ms_since(index_start),
                "success_count": index_succeeded,
                "failure_count": index_failed,
            }

            # ── Overall ──
            metrics["total_ingestion_duration_ms"] = _ms_since(ingestion_start)
            metrics["status"] = "success" if index_failed == 0 else "partial_failure"

            metrics_logger.info(json.dumps(metrics))
            logger.info(
                f"Indexed {index_succeeded}/{len(chunks)} chunks for {blob_name} "
                f"in {metrics['total_ingestion_duration_ms']}ms"
            )

        except Exception as e:
            metrics["status"] = "error"
            metrics["error"] = str(e)
            metrics["total_ingestion_duration_ms"] = _ms_since(ingestion_start)
            metrics_logger.info(json.dumps(metrics))
            logger.error(f"Failed to process document: {e}")
            raise

    return {"statusCode": 200, "body": "OK"}
