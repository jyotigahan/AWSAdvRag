"""Generate embeddings via AWS Bedrock (Titan Embeddings v2) with token and cost tracking."""
import os
import json
import logging
import boto3

# Structured LLM usage logger
metrics_logger = logging.getLogger("MetricsLLM")
metrics_logger.setLevel(logging.INFO)
if not metrics_logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(name)s | %(message)s"))
    metrics_logger.addHandler(handler)

# Titan Embeddings v2 pricing per 1K input tokens (us-east-1)
TITAN_EMBED_PRICE_PER_1K = 0.00002


def _get_bedrock_client():
    return boto3.client(
        "bedrock-runtime",
        region_name=os.environ.get("AWS_REGION", "us-east-1"),
    )


def get_embeddings(texts: list[str], caller: str = "unknown") -> list[list[float]]:
    """Get embeddings for a list of texts using Amazon Titan Embeddings v2.

    Args:
        caller: identifier for who's calling (e.g. 'ingestion', 'query')
    """
    client = _get_bedrock_client()
    model_id = os.environ.get("BEDROCK_EMBEDDING_MODEL_ID", "amazon.titan-embed-text-v2:0")
    embeddings = []
    total_input_tokens = 0

    for text in texts:
        response = client.invoke_model(
            modelId=model_id,
            contentType="application/json",
            accept="application/json",
            body=json.dumps({"inputText": text}),
        )
        result = json.loads(response["body"].read())
        embeddings.append(result["embedding"])
        total_input_tokens += result.get("inputTextTokenCount", 0)

    cost = total_input_tokens / 1000 * TITAN_EMBED_PRICE_PER_1K

    metrics_logger.info(json.dumps({
        "type": "embedding",
        "caller": caller,
        "model_id": model_id,
        "texts_count": len(texts),
        "total_input_tokens": total_input_tokens,
        "embedding_dimension": len(embeddings[0]) if embeddings else 0,
        "estimated_cost_usd": round(cost, 6),
    }))

    return embeddings
