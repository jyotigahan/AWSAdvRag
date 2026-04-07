"""Push chunks + embeddings to Amazon OpenSearch Serverless."""
import os
import hashlib
import logging
from opensearchpy import OpenSearch, RequestsHttpConnection
from requests_aws4auth import AWS4Auth
import boto3

logger = logging.getLogger(__name__)


def _get_opensearch_client() -> OpenSearch:
    """Create an OpenSearch client with AWS SigV4 auth."""
    region = os.environ.get("AWS_REGION", "us-east-1")
    credentials = boto3.Session().get_credentials()
    awsauth = AWS4Auth(
        credentials.access_key,
        credentials.secret_key,
        region,
        "aoss",
        session_token=credentials.token,
    )
    endpoint = os.environ["OPENSEARCH_ENDPOINT"]
    host = endpoint.replace("https://", "").replace("http://", "")

    return OpenSearch(
        hosts=[{"host": host, "port": 443}],
        http_auth=awsauth,
        use_ssl=True,
        verify_certs=True,
        connection_class=RequestsHttpConnection,
    )


def index_chunks(
    chunks: list[dict],
    embeddings: list[list[float]],
    metadata: dict,
) -> dict:
    """Upload chunks with embeddings and metadata to OpenSearch.

    Returns:
        dict with 'succeeded' and 'failed' counts.
    """
    client = _get_opensearch_client()
    index_name = os.environ.get("OPENSEARCH_INDEX_NAME", "rag-index")
    succeeded = 0
    failed = 0

    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        doc_id = hashlib.sha256(
            f"{metadata['source']}_{i}".encode()
        ).hexdigest()[:32]

        document = {
            "content": chunk["content"],
            "content_vector": embedding,
            "source": metadata.get("source", ""),
            "department": metadata.get("department", "general"),
            "access_level": metadata.get("access_level", "public"),
            "author": metadata.get("author", "unknown"),
            "page": metadata.get("page", 0),
            "chunk_index": chunk["chunk_index"],
            "total_chunks": chunk["total_chunks"],
            "date_uploaded": metadata.get("date_uploaded"),
        }

        try:
            client.index(index=index_name, body=document)
            succeeded += 1
        except Exception as e:
            failed += 1
            logger.error(f"Failed to index chunk {i}: {e}")

    return {"succeeded": succeeded, "failed": failed}
