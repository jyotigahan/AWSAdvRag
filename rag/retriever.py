"""Hybrid retrieval: Vector (kNN) + BM25 via Amazon OpenSearch Serverless."""
import os
import sys
import boto3
from opensearchpy import OpenSearch, RequestsHttpConnection
from requests_aws4auth import AWS4Auth

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "function_app"))
from embedder import get_embeddings


def _get_opensearch_client() -> OpenSearch:
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


def hybrid_search(
    query: str,
    top_k: int = 20,
    filters: dict | None = None,
) -> list[dict]:
    """Hybrid search: vector kNN + BM25 text search on OpenSearch."""
    client = _get_opensearch_client()
    index_name = os.environ.get("OPENSEARCH_INDEX_NAME", "rag-index")

    query_vector = get_embeddings([query])[0]

    # OpenSearch Serverless uses knn inside the query bool, not top-level
    knn_clause = {
        "knn": {
            "content_vector": {
                "vector": query_vector,
                "k": top_k,
            }
        }
    }

    # Combine BM25 text match + kNN vector search
    must_clauses = [{"match": {"content": {"query": query, "boost": 0.3}}}]

    # Apply filters if provided
    filter_clauses = []
    if filters:
        filter_clauses = filters if isinstance(filters, list) else [filters]

    search_body = {
        "size": top_k,
        "query": {
            "bool": {
                "should": [
                    {"match": {"content": {"query": query, "boost": 0.3}}},
                    knn_clause,
                ],
                "filter": filter_clauses,
                "minimum_should_match": 1,
            }
        },
    }

    results = client.search(index=index_name, body=search_body)

    docs = []
    for hit in results["hits"]["hits"]:
        source = hit["_source"]
        docs.append({
            "id": hit["_id"],
            "content": source.get("content", ""),
            "source": source.get("source", ""),
            "department": source.get("department", ""),
            "access_level": source.get("access_level", ""),
            "chunk_index": source.get("chunk_index", 0),
            "score": hit.get("_score", 0),
        })

    return docs
