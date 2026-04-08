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
    """Hybrid search: run kNN and BM25 separately, merge and deduplicate."""
    client = _get_opensearch_client()
    index_name = os.environ.get("OPENSEARCH_INDEX_NAME", "rag-index")

    query_vector = get_embeddings([query])[0]

    # Build filter
    filter_clauses = []
    if filters:
        filter_clauses = filters if isinstance(filters, list) else [filters]

    # 1. kNN vector search
    knn_body = {"size": top_k, "query": {"knn": {"content_vector": {"vector": query_vector, "k": top_k}}}}
    if filter_clauses:
        knn_body["query"] = {"bool": {"must": [knn_body["query"]], "filter": filter_clauses}}

    knn_results = client.search(index=index_name, body=knn_body)

    # 2. BM25 text search
    bm25_body = {"size": top_k, "query": {"match": {"content": {"query": query}}}}
    if filter_clauses:
        bm25_body["query"] = {"bool": {"must": [bm25_body["query"]], "filter": filter_clauses}}

    bm25_results = client.search(index=index_name, body=bm25_body)

    # 3. Merge and deduplicate (reciprocal rank fusion)
    doc_scores = {}
    doc_data = {}

    for rank, hit in enumerate(knn_results["hits"]["hits"]):
        doc_id = hit["_id"]
        doc_scores[doc_id] = doc_scores.get(doc_id, 0) + 1.0 / (rank + 1)
        doc_data[doc_id] = hit

    for rank, hit in enumerate(bm25_results["hits"]["hits"]):
        doc_id = hit["_id"]
        doc_scores[doc_id] = doc_scores.get(doc_id, 0) + 1.0 / (rank + 1)
        if doc_id not in doc_data:
            doc_data[doc_id] = hit

    # Sort by combined score
    sorted_ids = sorted(doc_scores.keys(), key=lambda x: doc_scores[x], reverse=True)[:top_k]

    docs = []
    for doc_id in sorted_ids:
        hit = doc_data[doc_id]
        source = hit["_source"]
        docs.append({
            "id": doc_id,
            "content": source.get("content", ""),
            "source": source.get("source", ""),
            "department": source.get("department", ""),
            "access_level": source.get("access_level", ""),
            "chunk_index": source.get("chunk_index", 0),
            "score": doc_scores[doc_id],
        })

    return docs
