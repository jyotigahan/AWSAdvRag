"""Create the OpenSearch Serverless vector index."""
import os
import time
import boto3
from opensearchpy import OpenSearch, RequestsHttpConnection
from requests_aws4auth import AWS4Auth
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

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

client = OpenSearch(
    hosts=[{"host": host, "port": 443}],
    http_auth=awsauth,
    use_ssl=True,
    verify_certs=True,
    connection_class=RequestsHttpConnection,
)

index_name = os.environ.get("OPENSEARCH_INDEX_NAME", "rag-index")

# Titan Embeddings v2 produces 1024-dim vectors by default
index_body = {
    "settings": {
        "index": {
            "knn": True,
            "knn.algo_param.ef_search": 512,
        }
    },
    "mappings": {
        "properties": {
            "content": {"type": "text", "analyzer": "standard"},
            "content_vector": {
                "type": "knn_vector",
                "dimension": 1024,
                "method": {
                    "name": "hnsw",
                    "space_type": "cosinesimil",
                    "engine": "nmslib",
                    "parameters": {"ef_construction": 512, "m": 16},
                },
            },
            "source": {"type": "keyword"},
            "department": {"type": "keyword"},
            "access_level": {"type": "keyword"},
            "author": {"type": "keyword"},
            "page": {"type": "integer"},
            "chunk_index": {"type": "integer"},
            "total_chunks": {"type": "integer"},
            "date_uploaded": {"type": "date"},
        }
    },
}

if client.indices.exists(index=index_name):
    print(f"Index '{index_name}' already exists. Deleting...")
    client.indices.delete(index=index_name)
    time.sleep(2)

client.indices.create(index=index_name, body=index_body)
print(f"Index '{index_name}' created successfully.")
