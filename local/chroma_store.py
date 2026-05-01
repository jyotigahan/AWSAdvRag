"""ChromaDB vector store — replaces OpenSearch Serverless for local mode."""
import os
import hashlib
import logging
import chromadb

logger = logging.getLogger(__name__)

_client = None
_collection = None

CHROMA_DIR = os.path.join(os.path.dirname(__file__), "chroma_data")
COLLECTION_NAME = "rag-index"


def _get_collection():
    global _client, _collection
    if _collection is None:
        _client = chromadb.PersistentClient(path=CHROMA_DIR)
        _collection = _client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
    return _collection


def index_chunks(chunks: list[dict], embeddings: list[list[float]], metadata: dict) -> dict:
    """Index chunks with embeddings into ChromaDB."""
    collection = _get_collection()
    succeeded = 0
    failed = 0

    ids = []
    documents = []
    metadatas = []
    embs = []

    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        doc_id = hashlib.sha256(f"{metadata['source']}_{i}".encode()).hexdigest()[:32]
        ids.append(doc_id)
        documents.append(chunk["content"])
        metadatas.append({
            "source": metadata.get("source", ""),
            "department": metadata.get("department", "general"),
            "access_level": metadata.get("access_level", "public"),
            "author": metadata.get("author", "unknown"),
            "chunk_index": chunk["chunk_index"],
            "total_chunks": chunk["total_chunks"],
            "date_uploaded": metadata.get("date_uploaded", ""),
        })
        embs.append(embedding)

    try:
        collection.upsert(ids=ids, documents=documents, metadatas=metadatas, embeddings=embs)
        succeeded = len(ids)
    except Exception as e:
        failed = len(ids)
        logger.error(f"Failed to index chunks: {e}")

    return {"succeeded": succeeded, "failed": failed}


def hybrid_search(
    query: str,
    query_embedding: list[float],
    top_k: int = 20,
    filters: dict | None = None,
) -> list[dict]:
    """Search ChromaDB using vector similarity + optional metadata filters."""
    collection = _get_collection()

    where_filter = None
    if filters:
        # Build ChromaDB where filter from our access control filters
        conditions = []
        for f in filters:
            if "terms" in f:
                for field, values in f["terms"].items():
                    conditions.append({"$or": [{field: {"$eq": v}} for v in values]})
            elif "term" in f:
                for field, value in f["term"].items():
                    conditions.append({field: {"$eq": value}})

        if len(conditions) == 1:
            where_filter = conditions[0]
        elif len(conditions) > 1:
            where_filter = {"$and": conditions}

    try:
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, collection.count()) if collection.count() > 0 else top_k,
            where=where_filter if where_filter else None,
        )
    except Exception as e:
        logger.warning(f"ChromaDB query with filter failed ({e}), retrying without filter")
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, collection.count()) if collection.count() > 0 else top_k,
        )

    docs = []
    if results and results["ids"] and results["ids"][0]:
        for i, doc_id in enumerate(results["ids"][0]):
            meta = results["metadatas"][0][i] if results["metadatas"] else {}
            content = results["documents"][0][i] if results["documents"] else ""
            distance = results["distances"][0][i] if results["distances"] else 0
            score = 1.0 - distance  # ChromaDB returns distances, convert to similarity

            docs.append({
                "id": doc_id,
                "content": content,
                "source": meta.get("source", ""),
                "department": meta.get("department", ""),
                "access_level": meta.get("access_level", ""),
                "chunk_index": meta.get("chunk_index", 0),
                "total_chunks": meta.get("total_chunks", 1),
                "score": score,
            })

    return docs
