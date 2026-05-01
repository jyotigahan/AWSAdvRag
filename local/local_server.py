"""Fully offline local server — no AWS services required.

Uses:
  - Ollama for LLM (chat + reranking)
  - sentence-transformers for embeddings
  - ChromaDB for vector storage
  - Local filesystem for document storage
"""
import os
import sys
import json
import http.server
import socketserver
import logging
import time
from datetime import datetime, timezone
from urllib.parse import urlparse

# Setup paths
LOCAL_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(LOCAL_DIR)
sys.path.insert(0, LOCAL_DIR)
sys.path.insert(0, os.path.join(PROJECT_DIR, "function_app"))
sys.path.insert(0, os.path.join(PROJECT_DIR, "rag"))

# Monkey-patch imports: replace AWS modules with local ones BEFORE importing rag modules
import local_embedder as embedder_mod
import ollama_llm as bedrock_llm_mod
import chroma_store

sys.modules["embedder"] = embedder_mod
sys.modules["bedrock_llm"] = bedrock_llm_mod

from parser import parse_document
from chunker import chunk_text
from context_manager import build_context
from access_control import get_access_filter
from metadata_filter import build_filter, combine_filters

logger = logging.getLogger("LocalServer")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s | %(message)s")

PORT = 8080
UI_DIR = os.path.join(PROJECT_DIR, "ui")
UPLOAD_DIR = os.path.join(LOCAL_DIR, "uploaded_docs")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Reranker prompt (same as rag/reranker.py)
RERANK_PROMPT = """Score the relevance of the following document chunk to the query on a scale of 0-10.
Return ONLY the numeric score.

Query: {query}
Document: {document}
Relevance score:"""

REWRITE_PROMPT = """Given a user query, rewrite it to be more specific and effective for document retrieval.
Return ONLY the rewritten query, nothing else.
If the query is already clear and specific, return it unchanged.

User query: {query}
Rewritten query:"""


def _ms_since(start: float) -> float:
    return round((time.time() - start) * 1000, 2)


def rewrite_query(query: str) -> str:
    return bedrock_llm_mod.chat_completion(
        system_prompt="You are a query rewriting assistant.",
        user_prompt=REWRITE_PROMPT.format(query=query),
        temperature=0.0, max_tokens=200,
    )


def llm_rerank(query: str, documents: list[dict], top_k: int = 5) -> list[dict]:
    scored = []
    for doc in documents:
        response = bedrock_llm_mod.chat_completion(
            system_prompt="You are a relevance scoring assistant. Return ONLY a number 0-10.",
            user_prompt=RERANK_PROMPT.format(query=query, document=doc["content"][:1000]),
            temperature=0.0, max_tokens=5,
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
    """Fallback: use LLM reranking since sentence-transformers requires torch."""
    logger.info("Cross-encoder not available locally, falling back to LLM reranker")
    return llm_rerank(query, documents, top_k)


def ingest_document(filepath: str, filename: str, metadata: dict) -> dict:
    """Full ingestion pipeline: parse → chunk → embed → index to ChromaDB."""
    start = time.time()
    with open(filepath, "rb") as f:
        content = f.read()

    text = parse_document(content, filename)
    if not text.strip():
        return {"status": "error", "error": "Empty document after parsing"}

    chunks = chunk_text(text, chunk_size=512, chunk_overlap=128)
    texts = [c["content"] for c in chunks]
    embeddings = embedder_mod.get_embeddings(texts, caller="ingestion")

    doc_metadata = {
        "source": filename,
        "department": metadata.get("department", "general"),
        "access_level": metadata.get("access_level", "public"),
        "author": metadata.get("author", "unknown"),
        "date_uploaded": datetime.now(timezone.utc).isoformat(),
    }
    doc_metadata.update(metadata)

    result = chroma_store.index_chunks(chunks, embeddings, doc_metadata)
    duration = _ms_since(start)

    logger.info(f"Ingested {filename}: {result['succeeded']} chunks in {duration}ms")
    return {"status": "success", "chunks": result["succeeded"], "duration_ms": duration}


class LocalRAGHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=UI_DIR, **kwargs)

    def log_message(self, format, *args):
        logger.info(f"{self.address_string()} - {format % args}")

    def do_OPTIONS(self):
        self.send_response(200)
        self._cors_headers()
        self.end_headers()

    def do_POST(self):
        path = urlparse(self.path).path
        content_len = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(content_len) if content_len else b""

        # Check if this is a multipart file upload
        content_type = self.headers.get("Content-Type", "")

        if path == "/api/upload":
            self._handle_direct_upload(raw, content_type)
        elif path == "/api/get-upload-url":
            # For local mode, return a local upload endpoint
            body = json.loads(raw) if raw else {}
            self._json_response(200, {
                "upload_url": f"http://localhost:{PORT}/api/upload?filename={body.get('filename', 'doc.txt')}",
                "local_mode": True,
            })
        elif path == "/api/queue-processing":
            body = json.loads(raw) if raw else {}
            self._handle_process(body)
        elif path == "/api/query":
            body = json.loads(raw) if raw else {}
            self._handle_query(body)
        else:
            self._json_response(404, {"error": "not found"})

    def do_PUT(self):
        """Handle PUT requests for presigned-URL-style uploads."""
        path = urlparse(self.path).path
        if path == "/api/upload":
            content_len = int(self.headers.get("Content-Length", 0))
            raw = self.rfile.read(content_len) if content_len else b""
            content_type = self.headers.get("Content-Type", "")
            self._handle_direct_upload(raw, content_type)
        else:
            self._json_response(404, {"error": "not found"})

    def _handle_direct_upload(self, raw_body: bytes, content_type: str):
        """Save uploaded file to local disk."""
        from urllib.parse import parse_qs, urlparse
        query = parse_qs(urlparse(self.path).query)
        filename = query.get("filename", ["uploaded_doc.txt"])[0]

        filepath = os.path.join(UPLOAD_DIR, filename)
        with open(filepath, "wb") as f:
            f.write(raw_body)

        logger.info(f"Saved upload: {filename} ({len(raw_body)} bytes)")
        self._json_response(200, {"status": "saved", "filename": filename})

    def _handle_process(self, body: dict):
        """Process a document that was already uploaded to local disk."""
        try:
            blob_name = body.get("blob_name", "")
            metadata = body.get("metadata", {})
            filepath = os.path.join(UPLOAD_DIR, blob_name)

            if not os.path.exists(filepath):
                self._json_response(404, {"error": f"File not found: {blob_name}"})
                return

            result = ingest_document(filepath, blob_name, metadata)
            self._json_response(200, {
                "status": result["status"],
                "blob_name": blob_name,
                "chunks_indexed": result.get("chunks", 0),
                "duration_ms": result.get("duration_ms", 0),
            })
        except Exception as e:
            logger.error(f"Process error: {e}")
            self._json_response(500, {"error": str(e)})

    def _handle_query(self, body: dict):
        """Full RAG query pipeline using local services."""
        start = time.time()
        try:
            question = body["question"]
            user_role = body.get("user_role", "employee")
            user_department = body.get("user_department")
            reranker_type = body.get("reranker", "llm")

            # 1. Rewrite query
            rewritten = rewrite_query(question)

            # 2. Build filters
            access_filter = get_access_filter(user_role, user_department)
            meta_filter = build_filter(body.get("metadata_criteria")) if body.get("metadata_criteria") else None
            combined_filter = combine_filters(access_filter, meta_filter)

            # 3. Embed query + search ChromaDB
            query_embedding = embedder_mod.get_embeddings([rewritten], caller="query")[0]
            retrieved = chroma_store.hybrid_search(
                rewritten, query_embedding, top_k=20, filters=combined_filter,
            )

            if not retrieved:
                self._json_response(200, {
                    "question": question,
                    "rewritten_query": rewritten,
                    "answer": "No documents found. Please upload some documents first.",
                    "sources": [],
                    "retrieved_chunks": 0,
                    "eval_scores": {},
                })
                return

            # 4. Rerank
            if reranker_type == "cross-encoder":
                ranked = cross_encoder_rerank(rewritten, retrieved, top_k=5)
            else:
                ranked = llm_rerank(rewritten, retrieved, top_k=5)

            # 5. Build context
            context_result = build_context(ranked, max_tokens=4000, window_size=1)
            context_text = context_result["text"]

            # 6. Generate answer
            system_prompt = (
                "You are a helpful assistant that answers questions based on the provided context. "
                "Always cite the source document when possible. "
                "If the context doesn't contain enough information, say so clearly."
            )
            gen_result = bedrock_llm_mod.chat_completion_with_usage(
                system_prompt,
                f"Context:\n{context_text}\n\nQuestion: {question}",
                temperature=0.1, max_tokens=1000, caller="query_generation",
            )
            answer = gen_result["text"]
            sources = list({doc.get("source", "") for doc in ranked if doc.get("source")})

            duration = _ms_since(start)
            logger.info(f"Query answered in {duration}ms")

            self._json_response(200, {
                "question": question,
                "rewritten_query": rewritten,
                "answer": answer,
                "sources": sources,
                "retrieved_chunks": len(ranked),
                "usage": {
                    "generation_tokens": gen_result["total_tokens"],
                    "generation_cost_usd": 0.0,
                },
                "eval_scores": {},
            })
        except Exception as e:
            logger.error(f"Query error: {e}", exc_info=True)
            self._json_response(500, {"error": str(e)})

    def _json_response(self, code: int, body: dict):
        self.send_response(code)
        self._cors_headers()
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(body).encode())

    def _cors_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET,POST,PUT,OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")


if __name__ == "__main__":
    print("=" * 60)
    print("  RAG Local Mode — Fully Offline")
    print("=" * 60)
    print(f"  LLM:        Ollama (llama3.2)")
    print(f"  Embeddings: sentence-transformers (all-MiniLM-L6-v2)")
    print(f"  VectorDB:   ChromaDB (local)")
    print(f"  Storage:    {UPLOAD_DIR}")
    print("=" * 60)
    print(f"  Upload UI:  http://localhost:{PORT}/index.html")
    print(f"  Query UI:   http://localhost:{PORT}/query.html")
    print("=" * 60)

    with socketserver.TCPServer(("", PORT), LocalRAGHandler) as httpd:
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down.")
