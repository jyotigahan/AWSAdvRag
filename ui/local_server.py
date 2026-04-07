"""Local HTTP server that serves the UI and handles upload/query APIs."""
import os
import sys
import json
import http.server
import socketserver
from urllib.parse import urlparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "function_app"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "rag"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "dashboard"))

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

# Install dashboard metrics
try:
    from metrics_handler import install as install_dashboard
    install_dashboard()
except ImportError:
    pass

import boto3

PORT = 8080
UI_DIR = os.path.dirname(__file__)
s3_client = boto3.client("s3", region_name=os.environ.get("AWS_REGION", "us-east-1"))
sqs_client = boto3.client("sqs", region_name=os.environ.get("AWS_REGION", "us-east-1"))


class RAGHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=UI_DIR, **kwargs)

    def do_OPTIONS(self):
        self.send_response(200)
        self._cors_headers()
        self.end_headers()

    def do_POST(self):
        path = urlparse(self.path).path
        content_len = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(content_len)) if content_len else {}

        if path == "/api/get-upload-url":
            self._handle_upload_url(body)
        elif path == "/api/queue-processing":
            self._handle_queue(body)
        elif path == "/api/query":
            self._handle_query(body)
        else:
            self._json_response(404, {"error": "not found"})

    def _handle_upload_url(self, body):
        try:
            filename = body["filename"]
            bucket = os.environ.get("S3_BUCKET_NAME", "rag-documents")
            url = s3_client.generate_presigned_url(
                "put_object",
                Params={"Bucket": bucket, "Key": filename, "ContentType": body.get("content_type", "application/octet-stream")},
                ExpiresIn=1800,
            )
            self._json_response(200, {"upload_url": url})
        except Exception as e:
            self._json_response(500, {"error": str(e)})

    def _handle_queue(self, body):
        try:
            queue_url = os.environ["SQS_QUEUE_URL"]
            sqs_client.send_message(QueueUrl=queue_url, MessageBody=json.dumps(body))
            self._json_response(200, {"status": "queued", "blob_name": body.get("blob_name")})
        except Exception as e:
            self._json_response(500, {"error": str(e)})

    def _handle_query(self, body):
        try:
            from query_rewriter import rewrite_query
            from retriever import hybrid_search
            from reranker import llm_rerank, cross_encoder_rerank
            from context_manager import build_context
            from access_control import get_access_filter
            from metadata_filter import build_filter, combine_filters
            from bedrock_llm import chat_completion

            question = body["question"]
            user_role = body.get("user_role", "employee")
            user_department = body.get("user_department")
            reranker_type = body.get("reranker", "llm")

            rewritten = rewrite_query(question)
            access_filter = get_access_filter(user_role, user_department)
            retrieved = hybrid_search(rewritten, top_k=20, filters=access_filter)

            if reranker_type == "cross-encoder":
                ranked = cross_encoder_rerank(rewritten, retrieved, top_k=5)
            else:
                ranked = llm_rerank(rewritten, retrieved, top_k=5)

            context_result = build_context(ranked, max_tokens=4000, window_size=1)
            context_text = context_result["text"]

            system_prompt = "You are a helpful banking assistant. Answer based on the provided context. Cite sources. If context is insufficient, say so."
            answer = chat_completion(system_prompt, f"Context:\n{context_text}\n\nQuestion: {question}", temperature=0.1, max_tokens=1000)
            sources = list({doc.get("source", "") for doc in ranked if doc.get("source")})

            self._json_response(200, {
                "question": question,
                "rewritten_query": rewritten,
                "answer": answer,
                "sources": sources,
                "retrieved_chunks": len(ranked),
            })
        except Exception as e:
            self._json_response(500, {"error": str(e)})

    def _json_response(self, code, body):
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
    print(f"RAG UI running at http://localhost:{PORT}/index.html (upload)")
    print(f"                   http://localhost:{PORT}/query.html (query)")
    with socketserver.TCPServer(("", PORT), RAGHandler) as httpd:
        httpd.serve_forever()
