"""Ingest banking test documents into OpenSearch for evaluation.

Usage:
    cd AWSAdvRag
    python test_data/ingest_test_docs.py
"""
import os
import sys
import glob

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "function_app"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "rag"))

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

from parser import parse_document
from chunker import chunk_text
from embedder import get_embeddings
from metadata_extractor import extract_metadata
from indexer import index_chunks

# Document metadata mapping
DOC_METADATA = {
    "mortgage_guidelines.txt": {"department": "finance", "access_level": "internal"},
    "loan_policy.txt": {"department": "finance", "access_level": "internal"},
    "fraud_prevention.txt": {"department": "legal", "access_level": "confidential"},
    "kyc_aml_policy.txt": {"department": "legal", "access_level": "confidential"},
    "credit_card_terms.txt": {"department": "finance", "access_level": "public"},
}


def ingest_all():
    docs_dir = os.path.join(os.path.dirname(__file__), "documents")
    files = glob.glob(os.path.join(docs_dir, "*.txt"))

    if not files:
        print("No documents found in test_data/documents/")
        return

    print(f"Found {len(files)} documents to ingest.\n")

    for filepath in sorted(files):
        filename = os.path.basename(filepath)
        print(f"Processing: {filename}")

        with open(filepath, "rb") as f:
            content = f.read()

        # Parse
        text = parse_document(content, filename)
        print(f"  Parsed: {len(text)} chars")

        # Chunk
        chunks = chunk_text(text, chunk_size=512, chunk_overlap=128)
        print(f"  Chunks: {len(chunks)}")

        # Embed
        texts = [c["content"] for c in chunks]
        all_embeddings = []
        batch_size = 16
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            all_embeddings.extend(get_embeddings(batch))
        print(f"  Embeddings: {len(all_embeddings)} (dim={len(all_embeddings[0])})")

        # Metadata
        extra = DOC_METADATA.get(filename, {})
        doc_metadata = extract_metadata(filename, extra)

        # Index
        result = index_chunks(chunks, all_embeddings, doc_metadata)
        print(f"  Indexed: {result['succeeded']}/{len(chunks)} (failed: {result['failed']})")
        print()

    print("All documents ingested.")


if __name__ == "__main__":
    ingest_all()
