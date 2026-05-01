"""Ingest test documents directly into local ChromaDB — no server needed."""
import os
import sys
import glob

LOCAL_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(LOCAL_DIR)
sys.path.insert(0, LOCAL_DIR)
sys.path.insert(0, os.path.join(PROJECT_DIR, "function_app"))

import local_embedder as embedder_mod
import chroma_store
sys.modules["embedder"] = embedder_mod

from parser import parse_document
from chunker import chunk_text
from datetime import datetime, timezone


def ingest_file(filepath: str, department: str = "general", access_level: str = "public"):
    filename = os.path.basename(filepath)
    print(f"  Processing: {filename}")

    with open(filepath, "rb") as f:
        content = f.read()

    text = parse_document(content, filename)
    if not text.strip():
        print(f"    ⚠️  Empty after parsing, skipping")
        return

    chunks = chunk_text(text, chunk_size=512, chunk_overlap=128)
    texts = [c["content"] for c in chunks]
    embeddings = embedder_mod.get_embeddings(texts, caller="ingestion")

    metadata = {
        "source": filename,
        "department": department,
        "access_level": access_level,
        "author": "test",
        "date_uploaded": datetime.now(timezone.utc).isoformat(),
    }

    result = chroma_store.index_chunks(chunks, embeddings, metadata)
    print(f"    ✅ {result['succeeded']} chunks indexed")


def main():
    docs_dir = os.path.join(PROJECT_DIR, "test_data", "documents")
    if not os.path.isdir(docs_dir):
        print(f"No test documents found at {docs_dir}")
        return

    files = glob.glob(os.path.join(docs_dir, "*"))
    if not files:
        print("No files found in test_data/documents/")
        return

    print(f"Ingesting {len(files)} test documents into ChromaDB...\n")
    for filepath in sorted(files):
        if os.path.isfile(filepath):
            ingest_file(filepath, department="finance", access_level="public")

    print(f"\nDone! ChromaDB data stored in: {chroma_store.CHROMA_DIR}")


if __name__ == "__main__":
    main()
