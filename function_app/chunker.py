"""Chunker with sliding window for context overlap."""
import tiktoken


def chunk_text(
    text: str,
    chunk_size: int = 512,
    chunk_overlap: int = 128,
    encoding_name: str = "cl100k_base",
) -> list[dict]:
    """Split text into overlapping chunks with token-based sizing."""
    enc = tiktoken.get_encoding(encoding_name)
    tokens = enc.encode(text)
    chunks = []
    start = 0

    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunk_tokens = tokens[start:end]
        chunk_text = enc.decode(chunk_tokens)
        chunks.append({
            "content": chunk_text,
            "chunk_index": len(chunks),
            "token_count": len(chunk_tokens),
        })
        if end >= len(tokens):
            break
        start += chunk_size - chunk_overlap

    for c in chunks:
        c["total_chunks"] = len(chunks)

    return chunks
