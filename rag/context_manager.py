"""Context management: windowing + token budget."""
import tiktoken


def build_context(
    chunks: list[dict],
    max_tokens: int = 4000,
    window_size: int = 1,
    encoding_name: str = "cl100k_base",
) -> dict:
    """Build context string from chunks with windowing and token budget.

    Returns:
        dict with 'text', 'token_count', 'chunks_used', 'truncated'.
    """
    enc = tiktoken.get_encoding(encoding_name)
    seen_ids = set()

    # Expand each chunk with its neighbors (windowing)
    for chunk in chunks:
        source = chunk.get("source", "")
        idx = chunk.get("chunk_index", 0)
        total = chunk.get("total_chunks", 1)

        for offset in range(-window_size, window_size + 1):
            neighbor_idx = idx + offset
            if 0 <= neighbor_idx < total:
                key = f"{source}_{neighbor_idx}"
                seen_ids.add(key)

    # Build context within token budget
    context_parts = []
    token_count = 0
    chunks_used = 0
    truncated = False

    for chunk in chunks:
        chunk_text = chunk["content"]
        chunk_tokens = len(enc.encode(chunk_text))

        if token_count + chunk_tokens > max_tokens:
            truncated = True
            remaining = max_tokens - token_count
            if remaining > 50:
                tokens = enc.encode(chunk_text)[:remaining]
                context_parts.append(enc.decode(tokens))
                token_count += remaining
                chunks_used += 1
            break

        source_header = f"[Source: {chunk.get('source', 'unknown')} | Chunk {chunk.get('chunk_index', 0)}]"
        context_parts.append(f"{source_header}\n{chunk_text}")
        token_count += chunk_tokens
        chunks_used += 1

    text = "\n\n---\n\n".join(context_parts)

    return {
        "text": text,
        "token_count": token_count,
        "chunks_used": chunks_used,
        "truncated": truncated,
    }
