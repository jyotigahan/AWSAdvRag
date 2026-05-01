"""Local embeddings using Ollama — no torch/sentence-transformers needed."""
import json
import logging
import urllib.request
import urllib.error

OLLAMA_BASE = "http://localhost:11434"
EMBED_MODEL = "nomic-embed-text"

metrics_logger = logging.getLogger("MetricsLLM")
metrics_logger.setLevel(logging.INFO)
if not metrics_logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(name)s | %(message)s"))
    metrics_logger.addHandler(handler)


def get_embeddings(texts: list[str], caller: str = "unknown") -> list[list[float]]:
    """Get embeddings using Ollama's embedding API."""
    embeddings = []
    for text in texts:
        payload = {"model": EMBED_MODEL, "input": text}
        req = urllib.request.Request(
            f"{OLLAMA_BASE}/api/embed",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                result = json.loads(resp.read())
                embeddings.append(result["embeddings"][0])
        except urllib.error.URLError as e:
            raise ConnectionError(
                f"Cannot connect to Ollama at {OLLAMA_BASE}. "
                "Make sure Ollama is running: 'ollama serve'"
            ) from e

    dim = len(embeddings[0]) if embeddings else 0
    metrics_logger.info(json.dumps({
        "type": "embedding",
        "caller": caller,
        "model_id": EMBED_MODEL,
        "texts_count": len(texts),
        "total_input_tokens": 0,
        "embedding_dimension": dim,
        "estimated_cost_usd": 0.0,
    }))
    return embeddings
