"""Local LLM wrapper using Ollama — drop-in replacement for bedrock_llm.py."""
import json
import logging
import urllib.request
import urllib.error

OLLAMA_BASE = "http://localhost:11434"

metrics_logger = logging.getLogger("MetricsLLM")
metrics_logger.setLevel(logging.INFO)
if not metrics_logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(name)s | %(message)s"))
    metrics_logger.addHandler(handler)


def _ollama_chat(model: str, messages: list[dict], temperature: float = 0.1, max_tokens: int = 1000) -> dict:
    """Call Ollama /api/chat endpoint."""
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
        },
    }
    req = urllib.request.Request(
        f"{OLLAMA_BASE}/api/chat",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            return json.loads(resp.read())
    except urllib.error.URLError as e:
        raise ConnectionError(
            f"Cannot connect to Ollama at {OLLAMA_BASE}. "
            "Make sure Ollama is running: 'ollama serve'"
        ) from e


def chat_completion(
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.1,
    max_tokens: int = 1000,
    model_id: str | None = None,
    caller: str = "unknown",
) -> str:
    model = model_id or "llama3.2"
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})

    result = _ollama_chat(model, messages, temperature, max_tokens)
    text = result.get("message", {}).get("content", "").strip()

    eval_count = result.get("eval_count", 0)
    prompt_eval_count = result.get("prompt_eval_count", 0)

    metrics_logger.info(json.dumps({
        "type": "chat", "caller": caller, "model_id": model,
        "input_tokens": prompt_eval_count, "output_tokens": eval_count,
        "total_tokens": prompt_eval_count + eval_count,
        "estimated_cost_usd": 0.0, "stop_reason": "stop",
    }))
    return text


def chat_completion_with_usage(
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.1,
    max_tokens: int = 1000,
    model_id: str | None = None,
    caller: str = "unknown",
) -> dict:
    model = model_id or "llama3.2"
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})

    result = _ollama_chat(model, messages, temperature, max_tokens)
    text = result.get("message", {}).get("content", "").strip()

    eval_count = result.get("eval_count", 0)
    prompt_eval_count = result.get("prompt_eval_count", 0)
    total = prompt_eval_count + eval_count

    metrics_logger.info(json.dumps({
        "type": "chat", "caller": caller, "model_id": model,
        "input_tokens": prompt_eval_count, "output_tokens": eval_count,
        "total_tokens": total, "estimated_cost_usd": 0.0, "stop_reason": "stop",
    }))
    return {
        "text": text, "input_tokens": prompt_eval_count, "output_tokens": eval_count,
        "total_tokens": total, "cost_usd": 0.0, "model_id": model,
    }
