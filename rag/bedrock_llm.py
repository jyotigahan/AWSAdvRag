"""AWS Bedrock LLM wrapper with token counting and cost tracking. Supports Claude and Nova."""
import os
import json
import logging
import boto3

metrics_logger = logging.getLogger("MetricsLLM")
metrics_logger.setLevel(logging.INFO)
if not metrics_logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(name)s | %(message)s"))
    metrics_logger.addHandler(handler)

PRICING = {
    "us.amazon.nova-micro-v1:0": {"input": 0.000035, "output": 0.00014},
    "us.amazon.nova-lite-v1:0": {"input": 0.00006, "output": 0.00024},
    "us.amazon.nova-pro-v1:0": {"input": 0.0008, "output": 0.0032},
    "anthropic.claude-3-haiku-20240307-v1:0": {"input": 0.00025, "output": 0.00125},
    "amazon.titan-embed-text-v2:0": {"input": 0.00002, "output": 0.0},
}


def _get_bedrock_client():
    return boto3.client("bedrock-runtime", region_name=os.environ.get("AWS_REGION", "us-east-1"))


def _estimate_cost(model_id: str, input_tokens: int, output_tokens: int) -> float:
    prices = PRICING.get(model_id, {"input": 0.0001, "output": 0.0004})
    return (input_tokens / 1000 * prices["input"]) + (output_tokens / 1000 * prices["output"])


def _is_nova(model_id: str) -> bool:
    return "nova" in model_id.lower()


def _build_request_body(model_id, system_prompt, user_prompt, temperature, max_tokens):
    if _is_nova(model_id):
        body = {
            "messages": [{"role": "user", "content": [{"text": user_prompt}]}],
            "inferenceConfig": {"maxTokens": max_tokens, "temperature": temperature},
        }
        if system_prompt:
            body["system"] = [{"text": system_prompt}]
        return body
    else:
        return {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": max_tokens,
            "temperature": temperature,
            "system": system_prompt,
            "messages": [{"role": "user", "content": user_prompt}],
        }


def _parse_response(model_id, result):
    if _is_nova(model_id):
        text = result["output"]["message"]["content"][0]["text"].strip()
        usage = result.get("usage", {})
        return text, usage.get("inputTokens", 0), usage.get("outputTokens", 0), result.get("stopReason", "unknown")
    else:
        text = result["content"][0]["text"].strip()
        usage = result.get("usage", {})
        return text, usage.get("input_tokens", 0), usage.get("output_tokens", 0), result.get("stop_reason", "unknown")


def chat_completion(
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.1,
    max_tokens: int = 1000,
    model_id: str | None = None,
    caller: str = "unknown",
) -> str:
    client = _get_bedrock_client()
    model_id = model_id or os.environ.get("BEDROCK_CHAT_MODEL_ID", "us.amazon.nova-micro-v1:0")

    body = _build_request_body(model_id, system_prompt, user_prompt, temperature, max_tokens)
    response = client.invoke_model(
        modelId=model_id, contentType="application/json", accept="application/json", body=json.dumps(body),
    )
    result = json.loads(response["body"].read())
    text, input_tokens, output_tokens, stop_reason = _parse_response(model_id, result)
    total_tokens = input_tokens + output_tokens
    cost = _estimate_cost(model_id, input_tokens, output_tokens)

    metrics_logger.info(json.dumps({
        "type": "chat", "caller": caller, "model_id": model_id,
        "input_tokens": input_tokens, "output_tokens": output_tokens,
        "total_tokens": total_tokens, "estimated_cost_usd": round(cost, 6), "stop_reason": stop_reason,
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
    client = _get_bedrock_client()
    model_id = model_id or os.environ.get("BEDROCK_CHAT_MODEL_ID", "us.amazon.nova-micro-v1:0")

    body = _build_request_body(model_id, system_prompt, user_prompt, temperature, max_tokens)
    response = client.invoke_model(
        modelId=model_id, contentType="application/json", accept="application/json", body=json.dumps(body),
    )
    result = json.loads(response["body"].read())
    text, input_tokens, output_tokens, stop_reason = _parse_response(model_id, result)
    total_tokens = input_tokens + output_tokens
    cost = _estimate_cost(model_id, input_tokens, output_tokens)

    metrics_logger.info(json.dumps({
        "type": "chat", "caller": caller, "model_id": model_id,
        "input_tokens": input_tokens, "output_tokens": output_tokens,
        "total_tokens": total_tokens, "estimated_cost_usd": round(cost, 6), "stop_reason": stop_reason,
    }))
    return {
        "text": text, "input_tokens": input_tokens, "output_tokens": output_tokens,
        "total_tokens": total_tokens, "cost_usd": round(cost, 6), "model_id": model_id,
    }
