"""Lambda handler that reads metrics from CloudWatch Logs and returns them for the dashboard."""
import os
import json
import time
import boto3

logs_client = boto3.client("logs", region_name=os.environ.get("AWS_REGION_NAME", "us-east-1"))

LOG_GROUPS = {
    "ingestion": "/aws/lambda/{ingest_fn}",
    "retrieval": "/aws/lambda/{api_fn}",
}

METRIC_PREFIXES = {
    "ingestion": "MetricsIngestion",
    "retrieval": "MetricsRetrieval",
    "ragas": "MetricsRAGAS",
    "llm": "MetricsLLM",
}


def metrics_handler(event, context):
    """Fetch structured metrics from CloudWatch Logs for the dashboard."""
    try:
        ingest_fn = os.environ.get("INGEST_FUNCTION_NAME", "")
        api_fn = os.environ.get("API_FUNCTION_NAME", "")

        log_groups = [f"/aws/lambda/{ingest_fn}", f"/aws/lambda/{api_fn}"]
        # Look back 24 hours
        start_time = int((time.time() - 86400) * 1000)
        end_time = int(time.time() * 1000)

        result = {"ingestion": [], "retrieval": [], "ragas": [], "llm": []}

        for log_group in log_groups:
            for category, prefix in METRIC_PREFIXES.items():
                try:
                    resp = logs_client.filter_log_events(
                        logGroupName=log_group,
                        filterPattern=prefix,
                        startTime=start_time,
                        endTime=end_time,
                        limit=50,
                    )
                    for event_item in resp.get("events", []):
                        msg = event_item.get("message", "")
                        if "|" in msg:
                            json_str = msg.split("|", 1)[1].strip()
                            try:
                                entry = json.loads(json_str)
                                result[category].append(entry)
                            except json.JSONDecodeError:
                                pass
                except Exception:
                    pass

        return {
            "statusCode": 200,
            "headers": {
                "Content-Type": "application/json",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "GET,OPTIONS",
                "Access-Control-Allow-Headers": "Content-Type",
            },
            "body": json.dumps(result),
        }
    except Exception as e:
        return {
            "statusCode": 500,
            "headers": {"Content-Type": "application/json", "Access-Control-Allow-Origin": "*"},
            "body": json.dumps({"error": str(e)}),
        }
