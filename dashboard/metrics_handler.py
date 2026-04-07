"""Custom logging handler that writes metrics to the dashboard store."""
import json
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from metrics_store import record

CATEGORY_MAP = {
    "MetricsIngestion": "ingestion",
    "MetricsRetrieval": "retrieval",
    "MetricsRAGAS": "ragas",
    "MetricsLLM": "llm",
}


class DashboardMetricsHandler(logging.Handler):
    """Logging handler that captures structured metrics and writes to the dashboard store."""

    def emit(self, log_record):
        try:
            category = CATEGORY_MAP.get(log_record.name)
            if not category:
                return
            # Parse the JSON message
            msg = log_record.getMessage()
            # Strip logger prefix if present (e.g. "MetricsLLM | {...}")
            if "|" in msg:
                msg = msg.split("|", 1)[1].strip()
            entry = json.loads(msg)
            record(category, entry)
        except (json.JSONDecodeError, Exception):
            pass  # Skip non-JSON messages


def install():
    """Install the dashboard handler on all metrics loggers."""
    handler = DashboardMetricsHandler()
    for logger_name in CATEGORY_MAP:
        logger = logging.getLogger(logger_name)
        # Avoid duplicate handlers
        if not any(isinstance(h, DashboardMetricsHandler) for h in logger.handlers):
            logger.addHandler(handler)
    return handler
