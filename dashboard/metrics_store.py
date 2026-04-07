"""Centralized metrics store — writes all metrics to a local JSON file for dashboard consumption."""
import os
import json
import threading
from datetime import datetime, timezone

METRICS_FILE = os.path.join(os.path.dirname(__file__), "metrics_data.json")
_lock = threading.Lock()


def _load() -> dict:
    if os.path.exists(METRICS_FILE):
        with open(METRICS_FILE) as f:
            return json.load(f)
    return {"ingestion": [], "retrieval": [], "ragas": [], "llm": []}


def _save(data: dict):
    with open(METRICS_FILE, "w") as f:
        json.dump(data, f, indent=2)


def record(category: str, entry: dict):
    """Append a metrics entry to the store. Thread-safe."""
    entry["_recorded_at"] = datetime.now(timezone.utc).isoformat()
    with _lock:
        data = _load()
        if category not in data:
            data[category] = []
        data[category].append(entry)
        _save(data)


def get_all() -> dict:
    """Return all stored metrics."""
    return _load()


def clear():
    """Reset all metrics."""
    _save({"ingestion": [], "retrieval": [], "ragas": [], "llm": []})
