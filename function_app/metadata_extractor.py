"""Extract metadata from uploaded documents."""
from datetime import datetime, timezone


def extract_metadata(filename: str, extra: dict | None = None) -> dict:
    """Build metadata dict for a document."""
    meta = {
        "source": filename,
        "department": "general",
        "access_level": "public",
        "author": "unknown",
        "date_uploaded": datetime.now(timezone.utc).isoformat(),
    }
    if extra:
        meta.update(extra)
    return meta
