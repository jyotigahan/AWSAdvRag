"""Build Amazon OpenSearch filter queries from metadata criteria."""


def build_filter(criteria: dict) -> list[dict] | None:
    """Build OpenSearch filter clauses from a dict of field -> value pairs.

    Supports:
        - Exact match: {"department": "finance"}
        - List match (OR): {"department": ["finance", "hr"]}
        - Date range: {"date_uploaded": {"gte": "2024-01-01", "lte": "2024-12-31"}}
    """
    filters = []

    for field, value in criteria.items():
        if isinstance(value, list):
            filters.append({"terms": {field: value}})
        elif isinstance(value, dict):
            range_clause = {}
            if "gte" in value:
                range_clause["gte"] = value["gte"]
            if "lte" in value:
                range_clause["lte"] = value["lte"]
            if range_clause:
                filters.append({"range": {field: range_clause}})
        else:
            filters.append({"term": {field: value}})

    return filters if filters else None


def combine_filters(*filter_lists) -> list[dict] | None:
    """Combine multiple filter lists into one."""
    combined = []
    for f in filter_lists:
        if f:
            if isinstance(f, list):
                combined.extend(f)
            else:
                combined.append(f)
    return combined if combined else None
