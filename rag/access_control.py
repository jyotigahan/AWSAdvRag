"""Source-level access control — builds OpenSearch filters per user."""

USER_PERMISSIONS = {
    "admin": {"departments": ["*"], "access_levels": ["public", "internal", "confidential"]},
    "manager": {"departments": ["*"], "access_levels": ["public", "internal"]},
    "employee": {"departments": [], "access_levels": ["public"]},
}


def get_access_filter(user_role: str, user_department: str | None = None) -> list[dict] | None:
    """Build OpenSearch filter clauses based on user role and department."""
    perms = USER_PERMISSIONS.get(user_role, USER_PERMISSIONS["employee"])
    filters = []

    # Access level filter
    levels = perms["access_levels"]
    if levels:
        filters.append({"terms": {"access_level": levels}})

    # Department filter
    depts = perms["departments"]
    if "*" not in depts and user_department:
        filters.append({
            "bool": {
                "should": [
                    {"term": {"department": user_department}},
                    {"term": {"department": "general"}},
                ],
                "minimum_should_match": 1,
            }
        })

    return filters if filters else None
