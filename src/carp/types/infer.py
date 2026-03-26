"""Schema inference helpers for generated type definitions."""

from __future__ import annotations

import json
from typing import Any


def _maybe_json_string(value: object) -> Any | None:
    """Parse JSON-like strings when possible."""

    if not isinstance(value, str):
        return None
    stripped = value.strip()
    if not stripped or stripped[0] not in "[{" or stripped[-1] not in "]}":
        return None
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, (dict, list)) else None


def merge_schema(schema: dict[str, Any], value: Any) -> None:
    """Merge a Python value into an inferred schema."""

    if value is None:
        schema["nullable"] = True
        return
    parsed = _maybe_json_string(value)
    if parsed is not None:
        schema["is_json_string"] = True
        merge_schema(schema, parsed)
        return
    if isinstance(value, dict):
        schema["type"] = "object"
        fields = schema.setdefault("fields", {})
        for key, child in value.items():
            merge_schema(fields.setdefault(key, {}), child)
        return
    if isinstance(value, list):
        schema["type"] = "list"
        item_type = schema.setdefault("item_type", {})
        for child in value:
            merge_schema(item_type, child)
        return
    python_type = type(value).__name__
    if schema.get("type") == "primitive" and schema.get("python_type") != python_type:
        pair = {schema.get("python_type"), python_type}
        schema["python_type"] = "float" if pair == {"int", "float"} else "Any"
        return
    schema["type"] = "primitive"
    schema["python_type"] = python_type


def infer_schema(records: Any, sample_size: int) -> dict[str, Any]:
    """Infer a schema from sampled study records."""

    root = {"type": "object", "fields": {}}
    for index, item in enumerate(records):
        if index >= sample_size:
            break
        merge_schema(root, item)
    return root
