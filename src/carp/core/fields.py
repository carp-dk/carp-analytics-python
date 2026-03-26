"""Helpers for nested CARP record structures."""

from __future__ import annotations

from typing import Any

from carp.constants import UNKNOWN_VALUE


def get_nested_value(value: Any, path: str, default: Any = None) -> Any:
    """Resolve a dot-separated path from nested dictionaries."""

    current = value
    for part in path.split("."):
        if not isinstance(current, dict):
            return default
        current = current.get(part)
        if current is None:
            return default
    return current


def collect_field_paths(value: Any, prefix: str = "") -> set[str]:
    """Collect dot-separated field paths from nested dictionaries."""

    paths: set[str] = set()
    if isinstance(value, dict):
        for key, child in value.items():
            path = f"{prefix}.{key}" if prefix else key
            paths.add(path)
            paths.update(collect_field_paths(child, path))
    elif isinstance(value, list):
        if prefix:
            paths.add(f"{prefix}[]")
        if value:
            paths.update(collect_field_paths(value[0], f"{prefix}[]"))
    return paths


def full_data_type(item: dict[str, Any]) -> str:
    """Return the fully qualified data type for a CARP record."""

    data_type = get_nested_value(item, "dataStream.dataType", {})
    namespace = data_type.get("namespace", UNKNOWN_VALUE)
    name = data_type.get("name", UNKNOWN_VALUE)
    return f"{namespace}.{name}"


def deployment_id_from_record(item: dict[str, Any]) -> str | None:
    """Return the deployment identifier for a CARP record."""

    top_level = item.get("studyDeploymentId")
    if isinstance(top_level, str):
        return top_level
    nested = get_nested_value(item, "dataStream.studyDeploymentId")
    return nested if isinstance(nested, str) else None
