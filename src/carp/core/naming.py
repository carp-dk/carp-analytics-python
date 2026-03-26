"""File and identifier naming helpers."""

from __future__ import annotations

from carp.constants import UNKNOWN_VALUE


def sanitize_filename(value: str, allowed: str = "-_") -> str:
    """Return a filesystem-safe representation of a string."""

    safe = "".join(char for char in value if char.isalnum() or char in allowed).strip()
    return safe or UNKNOWN_VALUE


def parquet_stem(data_type: str) -> str:
    """Return a namespace-aware parquet stem for a data type."""

    namespace, _, name = data_type.rpartition(".")
    stem = f"{namespace}__{name}" if namespace else data_type
    return sanitize_filename(stem, allowed="-_.")
