"""Schema discovery for CARP studies."""

from __future__ import annotations

from collections import defaultdict
from typing import Any


class SchemaService:
    """Infer lightweight measurement schemas grouped by data type."""

    def __init__(self, records: Any) -> None:
        self._records = records
        self._cache: dict[str, list[str]] = {}

    def scan(self) -> dict[str, list[str]]:
        """Return inferred measurement keys grouped by data type."""

        schemas: dict[str, set[str]] = defaultdict(set)
        for item in self._records.iter_records():
            measurement = item.get("measurement", {}).get("data", {})
            for key in measurement.keys():
                schemas[self._records.data_type(item)].add(key)
        self._cache = {key: sorted(values) for key, values in sorted(schemas.items())}
        return self._cache

    def cached(self) -> dict[str, list[str]]:
        """Return the cached schema, scanning the study if needed."""

        return self._cache or self.scan()
