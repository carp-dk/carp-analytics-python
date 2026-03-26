"""Filesystem helpers for CARP Analytics."""

from __future__ import annotations

import json
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import ijson


def resolve_paths(file_paths: str | Path | Sequence[str | Path]) -> tuple[Path, ...]:
    """Validate and normalize data-stream paths."""

    raw_paths = [file_paths] if isinstance(file_paths, (str, Path)) else list(file_paths)
    resolved = tuple(Path(path) for path in raw_paths)
    for path in resolved:
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
    return resolved


def iter_json_array(file_path: Path) -> Iterator[dict[str, Any]]:
    """Stream JSON array items from disk using `ijson`."""

    with file_path.open("rb") as handle:
        yield from ijson.items(handle, "item", use_float=True)


class JsonArrayWriter:
    """Incrementally write JSON arrays without buffering the full payload."""

    def __init__(self, output_path: Path):
        self.output_path = output_path
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.output_path.open("w", encoding="utf-8")
        self._first_item = True
        self._handle.write("[")

    def write(self, item: dict[str, Any]) -> None:
        """Append one JSON object to the array."""

        if not self._first_item:
            self._handle.write(",")
        json.dump(item, self._handle)
        self._first_item = False

    def close(self) -> None:
        """Finalize and close the output file."""

        self._handle.write("]")
        self._handle.close()
