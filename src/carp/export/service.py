"""JSON export and grouping services."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

from carp.constants import UNKNOWN_VALUE
from carp.core.fields import get_nested_value
from carp.core.files import JsonArrayWriter
from carp.core.naming import sanitize_filename


class ExportService:
    """Export CARP records to JSON arrays."""

    def __init__(self, records: Any) -> None:
        self._records = records

    def export_json(self, output_path: str | Path, data_type: str | None = None) -> Path:
        """Write matching records to a JSON array file."""

        writer = JsonArrayWriter(Path(output_path))
        try:
            for item in self._records.iter_records(data_type):
                writer.write(item)
        finally:
            writer.close()
        return Path(output_path)

    def group_by_field(self, field_path: str, output_dir: str | Path) -> list[Path]:
        """Group records by a nested field path."""

        def key_factory(item: dict[str, Any]) -> str:
            value = get_nested_value(item, field_path, UNKNOWN_VALUE)
            return sanitize_filename(str(value), allowed="-_.@")

        return self._write_groups(Path(output_dir), self._records.iter_records(), key_factory)

    def group_by_participant(self, output_dir: str | Path, data_type: str | None = None) -> list[Path]:
        """Group records by unified participant identifier."""

        def key_factory(item: dict[str, Any]) -> str:
            participant = item.get("_participant", {})
            return sanitize_filename(
                str(participant.get("unified_participant_id", UNKNOWN_VALUE)),
                allowed="-_.@",
            )

        return self._write_groups(Path(output_dir), self._records.iter_with_participants(data_type), key_factory)

    def group_by_identity(
        self,
        field_name: str,
        output_dir: str | Path,
        data_type: str | None = None,
    ) -> list[Path]:
        """Group records by a participant identity field."""

        def key_factory(item: dict[str, Any]) -> str:
            participant = item.get("_participant", {})
            value = participant.get(field_name) or UNKNOWN_VALUE
            return sanitize_filename(str(value), allowed="-_.@")

        return self._write_groups(
            Path(output_dir),
            self._records.iter_with_participants(data_type),
            key_factory,
        )

    def _write_groups(
        self,
        output_dir: Path,
        items: Iterable[dict[str, Any]],
        key_factory: Callable[[dict[str, Any]], str],
    ) -> list[Path]:
        """Write grouped JSON files and return created paths."""

        writers: dict[str, JsonArrayWriter] = {}
        output_dir.mkdir(parents=True, exist_ok=True)
        try:
            for item in items:
                key = key_factory(item)
                writer = writers.get(key)
                if writer is None:
                    writer = JsonArrayWriter(output_dir / f"{key}.json")
                    writers[key] = writer
                writer.write(item)
        finally:
            for writer in writers.values():
                writer.close()
        return sorted(writer.output_path for writer in writers.values())
