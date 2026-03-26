"""CARP record iteration, filtering, and inspection."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import Any

from carp.core.fields import collect_field_paths, deployment_id_from_record, full_data_type
from carp.core.files import iter_json_array


class RecordService:
    """Stream and filter CARP records."""

    def __init__(self, file_paths: tuple[Any, ...], participant_directory: Any) -> None:
        self._file_paths = file_paths
        self._participants = participant_directory

    def iter_records(
        self,
        data_type: str | None = None,
        deployment_ids: Iterable[str] | None = None,
    ) -> Iterator[dict[str, Any]]:
        """Yield records matching optional data-type and deployment filters."""

        allowed_ids = set(deployment_ids or [])
        for file_path in self._file_paths:
            for item in iter_json_array(file_path):
                if allowed_ids and deployment_id_from_record(item) not in allowed_ids:
                    continue
                if data_type and full_data_type(item) != data_type:
                    continue
                yield item

    def iter_with_participants(self, data_type: str | None = None) -> Iterator[dict[str, Any]]:
        """Yield records enriched with participant metadata."""

        for item in self.iter_records(data_type):
            participant = self._participants.get_participant(deployment_id_from_record(item) or "")
            if not participant:
                yield item
                continue
            enriched = dict(item)
            enriched["_participant"] = participant.to_dict()
            yield enriched

    def count(
        self,
        data_type: str | None = None,
        deployment_ids: Iterable[str] | None = None,
    ) -> int:
        """Return the number of matching records."""

        return sum(1 for _ in self.iter_records(data_type, deployment_ids))

    def list_fields(self, sample_size: int = 100) -> list[str]:
        """Return field paths sampled from the first records."""

        fields: set[str] = set()
        for index, item in enumerate(self.iter_records()):
            if index >= sample_size:
                break
            fields.update(self.collect_fields(item))
        return sorted(fields)

    def data_types(self) -> list[str]:
        """Return all observed record data types."""

        return sorted({self.data_type(item) for item in self.iter_records()})

    @staticmethod
    def collect_fields(item: dict[str, Any]) -> set[str]:
        """Collect field paths for one record."""

        return collect_field_paths(item)

    @staticmethod
    def data_type(item: dict[str, Any]) -> str:
        """Return the fully qualified data type for one record."""

        return full_data_type(item)
