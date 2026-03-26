"""Dataframe loading and parquet conversion for CARP studies."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

from carp.core.dependencies import import_or_raise
from carp.core.naming import parquet_stem


class FrameService:
    """Load CARP data as dataframes or parquet files."""

    def __init__(self, records: Any, participant_directory: Any) -> None:
        self._records = records
        self._participants = participant_directory

    def parquet_path(self, data_type: str, output_dir: str | Path) -> Path:
        """Return the namespace-aware parquet path for a data type."""

        return Path(output_dir) / f"{parquet_stem(data_type)}.parquet"

    def get_dataframe(self, data_type: str, parquet_dir: str | Path | None = None) -> Any:
        """Return a dataframe for one data type."""

        pandas = import_or_raise("pandas", "pandas")
        if parquet_dir:
            parquet_path = self.parquet_path(data_type, parquet_dir)
            if parquet_path.exists():
                return pandas.read_parquet(parquet_path)
        return pandas.DataFrame(list(self._records.iter_records(data_type)))

    def get_dataframe_with_participants(
        self,
        data_type: str,
        parquet_dir: str | Path | None = None,
    ) -> Any:
        """Return a dataframe enriched with participant metadata."""

        pandas = import_or_raise("pandas", "pandas")
        frame = self.get_dataframe(data_type, parquet_dir)
        if frame.empty:
            return frame
        deployment_ids = self._deployment_series(frame)
        participant_rows = deployment_ids.apply(self._participant_row)
        return pandas.concat([frame, participant_rows], axis=1)

    def convert_to_parquet(
        self,
        output_dir: str | Path,
        batch_size: int = 10_000,
    ) -> list[Path]:
        """Convert the study to namespace-aware parquet files."""

        pyarrow = import_or_raise("pyarrow", "pandas")
        parquet = import_or_raise("pyarrow.parquet", "pandas")
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        buffers: dict[str, list[dict[str, Any]]] = defaultdict(list)
        writers: dict[str, Any] = {}
        try:
            for item in self._records.iter_records():
                data_type = self._records.data_type(item)
                buffers[data_type].append(item)
                if len(buffers[data_type]) >= batch_size:
                    self._flush_buffer(pyarrow, parquet, output_path, data_type, buffers, writers)
        finally:
            for data_type, buffer in buffers.items():
                if buffer:
                    self._flush_buffer(pyarrow, parquet, output_path, data_type, buffers, writers)
            for writer in writers.values():
                writer.close()
        return sorted(self.parquet_path(data_type, output_path) for data_type in writers)

    def _participant_row(self, deployment_id: str | None) -> Any:
        """Return participant columns for one deployment identifier."""

        pandas = import_or_raise("pandas", "pandas")
        participant = self._participants.get_participant(deployment_id or "")
        if not participant:
            return pandas.Series(
                {
                    "participant_id": None,
                    "participant_email": None,
                    "participant_folder": None,
                }
            )
        return pandas.Series(
            {
                "participant_id": participant.unified_participant_id,
                "participant_email": participant.email,
                "participant_folder": participant.source_folder,
            }
        )

    def _deployment_series(self, frame: Any) -> Any:
        """Return deployment identifiers from a dataframe."""

        if "studyDeploymentId" in frame.columns:
            return frame["studyDeploymentId"]
        return frame["dataStream"].apply(lambda value: value.get("studyDeploymentId") if isinstance(value, dict) else None)

    def _flush_buffer(
        self,
        pyarrow: Any,
        parquet: Any,
        output_path: Path,
        data_type: str,
        buffers: dict[str, list[dict[str, Any]]],
        writers: dict[str, Any],
    ) -> None:
        """Flush one buffered parquet batch to disk."""

        table = pyarrow.Table.from_pylist(buffers[data_type])
        path = self.parquet_path(data_type, output_path)
        writer = writers.get(data_type)
        if writer is None:
            writers[data_type] = parquet.ParquetWriter(path, table.schema)
            writer = writers[data_type]
        elif not table.schema.equals(writer.schema):
            table = self._align_table(pyarrow, table, writer.schema)
        writer.write_table(table)
        buffers[data_type].clear()

    def _align_table(self, pyarrow: Any, table: Any, schema: Any) -> Any:
        """Align a batch to an existing parquet schema."""

        columns = []
        for field in schema:
            if field.name not in table.column_names:
                columns.append(pyarrow.nulls(len(table), type=field.type))
                continue
            column = table[field.name]
            if not column.type.equals(field.type):
                column = column.cast(field.type)
            columns.append(column)
        return pyarrow.Table.from_arrays(columns, schema=schema)
