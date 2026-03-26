"""Type-definition generation services."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .infer import infer_schema
from .render import render_types


class TypeDefinitionService:
    """Generate typed Python models from sampled CARP records."""

    def __init__(self, records: Any) -> None:
        self._records = records

    def generate(
        self,
        output_file: str | Path = "generated_types.py",
        sample_size: int = 1_000,
    ) -> Path:
        """Generate a Python module containing inferred dataclasses."""

        schema = infer_schema(self._records.iter_records(), sample_size)
        output_path = Path(output_file)
        output_path.write_text(render_types(schema), encoding="utf-8")
        return output_path
