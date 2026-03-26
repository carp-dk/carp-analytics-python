"""Shared CLI helpers and presenters."""

from __future__ import annotations

from typing import Any

from rich.console import Console
from rich.table import Table

from carp import __version__
from carp.study import CarpStudy

console = Console()


def build_study(files: Any, load_participants: bool = True) -> CarpStudy:
    """Construct a study from CLI arguments."""

    return CarpStudy(files, load_participants=load_participants)


def print_version() -> int:
    """Print the package version and return a success status."""

    console.print(f"carp-analytics-python version {__version__}")
    return 0


def print_schema(schema_map: dict[str, list[str]]) -> None:
    """Render a schema table."""

    table = Table(title="Inferred Schema")
    table.add_column("Data Type", style="cyan")
    table.add_column("Fields", style="magenta")
    for data_type, fields in schema_map.items():
        table.add_row(data_type, ", ".join(fields))
    console.print(table)


def print_participants(rows: list[dict[str, str]]) -> None:
    """Render participant summary rows."""

    table = Table(title="Participants Summary")
    for column in ("unified_id", "deployments", "folders", "emails", "ssns", "names"):
        table.add_column(column.replace("_", " ").title())
    for row in rows:
        table.add_row(*(row[key] for key in row))
    console.print(table)
