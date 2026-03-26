"""CLI commands for exporting study data."""

from __future__ import annotations

from typing import Any

from .common import build_study, console


def register_export(subparsers: Any) -> None:
    """Register the `export` subcommand."""

    parser = subparsers.add_parser("export", help="Export data to JSON")
    parser.add_argument("files", nargs="+", help="JSON data files to process")
    parser.add_argument("-o", "--output", required=True, help="Output JSON file path")
    parser.add_argument("-t", "--type", dest="data_type", help="Filter by data type")
    parser.set_defaults(handler=run_export)


def register_group(subparsers: Any) -> None:
    """Register the `group` subcommand."""

    parser = subparsers.add_parser("group", help="Group data by field")
    parser.add_argument("files", nargs="+", help="JSON data files to process")
    parser.add_argument(
        "-f",
        "--field",
        default="dataStream.dataType.name",
        help="Field path to group by",
    )
    parser.add_argument("-o", "--output", default="output_grouped", help="Output directory")
    parser.set_defaults(handler=run_group)


def run_export(args: Any) -> int:
    """Execute the `export` subcommand."""

    output = build_study(args.files, load_participants=False).export.export_json(
        args.output,
        args.data_type,
    )
    console.print(f"[bold green]Exported data to {output}[/bold green]")
    return 0


def run_group(args: Any) -> int:
    """Execute the `group` subcommand."""

    files = build_study(args.files, load_participants=False).export.group_by_field(
        args.field,
        args.output,
    )
    console.print(f"[bold green]Created {len(files)} grouped files.[/bold green]")
    return 0
