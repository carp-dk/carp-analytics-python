"""CLI command for schema discovery."""

from __future__ import annotations

from typing import Any

from .common import build_study, print_schema


def register(subparsers: Any) -> None:
    """Register the `schema` subcommand."""

    parser = subparsers.add_parser("schema", help="Scan and print data schema")
    parser.add_argument("files", nargs="+", help="JSON data files to process")
    parser.set_defaults(handler=run)


def run(args: Any) -> int:
    """Execute the `schema` subcommand."""

    print_schema(build_study(args.files, load_participants=False).schema.scan())
    return 0
