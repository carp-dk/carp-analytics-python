"""CLI command for record counting."""

from __future__ import annotations

from typing import Any

from .common import build_study, console


def register(subparsers: Any) -> None:
    """Register the `count` subcommand."""

    parser = subparsers.add_parser("count", help="Count items in data files")
    parser.add_argument("files", nargs="+", help="JSON data files to count")
    parser.set_defaults(handler=run)


def run(args: Any) -> int:
    """Execute the `count` subcommand."""

    count = build_study(args.files, load_participants=False).records.count()
    console.print(f"[bold green]Total items: {count:,}[/bold green]")
    return 0
