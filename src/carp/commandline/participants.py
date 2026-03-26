"""CLI command for participant summaries."""

from __future__ import annotations

from typing import Any

from .common import build_study, print_participants


def register(subparsers: Any) -> None:
    """Register the `participants` subcommand."""

    parser = subparsers.add_parser("participants", help="List participants from data files")
    parser.add_argument("files", nargs="+", help="JSON data files to process")
    parser.set_defaults(handler=run)


def run(args: Any) -> int:
    """Execute the `participants` subcommand."""

    print_participants(build_study(args.files).participants.summary_rows())
    return 0
