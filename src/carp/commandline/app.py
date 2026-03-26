"""Argument parsing and dispatch for the CARP CLI."""

from __future__ import annotations

import argparse

from .common import console, print_version
from .convert import register as register_convert
from .count import register as register_count
from .export import register_export, register_group
from .participants import register as register_participants
from .schema import register as register_schema


def _build_parser() -> argparse.ArgumentParser:
    """Construct the top-level CLI parser."""

    parser = argparse.ArgumentParser(
        prog="carp",
        description="CARP Analytics - Process and analyze data from CARP research studies",
    )
    parser.add_argument("--version", action="store_true", help="Show version and exit")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    for register in (
        register_schema,
        register_convert,
        register_count,
        register_participants,
        register_export,
        register_group,
    ):
        register(subparsers)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the CARP command-line interface."""

    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.version:
        return print_version()
    if not args.command:
        parser.print_help()
        return 0
    try:
        return int(args.handler(args))
    except FileNotFoundError as exc:
        console.print(f"[bold red]Error: {exc}[/bold red]")
        return 1
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted.[/yellow]")
        return 130
    except Exception as exc:
        console.print(f"[bold red]Error: {exc}[/bold red]")
        return 1
