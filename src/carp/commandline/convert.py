"""CLI command for parquet conversion."""

from __future__ import annotations

from typing import Any

from .common import build_study, console


def register(subparsers: Any) -> None:
    """Register the `convert` subcommand."""

    parser = subparsers.add_parser("convert", help="Convert JSON to Parquet")
    parser.add_argument("files", nargs="+", help="JSON data files to convert")
    parser.add_argument("-o", "--output", default="output_parquet", help="Output directory")
    parser.add_argument("--batch-size", type=int, default=10_000, help="Batch size")
    parser.set_defaults(handler=run)


def run(args: Any) -> int:
    """Execute the `convert` subcommand."""

    files = build_study(args.files, load_participants=False).frames.convert_to_parquet(
        args.output,
        batch_size=args.batch_size,
    )
    console.print(f"[bold green]Created {len(files)} parquet files.[/bold green]")
    return 0
