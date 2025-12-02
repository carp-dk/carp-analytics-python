"""
Command-line interface for CARP Analytics Python.
"""

import argparse
import sys

from rich.console import Console

console = Console()


def main() -> int:
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(
        prog="carp",
        description="CARP Analytics - Process and analyze data from CARP research studies",
    )
    parser.add_argument(
        "--version",
        action="store_true",
        help="Show version and exit",
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Schema command
    schema_parser = subparsers.add_parser("schema", help="Scan and print data schema")
    schema_parser.add_argument("files", nargs="+", help="JSON data files to process")
    
    # Convert command
    convert_parser = subparsers.add_parser("convert", help="Convert JSON to Parquet")
    convert_parser.add_argument("files", nargs="+", help="JSON data files to convert")
    convert_parser.add_argument(
        "-o", "--output",
        default="output_parquet",
        help="Output directory for Parquet files (default: output_parquet)",
    )
    convert_parser.add_argument(
        "--batch-size",
        type=int,
        default=10000,
        help="Batch size for conversion (default: 10000)",
    )
    
    # Count command
    count_parser = subparsers.add_parser("count", help="Count items in data files")
    count_parser.add_argument("files", nargs="+", help="JSON data files to count")
    
    # Participants command
    participants_parser = subparsers.add_parser(
        "participants",
        help="List participants from data files",
    )
    participants_parser.add_argument("files", nargs="+", help="JSON data files to process")
    
    # Export command
    export_parser = subparsers.add_parser("export", help="Export data to JSON")
    export_parser.add_argument("files", nargs="+", help="JSON data files to process")
    export_parser.add_argument(
        "-o", "--output",
        required=True,
        help="Output JSON file path",
    )
    export_parser.add_argument(
        "-t", "--type",
        dest="data_type",
        help="Filter by data type (e.g., dk.cachet.carp.stepcount)",
    )
    
    # Group command
    group_parser = subparsers.add_parser("group", help="Group data by field")
    group_parser.add_argument("files", nargs="+", help="JSON data files to process")
    group_parser.add_argument(
        "-f", "--field",
        default="dataStream.dataType.name",
        help="Field path to group by (default: dataStream.dataType.name)",
    )
    group_parser.add_argument(
        "-o", "--output",
        default="output_grouped",
        help="Output directory (default: output_grouped)",
    )
    
    args = parser.parse_args()
    
    if args.version:
        from carp import __version__
        console.print(f"carp-analytics-python version {__version__}")
        return 0
    
    if not args.command:
        parser.print_help()
        return 0
    
    # Import here to avoid slow startup for --help
    from carp import CarpDataStream
    
    try:
        if args.command == "schema":
            sd = CarpDataStream(args.files, load_participants=False)
            sd.print_schema()
            
        elif args.command == "convert":
            sd = CarpDataStream(args.files, load_participants=False)
            sd.convert_to_parquet(args.output, batch_size=args.batch_size)
            
        elif args.command == "count":
            sd = CarpDataStream(args.files, load_participants=False)
            count = sd.count_items()
            console.print(f"[bold green]Total items: {count:,}[/bold green]")
            
        elif args.command == "participants":
            sd = CarpDataStream(args.files, load_participants=True)
            sd.print_participants()
            
        elif args.command == "export":
            sd = CarpDataStream(args.files, load_participants=False)
            sd.export_to_json(args.output, data_type=args.data_type)
            
        elif args.command == "group":
            sd = CarpDataStream(args.files, load_participants=False)
            sd.group_by_field(args.field, args.output)
            
    except FileNotFoundError as e:
        console.print(f"[bold red]Error: {e}[/bold red]")
        return 1
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted.[/yellow]")
        return 130
    except Exception as e:
        console.print(f"[bold red]Error: {e}[/bold red]")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
