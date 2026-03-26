"""Compact schema-discovery example for `CarpStudy`."""

from __future__ import annotations

import sys
from pathlib import Path

from carp import CarpStudy


def _default_paths() -> list[Path]:
    """Return bundled data-stream files for schema discovery."""

    sleep_paths = sorted(Path("sleep-data").glob("phase-*/data-streams.json"))
    if sleep_paths:
        return sleep_paths
    return sorted(Path("tests/fixtures/multi_phase").glob("*/data-streams.json"))


def main() -> int:
    """Load a study and print schema and field examples."""

    file_paths = [Path(arg) for arg in sys.argv[1:]] or _default_paths()
    study = CarpStudy(file_paths, load_participants=False)
    print("Observed data types:")
    for data_type in study.records.data_types():
        print(f"  - {data_type}")
    print("\nSchema summary:")
    for data_type, fields in study.schema.scan().items():
        print(f"  {data_type}: {', '.join(fields)}")
    print("\nSample field paths:")
    for field in study.records.list_fields(sample_size=3)[:12]:
        print(f"  - {field}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
