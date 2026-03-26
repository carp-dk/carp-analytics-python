#!/usr/bin/env python3
"""End-to-end example usage for `CarpStudy`."""

from __future__ import annotations

import sys
from pathlib import Path

from carp import CarpStudy


def _default_paths() -> list[Path]:
    """Return bundled study paths for the example."""

    sleep_paths = sorted(Path("sleep-data").glob("phase-*/data-streams.json"))
    if sleep_paths:
        return sleep_paths
    fixture_root = Path("tests/fixtures/multi_phase")
    return sorted(fixture_root.glob("*/data-streams.json"))


def main() -> int:
    """Run the example against one or more study files."""

    file_paths = [Path(arg) for arg in sys.argv[1:]] or _default_paths()
    study = CarpStudy(file_paths, load_participants=True)
    print(f"Loaded {len(file_paths)} study file(s)")
    print(f"Total records: {study.records.count():,}")
    print(f"Data types: {', '.join(study.records.data_types())}")
    rows = study.participants.summary_rows()
    print(f"Unified participants: {len(rows)}")
    for row in rows[:3]:
        print(f"  {row['unified_id']}: {row['emails']} ({row['deployments']} deployments)")
    example_email = next((row["emails"] for row in rows if row["emails"] != "N/A"), None)
    if example_email:
        participant = study.participant(example_email)
        print(f"Example participant: {participant.info()}")
    try:
        step_frame = study.frames.get_dataframe("dk.cachet.carp.stepcount")
    except RuntimeError as exc:
        print(f"Skipping dataframe example: {exc}")
    else:
        print("Step-count preview:")
        print(step_frame.head().to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
