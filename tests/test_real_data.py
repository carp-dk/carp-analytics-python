"""Optional real-data integration tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from carp import CarpStudy

SLEEP_DATA_ROOT = Path(__file__).resolve().parents[1] / "sleep-data"


@pytest.mark.skipif(not SLEEP_DATA_ROOT.exists(), reason="sleep-data is not available")
def test_real_data_smoke() -> None:
    """Exercise stable invariants on local real study data."""

    file_paths = sorted(SLEEP_DATA_ROOT.glob("phase-*/data-streams.json"))
    study = CarpStudy(file_paths)
    assert study.records.count() > 0
    assert len(study.records.data_types()) >= 3
    assert len(study.schema.scan()) >= 3
    assert len(study.participants.summary_rows()) >= 1
