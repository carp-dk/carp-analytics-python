"""Shared pytest fixtures for CARP Analytics."""

from __future__ import annotations

from pathlib import Path

import pytest

from carp import CarpStudy


@pytest.fixture()
def fixture_root() -> Path:
    """Return the self-contained multi-phase fixture root."""

    return Path(__file__).parent / "fixtures" / "multi_phase"


@pytest.fixture()
def study_paths(fixture_root: Path) -> list[Path]:
    """Return the default synthetic study file paths."""

    return [
        fixture_root / "phase_a" / "data-streams.json",
        fixture_root / "phase_b" / "data-streams.json",
    ]


@pytest.fixture()
def study(study_paths: list[Path]) -> CarpStudy:
    """Return a study backed by self-contained fixtures."""

    return CarpStudy(study_paths)
