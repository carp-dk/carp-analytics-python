"""Structural repository tests."""

from __future__ import annotations

from pathlib import Path


def test_python_files_stay_under_two_hundred_lines() -> None:
    """Enforce the 200-line limit for Python source and test files."""

    root = Path(__file__).resolve().parents[1]
    python_files = [
        path
        for path in root.rglob("*.py")
        if all(part not in {".venv.nosync", "dist", "__pycache__"} for part in path.parts)
    ]
    offenders = []
    for path in python_files:
        line_count = len(path.read_text(encoding="utf-8").splitlines())
        if line_count > 200:
            offenders.append((path.relative_to(root), line_count))
    assert offenders == []
