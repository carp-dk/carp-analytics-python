"""Tests for JSON export and grouping flows."""

from __future__ import annotations

import json


def test_export_json_and_group_by_field(study, tmp_path) -> None:
    """Exercise JSON export and field-based grouping."""

    export_path = study.export.export_json(tmp_path / "records.json", "dk.cachet.carp.location")
    payload = json.loads(export_path.read_text(encoding="utf-8"))
    assert len(payload) == 3
    grouped = study.export.group_by_field("dataStream.dataType.namespace", tmp_path / "grouped")
    assert {path.name for path in grouped} == {"com.acme.json", "dk.cachet.carp.json"}


def test_group_by_participant_and_identity(study, tmp_path) -> None:
    """Exercise participant-aware grouping flows."""

    participant_files = study.export.group_by_participant(tmp_path / "participants")
    identity_files = study.export.group_by_identity("email", tmp_path / "emails")
    assert len(participant_files) == 5
    assert {path.name for path in identity_files} == {"alice@example.com.json", "unknown.json"}
