"""Tests for shared CARP helpers."""

from __future__ import annotations

from pathlib import Path

import pytest

from carp.core.dependencies import import_or_raise, module_available
from carp.core.fields import collect_field_paths, deployment_id_from_record, get_nested_value
from carp.core.files import JsonArrayWriter, iter_json_array, resolve_paths
from carp.core.naming import parquet_stem, sanitize_filename
from carp.participants.directory import ParticipantDirectory
from carp.participants.parser import load_participant_file


def test_core_helpers_cover_nested_values_and_paths(study_paths: list[Path]) -> None:
    """Exercise shared path and field helpers."""

    record = next(iter_json_array(study_paths[0]))
    assert resolve_paths(study_paths) == tuple(study_paths)
    assert get_nested_value(record, "measurement.data.steps") == 100
    assert get_nested_value(record, "missing.value", "fallback") == "fallback"
    assert deployment_id_from_record(record) == "deploy-email-a"
    assert "measurement.data.steps" in collect_field_paths(record)
    assert sanitize_filename("alice@example.com", allowed="-_.@") == "alice@example.com"
    assert parquet_stem("dk.cachet.carp.stepcount") == "dk.cachet.carp__stepcount"


def test_json_array_writer_and_module_helpers(tmp_path: Path) -> None:
    """Exercise JSON writing and optional dependency errors."""

    output_path = tmp_path / "output.json"
    writer = JsonArrayWriter(output_path)
    writer.write({"value": 1})
    writer.write({"value": 2})
    writer.close()
    assert output_path.read_text(encoding="utf-8") == '[{"value": 1},{"value": 2}]'
    assert module_available("json") is True
    with pytest.raises(RuntimeError):
        import_or_raise("module_that_does_not_exist_for_tests", "test")


def test_participant_loader_handles_invalid_consent(tmp_path: Path) -> None:
    """Exercise parser branches for invalid consent payloads and missing folders."""

    participant_file = tmp_path / "participant-data.json"
    participant_file.write_text(
        '[{"studyDeploymentId":"x","roles":[{"roleName":"Participant","data":{"dk.carp.webservices.input.informed_consent":"broken"}}]}]',
        encoding="utf-8",
    )
    loaded = load_participant_file(participant_file)
    assert loaded["x"].consent_signed is False
    empty_directory = ParticipantDirectory.from_folders((tmp_path / "missing",))
    assert empty_directory.summary_rows() == []
