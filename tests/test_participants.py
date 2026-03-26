"""Tests for participant lookup and unified views."""

from __future__ import annotations

from carp.participants.view import ParticipantView


def test_participant_lookups_and_summary(study) -> None:
    """Exercise participant lookup methods and summary rows."""

    assert len(study.participants.by_email("alice@example.com")) == 2
    assert len(study.participants.by_ssn("1111")) == 2
    assert len(study.participants.by_name("Charlie Example")) == 2
    summary_rows = study.participants.summary_rows()
    assert len(summary_rows) == 4
    assert any(row["emails"] == "alice@example.com" for row in summary_rows)


def test_participant_view_info_fields_and_dataframe(study, tmp_path) -> None:
    """Exercise the participant-scoped view object."""

    participant = study.participant("alice@example.com")
    assert isinstance(participant, ParticipantView)
    info = participant.info()
    assert info is not None
    assert info["num_deployments"] == 2
    assert participant.count() == 4
    assert participant.data_types() == ["dk.cachet.carp.location", "dk.cachet.carp.stepcount"]
    assert "measurement.data.latitude" in participant.available_fields()
    assert "measurement.data.steps" in participant.available_fields()
    assert participant.dataframe("dk.cachet.carp.stepcount").shape[0] == 2
    assert participant.plot_location(output_file=str(tmp_path / "participant.html")) is not None


def test_missing_participant_view_and_unified_lookup(study) -> None:
    """Exercise missing participants and unified participant lookups."""

    missing = study.participant("nobody@example.com")
    assert missing.exists is False
    assert missing.info() is None
    unified_id = study.participant("alice@example.com").info()["unified_id"]
    assert len(study.participants.unified(unified_id)) == 2
