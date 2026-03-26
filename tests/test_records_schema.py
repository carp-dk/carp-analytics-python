"""Tests for record iteration and schema discovery."""

from __future__ import annotations


def test_record_filters_and_participant_enrichment(study) -> None:
    """Exercise record filtering and participant enrichment."""

    assert study.records.count() == 11
    assert study.records.count("dk.cachet.carp.stepcount") == 4
    filtered = list(study.records.iter_records(deployment_ids=("deploy-email-a",)))
    assert len(filtered) == 2
    enriched = list(study.records.iter_with_participants("dk.cachet.carp.stepcount"))
    assert all("_participant" in item for item in enriched)


def test_record_field_listing_data_types_and_schema_cache(study) -> None:
    """Exercise schema discovery and deployment-id fallback paths."""

    data_types = study.records.data_types()
    assert "com.acme.stepcount" in data_types
    assert "dk.cachet.carp.survey" in data_types
    assert "triggerIds[]" in study.records.list_fields()
    survey = list(study.records.iter_records("dk.cachet.carp.survey"))
    assert len(survey) == 2
    assert study.records.count(deployment_ids=("deploy-name-a", "deploy-name-b")) == 3
    schema = study.schema.scan()
    assert schema["dk.cachet.carp.stepcount"] == ["cadence", "steps"]
    assert study.schema.cached() == schema
