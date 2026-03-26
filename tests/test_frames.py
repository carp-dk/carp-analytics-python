"""Tests for dataframe and parquet services."""

from __future__ import annotations


def test_dataframe_loading_and_participant_columns(study) -> None:
    """Exercise dataframe loading from JSON and participant enrichment."""

    frame = study.frames.get_dataframe("dk.cachet.carp.stepcount")
    assert frame.shape[0] == 4
    enriched = study.frames.get_dataframe_with_participants("dk.cachet.carp.weather")
    assert enriched.loc[0, "participant_id"] is None
    assert study.frames.parquet_path("dk.cachet.carp.stepcount", "out").name == "dk.cachet.carp__stepcount.parquet"


def test_parquet_conversion_and_reload(study, tmp_path) -> None:
    """Exercise namespace-aware parquet conversion and reload."""

    output_dir = tmp_path / "parquet"
    created = study.frames.convert_to_parquet(output_dir, batch_size=1)
    assert {path.name for path in created} == {
        "com.acme__stepcount.parquet",
        "dk.cachet.carp__location.parquet",
        "dk.cachet.carp__stepcount.parquet",
        "dk.cachet.carp__survey.parquet",
        "dk.cachet.carp__weather.parquet",
    }
    frame = study.frames.get_dataframe("dk.cachet.carp.stepcount", output_dir)
    assert set(frame.columns) >= {"studyDeploymentId", "measurement"}
