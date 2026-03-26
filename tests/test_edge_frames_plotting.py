"""Additional edge-case coverage for frames and plotting."""

from __future__ import annotations

from types import SimpleNamespace

from carp.core.dependencies import import_or_raise
from carp.core.fields import collect_field_paths
from carp.plotting.prepare import candidate_series, frames_from_items, prepare_location_frame, prepare_step_frame
from carp.plotting.render import _merge_steps, render_heatmap


def test_frame_service_edge_branches(study, tmp_path) -> None:
    """Exercise dataframe and parquet helper branches."""

    pandas = import_or_raise("pandas", "test")
    pyarrow = import_or_raise("pyarrow", "test")
    assert study.records.list_fields(sample_size=0) == []
    assert collect_field_paths([]) == set()
    assert "items[]" in collect_field_paths({"items": []})
    assert study.frames.get_dataframe_with_participants("missing.type").empty
    nested = pandas.DataFrame({"dataStream": [{"studyDeploymentId": "nested-id"}]})
    assert study.frames._deployment_series(nested).tolist() == ["nested-id"]
    assert study.frames._participant_row("deploy-email-a")["participant_email"] == "alice@example.com"
    aligned = study.frames._align_table(
        pyarrow,
        pyarrow.Table.from_pylist([{"steps": 1}]),
        pyarrow.schema([("steps", pyarrow.float64()), ("cadence", pyarrow.int64())]),
    )
    assert aligned.column_names == ["steps", "cadence"]
    assert aligned["steps"][0].as_py() == 1.0
    assert aligned["cadence"][0].as_py() is None
    assert study.participant("alice@example.com").dataframe("missing.type").empty
    assert study.participant("alice@example.com").available_fields(sample_size=0) == []
    assert study.frames.convert_to_parquet(tmp_path / "flush", batch_size=50)
    assert study.frames.get_dataframe("missing.type", tmp_path / "flush").empty


def test_plotting_helpers_and_edge_paths(study, tmp_path, monkeypatch) -> None:
    """Exercise helper functions and low-probability plotting branches."""

    pandas = import_or_raise("pandas", "test")
    location_items = [
        SimpleNamespace(
            measurement=SimpleNamespace(
                data=SimpleNamespace(latitude=1.0, longitude=2.0),
                sensorStartTime=10,
            )
        ),
        SimpleNamespace(measurement=None),
    ]
    step_items = [
        SimpleNamespace(measurement=SimpleNamespace(data=SimpleNamespace(steps=3), sensorStartTime=10)),
        SimpleNamespace(measurement=SimpleNamespace(data=SimpleNamespace(steps=None), sensorStartTime=11)),
    ]
    location_frame, step_frame = frames_from_items(location_items, step_items)
    assert not location_frame.empty and not step_frame.empty
    assert candidate_series(pandas.DataFrame({"value": [1]}), ["missing", "value"]).tolist() == [1]
    assert candidate_series(pandas.DataFrame({"nested": [{"a": {"b": 1}}]}), ["nested.a.b"]).tolist() == [1]
    assert candidate_series(pandas.DataFrame({"value": [1]}), ["missing.path"]) is None
    assert list(prepare_location_frame(study.frames.get_dataframe("dk.cachet.carp.location"))["_lat"]) == [55.1, 55.2, 56.0]
    assert list(prepare_step_frame(study.frames.get_dataframe("dk.cachet.carp.stepcount"))["_steps"]) == [100, 50, 150, 70]
    assert render_heatmap(location_frame.iloc[0:0], step_frame, tmp_path / "empty.html") is None
    assert render_heatmap(location_frame, pandas.DataFrame({"_steps": [0], "_time": [10], "_lat": [1.0], "_lon": [2.0]}), tmp_path / "zero.html") is not None
    assert _merge_steps(pandas, location_frame, pandas.DataFrame({"_steps": [1]})).empty
    assert study.plots.unified("missing") is None
    assert study.plots.deployment("missing", output_file=str(tmp_path / "missing.html")) is None
    assert study.plots.deployment("deploy-email-a", location_type="missing.type", output_file=str(tmp_path / "noloc.html")) is None
    assert study.plots.deployment("deploy-email-a", step_type="missing.type", output_file=str(tmp_path / "nosteps.html")) is not None
    assert study.plots.from_items(location_items, step_items, output_file=str(tmp_path / "objects.html")) is not None
    monkeypatch.setattr(study.plots, "candidate_series", lambda *_args, **_kwargs: None)
    assert study.participant("alice@example.com").dataframe("dk.cachet.carp.stepcount").shape[0] == 4

    calls = {"count": 0}

    def staged_series(*_args, **_kwargs):
        calls["count"] += 1
        frame = _args[0]
        if calls["count"] == 1:
            return pandas.Series(["deploy-email-a"] * len(frame), index=frame.index)
        return None

    monkeypatch.setattr("carp.plotting.service.candidate_series", staged_series)
    assert study.plots.deployment("deploy-email-a", output_file=str(tmp_path / "staged.html")) is not None
