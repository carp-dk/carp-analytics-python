"""Tests for generated types and plotting services."""

from __future__ import annotations

import importlib.util
import sys


def test_generate_type_definitions(study, tmp_path) -> None:
    """Exercise generated type definitions for JSON-string payloads."""

    output_path = study.types.generate(tmp_path / "generated_types.py", sample_size=11)
    code = output_path.read_text(encoding="utf-8")
    assert "parse_json_field" in code
    assert "class StudyItem" in code
    spec = importlib.util.spec_from_file_location("generated_types", output_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    payload = {"measurement": {"data": {"response_json": '{"score": 1}'}}}
    instance = module.StudyItem.from_dict(payload)
    assert instance.measurement.data.response_json.score == 1


def test_plot_service_outputs_html(study, tmp_path) -> None:
    """Exercise participant, deployment, unified, and item-based plots."""

    participant_path = study.plots.participant("alice@example.com", output_file=str(tmp_path / "alice.html"))
    assert participant_path is not None
    assert "leaflet" in (tmp_path / "alice.html").read_text(encoding="utf-8").lower()
    unified_id = study.participant("alice@example.com").info()["unified_id"]
    assert study.plots.unified(unified_id, output_file=str(tmp_path / "unified.html")) is not None
    assert study.plots.deployment("deploy-email-a", output_file=str(tmp_path / "solo.html"), include_steps=False) is not None
    location_items = []
    assert study.plots.from_items(location_items, output_file=str(tmp_path / "none.html")) is None


def test_plot_service_handles_missing_filters(study, monkeypatch, tmp_path) -> None:
    """Exercise plot branches for missing participants and missing columns."""

    assert study.plots.participant("missing@example.com") is None
    monkeypatch.setattr("carp.plotting.service.candidate_series", lambda *_args, **_kwargs: None)
    assert study.plots.deployment("deploy-email-a", output_file=str(tmp_path / "missing.html")) is None
