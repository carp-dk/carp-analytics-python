"""Additional edge-case coverage for CLI and type-generation helpers."""

from __future__ import annotations

import runpy
from pathlib import Path

from carp import CarpStudy
from carp.core.fields import get_nested_value
from carp.participants.parser import load_participant_file
from carp.types.infer import _maybe_json_string, infer_schema, merge_schema
from carp.types.render import render_types


def test_cli_module_entrypoint(monkeypatch) -> None:
    """Execute the module-level CLI entrypoint."""

    exit_codes = []
    monkeypatch.setattr("carp.commandline.app.main", lambda: 7)
    monkeypatch.setattr("sys.exit", lambda code: exit_codes.append(code))
    runpy.run_module("carp.cli", run_name="__main__")
    assert exit_codes == [7]


def test_parser_and_schema_edge_branches(study_paths: list[Path], tmp_path: Path) -> None:
    """Exercise parser branches not covered by the default fixture."""

    assert get_nested_value({"a": 1}, "a.b", "fallback") == "fallback"
    assert CarpStudy(study_paths).schema.cached()["dk.cachet.carp.location"] == ["latitude", "longitude"]

    participant_file = tmp_path / "participant-data.json"
    participant_file.write_text(
        """
        [
          {"roles": [{"data": {}}]},
          {
            "studyDeploymentId": "string-ssn",
            "roles": [
              {
                "roleName": "Participant",
                "data": {
                  "dk.carp.webservices.input.ssn": "2222",
                  "dk.carp.webservices.input.informed_consent": {
                    "name": "eve@example.com",
                    "consent": "{broken json}",
                    "note": 1
                  }
                }
              }
            ]
          },
          {
            "studyDeploymentId": "non-string-consent",
            "roles": [
              {
                "roleName": "Participant",
                "data": {
                  "dk.carp.webservices.input.informed_consent": {
                    "name": "nonstr@example.com",
                    "consent": 1
                  }
                }
              }
            ]
          }
        ]
        """,
        encoding="utf-8",
    )
    loaded = load_participant_file(participant_file)
    assert loaded["string-ssn"].ssn == "2222"
    assert loaded["string-ssn"].email == "eve@example.com"
    assert loaded["string-ssn"].full_name is None
    assert loaded["non-string-consent"].email == "nonstr@example.com"


def test_type_inference_and_rendering_edge_branches() -> None:
    """Exercise edge branches in schema inference and code rendering."""

    assert _maybe_json_string("plain text") is None
    assert _maybe_json_string("{broken}") is None
    schema = {}
    merge_schema(schema, None)
    merge_schema(schema, {"value": [1, 2.0]})
    assert schema["nullable"] is True
    assert infer_schema(iter([{"a": 1}, {"a": 2}]), sample_size=0)["fields"] == {}

    rendered = render_types(
        {
            "type": "object",
            "fields": {
                "child": {"type": "object", "fields": {}},
                "other": {"type": "object", "fields": {"child": {"type": "object", "fields": {"value": {"type": "primitive", "python_type": "int"}}}}},
                "matching": {"type": "object", "fields": {"child": {"type": "object", "fields": {}}}},
                "items": {"type": "list", "item_type": {"type": "object", "fields": {"from": {"type": "primitive", "python_type": "str"}}}},
                "mystery": {},
            },
        }
    )
    assert "class Child:" in rendered
    assert "class ChildItem:" in rendered
    assert "from_: str = None" in rendered
    assert "mystery: Any = None" in rendered
