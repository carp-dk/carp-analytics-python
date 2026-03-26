"""Tests for CLI wiring and command execution."""

from __future__ import annotations

from argparse import Namespace

from carp.commandline import app as cli_app


def test_cli_commands_and_help(capsys, study_paths, tmp_path) -> None:
    """Exercise the public CLI commands."""

    assert cli_app.main(["--version"]) == 0
    assert cli_app.main([]) == 0
    assert cli_app.main(["schema", *map(str, study_paths)]) == 0
    assert cli_app.main(["count", *map(str, study_paths)]) == 0
    assert cli_app.main(["participants", *map(str, study_paths)]) == 0
    assert cli_app.main(
        ["export", *map(str, study_paths), "-o", str(tmp_path / "export.json"), "-t", "dk.cachet.carp.location"]
    ) == 0
    assert cli_app.main(["group", *map(str, study_paths), "-o", str(tmp_path / "grouped")]) == 0
    captured = capsys.readouterr().out
    assert "carp-analytics-python version" in captured
    assert "Total items" in captured


def test_cli_convert_and_error_paths(monkeypatch, capsys, study_paths, tmp_path) -> None:
    """Exercise CLI conversion and exception-handling branches."""

    assert cli_app.main(["convert", *map(str, study_paths), "-o", str(tmp_path / "parquet"), "--batch-size", "1"]) == 0
    assert cli_app.main(["count", "missing.json"]) == 1

    class FakeParser:
        """Minimal fake parser for exception tests."""

        def parse_args(self, _argv):
            return Namespace(version=False, command="test", handler=lambda _args: (_ for _ in ()).throw(KeyboardInterrupt()))

        def print_help(self):
            return None

    monkeypatch.setattr(cli_app, "_build_parser", lambda: FakeParser())
    assert cli_app.main(["ignored"]) == 130
    monkeypatch.setattr(
        cli_app,
        "_build_parser",
        lambda: type(
            "BrokenParser",
            (),
            {
                "parse_args": lambda self, _argv: Namespace(version=False, command="x", handler=lambda _args: (_ for _ in ()).throw(ValueError("boom"))),
                "print_help": lambda self: None,
            },
        )(),
    )
    assert cli_app.main(["ignored"]) == 1
    assert "Error: boom" in capsys.readouterr().out
