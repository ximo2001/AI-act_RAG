"""Tests for the CLI error handling (no model or index needed)."""

from __future__ import annotations

from pathlib import Path

import pytest

from ai_act_rag.cli import main


def test_build_index_without_source_prints_hint_and_exits_1(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A missing AI Act file gives a one-line error with the download command, not a traceback."""
    monkeypatch.setenv("AIACT_SOURCE_PATH", str(tmp_path / "missing.xhtml"))
    monkeypatch.setenv("AIACT_INDEX_DIR", str(tmp_path / "index"))

    assert main(["build-index"]) == 1
    err = capsys.readouterr().err
    assert err.startswith("error: AI Act source not found")
    assert "download_ai_act.py" in err
