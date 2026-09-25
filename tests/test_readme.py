"""The README's Python API example must run as written, offline."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

README = Path(__file__).resolve().parents[1] / "README.md"


def python_api_example() -> str:
    text = README.read_text(encoding="utf-8")
    section = text.split("### Python API", 1)[1].split("\n## ", 1)[0]
    match = re.search(r"```python\n(.*?)```", section, re.DOTALL)
    assert match, "the Python API section has no python code block"
    return match.group(1)


def test_python_api_example_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    exec(compile(python_api_example(), str(README), "exec"), {"__name__": "readme_example"})
    printed = capsys.readouterr().out
    assert "animal" in printed
    assert (tmp_path / "demo_output" / "api_run" / "report.html").exists()
