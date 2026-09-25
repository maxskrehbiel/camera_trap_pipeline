from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from camera_trap_pipeline.atomic_io import atomic_write_json, atomic_write_text, write_parquet


def test_atomic_write_replaces_whole_file_and_leaves_no_temp(tmp_path: Path) -> None:
    target = tmp_path / "nested" / "file.txt"
    atomic_write_text(target, "first")
    atomic_write_text(target, "second")
    assert target.read_text(encoding="utf-8") == "second"
    assert [p.name for p in target.parent.iterdir()] == ["file.txt"]


def test_atomic_write_json_serializes_unknown_types_as_strings(tmp_path: Path) -> None:
    target = tmp_path / "data.json"
    atomic_write_json(target, {"when": pd.Timestamp("2026-04-06 06:30")})
    assert json.loads(target.read_text(encoding="utf-8")) == {"when": "2026-04-06 06:30:00"}


def test_write_parquet_round_trips(tmp_path: Path) -> None:
    frame = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    target = tmp_path / "t.parquet"
    write_parquet(frame, target)
    pd.testing.assert_frame_equal(pd.read_parquet(target), frame)


def test_failed_parquet_write_keeps_old_file(tmp_path: Path) -> None:
    target = tmp_path / "t.parquet"
    write_parquet(pd.DataFrame({"a": [1]}), target)
    unwritable = pd.DataFrame({"a": [object()]})
    with pytest.raises(Exception, match=r"."):
        write_parquet(unwritable, target)
    assert pd.read_parquet(target)["a"].tolist() == [1]
    assert [p.name for p in tmp_path.iterdir()] == ["t.parquet"]
