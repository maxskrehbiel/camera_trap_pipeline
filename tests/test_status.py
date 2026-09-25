from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from camera_trap_pipeline.status import JsonlCheckpoint, RunStatus

STAGES = ("a", "b", "c")


def by_id(record: dict[str, Any]) -> str:
    return str(record["id"])


def test_status_round_trips_and_reports_states(tmp_path: Path) -> None:
    path = tmp_path / "status.json"
    status = RunStatus(path, STAGES)
    status.mark_started("a")
    status.mark_done("a", rows=3)
    status.mark_started("b")

    reloaded = RunStatus(path, STAGES)
    assert reloaded.is_done("a")
    assert not reloaded.is_done("b")
    assert reloaded.info("a")["rows"] == 3
    assert [(r.stage, r.state) for r in reloaded.rows()] == [
        ("a", "done"),
        ("b", "incomplete"),
        ("c", "pending"),
    ]


def test_invalidate_from_clears_later_stages_only(tmp_path: Path) -> None:
    status = RunStatus(tmp_path / "status.json", STAGES)
    for stage in STAGES:
        status.mark_done(stage)
    assert status.invalidate_from("b") == ["b", "c"]
    assert status.is_done("a")
    assert not status.is_done("b")
    assert not status.is_done("c")


def test_unknown_stage_is_rejected(tmp_path: Path) -> None:
    status = RunStatus(tmp_path / "status.json", STAGES)
    with pytest.raises(ValueError, match="unknown stage"):
        status.mark_done("z")


def test_checkpoint_drops_torn_last_line_and_repairs_file(tmp_path: Path) -> None:
    path = tmp_path / "ck.jsonl"
    path.write_text(
        '{"id": "p1", "v": 1}\n{"id": "p2", "v": 2}\n{"id": "p3", "v"', encoding="utf-8"
    )

    checkpoint = JsonlCheckpoint(path, by_id)
    assert set(checkpoint.records) == {"p1", "p2"}
    assert checkpoint.dropped_lines == 1
    assert path.read_text(encoding="utf-8").endswith("\n")

    with checkpoint:
        checkpoint.append({"id": "p3", "v": 3})
    lines = path.read_text(encoding="utf-8").splitlines()
    assert [json.loads(line)["id"] for line in lines] == ["p1", "p2", "p3"]


def test_checkpoint_later_record_wins_and_errors_are_retried(tmp_path: Path) -> None:
    path = tmp_path / "ck.jsonl"
    with JsonlCheckpoint(path, by_id) as checkpoint:
        checkpoint.append({"id": "p1", "error": "OSError"})
        assert checkpoint.needs("p1")
        checkpoint.append({"id": "p1", "v": 1})
        assert not checkpoint.needs("p1")
        assert checkpoint.needs("p2")

    reopened = JsonlCheckpoint(path, by_id)
    assert len(reopened) == 1
    assert "p1" in reopened
    assert reopened.get("p1") == {"id": "p1", "v": 1}
    assert [r["id"] for r in reopened] == ["p1"]


def test_checkpoint_batches_flush(tmp_path: Path) -> None:
    path = tmp_path / "ck.jsonl"
    with JsonlCheckpoint(path, by_id) as checkpoint:
        checkpoint.append({"id": "a"}, flush=False)
        checkpoint.append({"id": "b"}, flush=False)
        checkpoint.flush()
        assert len(path.read_text(encoding="utf-8").splitlines()) == 2


def test_records_without_the_key_are_dropped(tmp_path: Path) -> None:
    path = tmp_path / "ck.jsonl"
    path.write_text('{"id": "a"}\n{"other": 1}\n[1, 2]\n', encoding="utf-8")
    checkpoint = JsonlCheckpoint(path, by_id)
    assert list(checkpoint.records) == ["a"]
    assert checkpoint.dropped_lines == 2
