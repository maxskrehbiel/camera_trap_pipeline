from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from camera_trap_pipeline.demo import WORK_MARKER, prepare_work_dir, run_demo
from camera_trap_pipeline.errors import InputError
from camera_trap_pipeline.synthetic import SyntheticSettings

SMALL = SyntheticSettings(n_cameras=3, n_days=4, width=192, height=128)
TEXT_SUFFIXES = (".csv", ".html")


def test_work_dir_is_only_wiped_when_the_demo_made_it(tmp_path: Path) -> None:
    foreign = tmp_path / "foreign"
    foreign.mkdir()
    (foreign / "keep.txt").write_text("mine", encoding="utf-8")
    with pytest.raises(InputError, match="not created by the demo"):
        prepare_work_dir(foreign)
    assert (foreign / "keep.txt").exists()

    ours = tmp_path / "ours"
    prepare_work_dir(ours)
    (ours / "old_output.txt").write_text("stale", encoding="utf-8")
    prepare_work_dir(ours)
    assert sorted(p.name for p in ours.iterdir()) == [WORK_MARKER]


def test_demo_crashes_resumes_self_checks_and_publishes(tmp_path: Path) -> None:
    out = tmp_path / "examples"
    result = run_demo(out, SMALL, seed=2, simulate_crash=True, work_dir=tmp_path / "work")
    assert result.crashed_after == result.n_images // 2
    assert result.passed, [c for c in result.checks if not c.passed]
    assert (tmp_path / "work" / "run" / "status.json").exists()
    names = {p.relative_to(out).as_posix() for p in result.published}
    assert {"report.html", "plots/camera_uptime.png", "summary/events.csv"} <= names
    for path in result.published:
        assert path.stat().st_size < 1_000_000
        if path.suffix in TEXT_SUFFIXES:
            data = path.read_bytes()
            assert b"\r\n" not in data
            assert data.endswith(b"\n")
            assert str(tmp_path).encode() not in data
    events = pd.read_csv(out / "summary" / "events.csv")
    assert len(events) == int(result.checks[0].detail.split()[0])


def test_text_outputs_are_byte_stable(tmp_path: Path) -> None:
    first = run_demo(tmp_path / "a", SMALL, seed=5)
    second = run_demo(tmp_path / "b", SMALL, seed=5)
    for path_a, path_b in zip(first.published, second.published, strict=True):
        if path_a.suffix in TEXT_SUFFIXES:
            assert path_a.read_bytes() == path_b.read_bytes(), path_a.name
