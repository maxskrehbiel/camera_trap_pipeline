from __future__ import annotations

import argparse
from collections.abc import Callable
from pathlib import Path

import pandas as pd
import pytest

from camera_trap_pipeline import __version__
from camera_trap_pipeline.cli import (
    fraction,
    iso_date,
    main,
    non_negative_int,
    positive_float,
    positive_int,
)
from camera_trap_pipeline.grading import Check
from camera_trap_pipeline.taxonomy import UNRESOLVED


@pytest.fixture(scope="module")
def photos(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("cli") / "photos"
    assert main(["synth", str(root), "--days", "2", "--cameras", "2", "--seed", "4"]) == 0
    return root


def mock_run_args(photos: Path, out: Path) -> list[str]:
    return ["run", str(photos), "--out", str(out), "--detector", "mock", "--classifier", "mock"]


def test_synth_writes_photos_and_truth(photos: Path) -> None:
    assert (photos / "ground_truth.json").exists()
    assert any(photos.glob("cam_01/*.jpg"))


def test_run_status_and_redo(
    photos: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "run"
    args = mock_run_args(photos, out)
    args += ["--workers", "2", "--strip-bottom", "0.07", "--event-gap", "90", "--gap-hours", "24"]
    assert main(args) == 0
    captured = capsys.readouterr()
    assert "report" in captured.out and "done" in captured.out
    assert captured.err == "", "the default log level is WARNING"

    assert main(["status", str(out)]) == 0
    status_lines = capsys.readouterr().out.splitlines()
    assert status_lines[0].split()[:2] == ["ingest", "done"]
    assert len(status_lines) == 7

    assert main(["-v", *args, "--redo", "report"]) == 0
    assert "[report] running" in capsys.readouterr().err
    assert (out / "report.html").exists()


def test_run_without_classifier_leaves_species_unresolved(photos: Path, tmp_path: Path) -> None:
    out = tmp_path / "run"
    args = ["run", str(photos), "--out", str(out), "--detector", "mock", "--classifier", "none"]
    assert main(args) == 0
    species = pd.read_parquet(out / "event_species.parquet")["species"]
    assert set(species) == {UNRESOLVED}


def test_missing_model_package_exits_3(
    photos: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    args = ["run", str(photos), "--out", str(tmp_path / "run"), "--device", "cpu"]
    assert main([*args, "--classifier", "none", "--detector-model", "MDV5A"]) == 3
    err = capsys.readouterr().err
    assert err.startswith("error: MegaDetector is optional")
    assert "Traceback" not in err


def test_status_of_missing_run_exits_2(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["status", str(tmp_path)]) == 2
    assert "no status.json" in capsys.readouterr().err


def test_input_errors_exit_2(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code = main(["fetch_sample", str(tmp_path), "--metadata-url", "http://example.org/meta.json"])
    assert code == 2
    assert "only https" in capsys.readouterr().err
    (tmp_path / "photos").mkdir()
    assert main(mock_run_args(tmp_path / "photos", tmp_path / "run")) == 2
    assert "ground_truth.json" in capsys.readouterr().err


def test_bad_argument_values_exit_2(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(["demo", "--days", "0"])
    assert excinfo.value.code == 2
    assert "must be at least 1" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("parse", "good", "expected", "bad"),
    [
        (positive_int, "3", 3, ["0", "x"]),
        (non_negative_int, "0", 0, ["-1", "1.5"]),
        (positive_float, "2.5", 2.5, ["0", "nan", "x"]),
        (fraction, "0.2", 0.2, ["1.5", "-0.1", "x"]),
        (iso_date, "2024-06-01", "2024-06-01", ["06/01/2024"]),
    ],
)
def test_argument_types(
    parse: Callable[[str], object], good: str, expected: object, bad: list[str]
) -> None:
    assert parse(good) == expected
    for value in bad:
        with pytest.raises(argparse.ArgumentTypeError):
            parse(value)


def test_demo_self_check_and_exit_codes(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    args = ["demo", "--out", str(tmp_path / "out"), "--days", "2", "--cameras", "2"]
    assert main([*args, "--simulate-crash"]) == 0
    printed = capsys.readouterr().out
    assert "crashed after" in printed
    assert "PASS  events match visits" in printed
    assert "FAIL" not in printed
    assert (tmp_path / "out" / "report.html").exists()

    failing = [Check("events match visits", False, "1 events for 2 visits")]
    monkeypatch.setattr("camera_trap_pipeline.demo.grade_run", lambda run, truth: failing)
    assert main(args) == 1
    assert "FAIL  events match visits: 1 events for 2 visits" in capsys.readouterr().out


def test_version_flag(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(["--version"])
    assert excinfo.value.code == 0
    assert __version__ in capsys.readouterr().out
