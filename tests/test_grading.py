"""The self-check: every check passes on a correct run and each one catches its own fault."""

from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd
import pytest

from camera_trap_pipeline.grading import GradingThresholds, grade_run
from camera_trap_pipeline.synthetic import GroundTruth

CHECK_NAMES = [
    "events match visits",
    "event types",
    "animal counts",
    "species calls",
    "unset clocks",
    "truncated uploads",
    "night frames",
    "fogged frames",
    "near-duplicates",
    "camera outage",
]


def verdicts(run: Path, truth: GroundTruth) -> dict[str, bool]:
    return {check.name: check.passed for check in grade_run(run, truth)}


def test_reference_run_passes_every_check(reference_run: Path, tiny_truth: GroundTruth) -> None:
    checks = grade_run(reference_run, tiny_truth)
    assert [c.name for c in checks] == CHECK_NAMES
    failed = [f"{c.name}: {c.detail}" for c in checks if not c.passed]
    assert failed == []
    details = {c.name: c.detail for c in checks}
    assert "0 misflagged" in details["unset clocks"]
    assert details["camera outage"].startswith("long silences on cam_03")


@pytest.fixture
def tampered(reference_run: Path, tmp_path: Path) -> Path:
    run = tmp_path / "run"
    shutil.copytree(reference_run, run)
    return run


def rewrite(run: Path, name: str, frame: pd.DataFrame) -> None:
    frame.to_parquet(run / name, index=False)


def test_merged_events_are_caught(tampered: Path, tiny_truth: GroundTruth) -> None:
    photos = pd.read_parquet(tampered / "photos.parquet")
    first_two = photos["event_id"].dropna().unique()[:2]
    photos.loc[photos["event_id"] == first_two[1], "event_id"] = first_two[0]
    rewrite(tampered, "photos.parquet", photos)
    assert not verdicts(tampered, tiny_truth)["events match visits"]


def test_wrong_counts_types_and_species_are_caught(tampered: Path, tiny_truth: GroundTruth) -> None:
    events = pd.read_parquet(tampered / "events.parquet")
    animal = events["observation_type"] == "animal"
    events.loc[animal, "max_animals"] += 1
    events.loc[~animal, "observation_type"] = "animal"
    events.loc[animal, "species"] = "not a label"
    rewrite(tampered, "events.parquet", events)
    result = verdicts(tampered, tiny_truth)
    assert not result["animal counts"]
    assert not result["event types"]
    assert not result["species calls"]


def test_quality_faults_are_caught(tampered: Path, tiny_truth: GroundTruth) -> None:
    photos = pd.read_parquet(tampered / "photos.parquet")
    photos["timestamp_valid"] = True
    photos["is_ir"] = ~photos["is_ir"]
    photos["is_blurry"] = False
    photos["decode_ok"] = photos["decode_ok"] | True
    photos["is_near_duplicate"] = False
    rewrite(tampered, "photos.parquet", photos)
    result = verdicts(tampered, tiny_truth)
    for name in ("unset clocks", "night frames", "fogged frames", "near-duplicates"):
        assert not result[name], name
    assert not result["truncated uploads"]


def test_missing_outage_is_caught(tampered: Path, tiny_truth: GroundTruth) -> None:
    gaps = pd.read_parquet(tampered / "summary" / "camera_gaps.parquet")
    rewrite(tampered, "summary/camera_gaps.parquet", gaps.iloc[0:0])
    assert not verdicts(tampered, tiny_truth)["camera outage"]


def test_thresholds_are_configurable(reference_run: Path, tiny_truth: GroundTruth) -> None:
    impossible = GradingThresholds(species_agreement=1.01)
    checks = {c.name: c.passed for c in grade_run(reference_run, tiny_truth, impossible)}
    assert not checks["species calls"]
