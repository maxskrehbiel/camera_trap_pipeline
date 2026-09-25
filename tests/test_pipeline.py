"""End-to-end runs on the synthetic dataset, checked against its ground truth."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from PIL import Image

from camera_trap_pipeline.errors import InputError
from camera_trap_pipeline.mock_models import (
    CrashingDetector,
    MockClassifier,
    MockDetector,
    SimulatedCrash,
)
from camera_trap_pipeline.pipeline import STAGES, Pipeline
from camera_trap_pipeline.settings import PipelineSettings
from camera_trap_pipeline.status import RunStatus
from camera_trap_pipeline.synthetic import GroundTruth, SyntheticDataset

SEED = 3


class CountingDetector(MockDetector):
    def __init__(self, truth: GroundTruth) -> None:
        super().__init__(truth, SEED)
        self.calls = 0

    def detect(self, image: Image.Image, *, path: str) -> list[dict[str, Any]]:
        self.calls += 1
        return super().detect(image, path=path)


class CountingClassifier(MockClassifier):
    def __init__(self, truth: GroundTruth) -> None:
        super().__init__(truth, SEED)
        self.calls = 0

    def classify(self, image: Image.Image, boxes: Any, *, path: str) -> list[dict[str, Any]]:
        self.calls += 1
        return super().classify(image, boxes, path=path)


def must_not_load() -> Any:
    raise AssertionError("a model was loaded although its stage had nothing to do")


def make_pipeline(
    root: Path,
    out: Path,
    settings: PipelineSettings,
    truth: GroundTruth,
    detector: Any = None,
    classifier: Any = None,
) -> Pipeline:
    return Pipeline(
        root,
        out,
        settings,
        (lambda: detector) if detector is not None else (lambda: MockDetector(truth, SEED)),
        (lambda: classifier) if classifier is not None else (lambda: MockClassifier(truth, SEED)),
        synthetic=True,
    )


def read(out: Path, name: str) -> pd.DataFrame:
    return pd.read_parquet(out / name)


def test_every_stage_finishes(reference_run: Path) -> None:
    status = RunStatus(reference_run / "status.json", STAGES)
    assert all(status.is_done(stage) for stage in STAGES)
    for name in ("report.html", "summary/camera_health.csv", "plots/activity_by_hour.png"):
        assert (reference_run / name).exists()


def test_report_has_no_local_paths_or_scripts(reference_run: Path) -> None:
    html = (reference_run / "report.html").read_text(encoding="utf-8")
    assert str(reference_run.parent) not in html
    assert "<script" not in html.lower()
    assert "Synthetic data" in html


def test_finished_run_loads_no_models(
    reference_run: Path, tiny_dataset: SyntheticDataset, settings: PipelineSettings
) -> None:
    status = Pipeline(
        tiny_dataset.root, reference_run, settings, must_not_load, must_not_load
    ).run()
    assert all(status.is_done(stage) for stage in STAGES)


def test_crash_mid_detection_resumes_where_it_stopped(
    tiny_dataset: SyntheticDataset,
    tiny_truth: GroundTruth,
    settings: PipelineSettings,
    reference_run: Path,
    tmp_path: Path,
) -> None:
    out = tmp_path / "run"
    crashing = CrashingDetector(tiny_truth, SEED, crash_after=40)
    with pytest.raises(SimulatedCrash):
        make_pipeline(tiny_dataset.root, out, settings, tiny_truth, detector=crashing).run()
    states = {row.stage: row.state for row in RunStatus(out / "status.json", STAGES).rows()}
    assert states["quality"] == "done"
    assert states["detect"] == "incomplete"
    assert states["classify"] == "pending"
    lines = (out / "checkpoints" / "detections.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 40

    counting = CountingDetector(tiny_truth)
    make_pipeline(tiny_dataset.root, out, settings, tiny_truth, detector=counting).run()
    decodable = int(read(out, "quality.parquet")["decode_ok"].sum())
    assert counting.calls == decodable - 40
    pd.testing.assert_frame_equal(
        read(out, "events.parquet"), read(reference_run, "events.parquet")
    )


def test_redo_detect_reuses_unchanged_classifications(
    tiny_dataset: SyntheticDataset,
    tiny_truth: GroundTruth,
    settings: PipelineSettings,
    reference_run: Path,
    tmp_path: Path,
) -> None:
    out = tmp_path / "run"
    shutil.copytree(reference_run, out)
    detector, classifier = CountingDetector(tiny_truth), CountingClassifier(tiny_truth)
    pipeline = make_pipeline(
        tiny_dataset.root, out, settings, tiny_truth, detector=detector, classifier=classifier
    )
    pipeline.run(redo=["detect"])
    assert detector.calls == int(read(out, "quality.parquet")["decode_ok"].sum())
    assert classifier.calls == 0
    assert RunStatus(out / "status.json", STAGES).is_done("report")


def test_new_photos_are_processed_incrementally(
    tiny_dataset: SyntheticDataset,
    tiny_truth: GroundTruth,
    settings: PipelineSettings,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    root, out = tmp_path / "photos", tmp_path / "run"
    shutil.copytree(tiny_dataset.root, root)
    held_back = tmp_path / "cam_03"
    shutil.move(root / "cam_03", held_back)
    make_pipeline(root, out, settings, tiny_truth).run()
    shutil.move(held_back, root / "cam_03")

    with caplog.at_level(logging.WARNING):
        Pipeline(root, out, settings, must_not_load, must_not_load).run()
    assert "were added since ingest" in caplog.text

    counting = CountingDetector(tiny_truth)
    make_pipeline(root, out, settings, tiny_truth, detector=counting).run(redo=["ingest"])
    media = read(out, "media.parquet")
    quality = read(out, "quality.parquet").merge(media[["photo_id", "camera_id"]], on="photo_id")
    new_decodable = int(quality.loc[quality["camera_id"] == "cam_03", "decode_ok"].sum())
    assert counting.calls == new_decodable
    assert set(read(out, "events.parquet")["camera_id"]) == {"cam_01", "cam_02", "cam_03"}


def test_bad_arguments_are_reported(
    tiny_dataset: SyntheticDataset, settings: PipelineSettings, tmp_path: Path
) -> None:
    with pytest.raises(InputError, match="unknown stage"):
        Pipeline(tiny_dataset.root, tmp_path / "a", settings, None, None).run(redo=["nope"])
    with pytest.raises(InputError, match="input folder not found"):
        Pipeline(tmp_path / "missing", tmp_path / "b", settings, None, None).run()
    with pytest.raises(InputError, match="no detector configured"):
        Pipeline(tiny_dataset.root, tmp_path / "c", settings, None, None).run()


def test_redo_all_clears_every_stage(
    reference_run: Path, tmp_path: Path, settings: PipelineSettings
) -> None:
    out = tmp_path / "run"
    shutil.copytree(reference_run, out)
    pipeline = Pipeline(tmp_path, out, settings, None, None)
    assert pipeline.apply_redo(["all"]) == list(STAGES)
    assert not (out / "media.parquet").exists()
    assert not (out / "summary").exists()
