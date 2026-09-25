from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest
from PIL import Image

from camera_trap_pipeline._types import Box
from camera_trap_pipeline.detect import MegaDetector, clean_boxes, run_detect
from camera_trap_pipeline.errors import ImageFailedError, ModelDependencyError


class FakeDetector:
    name = "fake"

    def __init__(self, fail_on: tuple[str, ...] = ()) -> None:
        self.calls: list[str] = []
        self.fail_on = fail_on

    def detect(self, image: Image.Image, *, path: str) -> list[Box]:
        self.calls.append(Path(path).name)
        if Path(path).name in self.fail_on:
            raise ImageFailedError(f"cannot read {path}")
        return [
            {"category": "animal", "conf": 0.5, "bbox": [0.1, 0.1, 0.2, 0.2]},
            {"category": "person", "conf": 0.9, "bbox": [0.5, 0.2, 0.1, 0.4]},
        ]


def photo_tables(root: Path, names: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    (root / "cam").mkdir(parents=True, exist_ok=True)
    for name in names:
        if name != "missing.jpg":
            Image.new("RGB", (40, 30), (10, 20, 30)).save(root / "cam" / name)
    ids = [f"id_{n}" for n in names]
    media = pd.DataFrame({"photo_id": ids, "rel_path": [f"cam/{n}" for n in names]})
    quality = pd.DataFrame({"photo_id": ids, "decode_ok": True})
    return media, quality


def test_clean_boxes_clips_rounds_and_sorts() -> None:
    boxes = clean_boxes(
        [
            {"category": "animal", "conf": 0.31234, "bbox": [-0.1, 0.5, 0.3, 0.7]},
            {"category": "animal", "conf": 0.9, "bbox": [0.95, 0.95, 0.2, 0.2]},
        ]
    )
    assert [b["conf"] for b in boxes] == [0.9, 0.3123]
    assert boxes[0]["bbox"] == [0.95, 0.95, 0.05, 0.05]
    assert boxes[1]["bbox"] == [0.0, 0.5, 0.3, 0.5]


def test_run_detect_records_boxes_and_per_photo_errors(tmp_path: Path) -> None:
    media, quality = photo_tables(tmp_path, ["a.jpg", "b.jpg", "missing.jpg"])
    detector = FakeDetector(fail_on=("b.jpg",))
    boxes, runs = run_detect(media, quality, tmp_path, lambda: detector, tmp_path / "ck.jsonl")
    assert detector.calls == ["a.jpg", "b.jpg"]
    by_photo = runs.set_index("photo_id")
    assert by_photo.loc["id_a.jpg", "n_boxes"] == 2
    assert pd.isna(by_photo.loc["id_a.jpg", "error"])
    assert by_photo.loc["id_b.jpg", "error"].startswith("ImageFailedError: cannot read <input>")
    assert by_photo.loc["id_missing.jpg", "error"].startswith("FileNotFoundError")
    assert boxes["category"].tolist() == ["person", "animal"]
    assert boxes["box_idx"].tolist() == [0, 1]


def test_run_detect_resumes_and_retries_only_failures(tmp_path: Path) -> None:
    media, quality = photo_tables(tmp_path, ["a.jpg", "b.jpg"])
    checkpoint = tmp_path / "ck.jsonl"
    run_detect(media, quality, tmp_path, lambda: FakeDetector(("b.jpg",)), checkpoint)
    second = FakeDetector()
    _, runs = run_detect(media, quality, tmp_path, lambda: second, checkpoint)
    assert second.calls == ["b.jpg"]
    assert runs["error"].isna().all()

    def must_not_load() -> FakeDetector:
        raise AssertionError("model loaded with nothing to do")

    run_detect(media, quality, tmp_path, must_not_load, checkpoint)


def test_unexpected_model_errors_stop_the_stage_but_keep_progress(tmp_path: Path) -> None:
    media, quality = photo_tables(tmp_path, ["a.jpg", "b.jpg", "c.jpg"])

    class BrokenDetector(FakeDetector):
        def detect(self, image: Image.Image, *, path: str) -> list[Box]:
            if Path(path).name == "b.jpg":
                raise RuntimeError("CUDA out of memory")
            return super().detect(image, path=path)

    checkpoint = tmp_path / "ck.jsonl"
    with pytest.raises(RuntimeError, match="out of memory"):
        run_detect(media, quality, tmp_path, BrokenDetector, checkpoint)
    assert len(checkpoint.read_text(encoding="utf-8").splitlines()) == 1


def test_undecodable_photos_are_skipped(tmp_path: Path) -> None:
    media, quality = photo_tables(tmp_path, ["a.jpg", "b.jpg"])
    quality.loc[quality["photo_id"] == "id_b.jpg", "decode_ok"] = False
    detector = FakeDetector()
    _, runs = run_detect(media, quality, tmp_path, lambda: detector, tmp_path / "ck.jsonl")
    assert detector.calls == ["a.jpg"]
    assert runs["photo_id"].tolist() == ["id_a.jpg"]


def install_fake_megadetector(monkeypatch: pytest.MonkeyPatch, result: dict[str, Any]) -> list[Any]:
    loaded: list[Any] = []

    class Model:
        def generate_detections_one_image(
            self, image: Image.Image, image_id: str, detection_threshold: float
        ) -> dict[str, Any]:
            loaded.append((image_id, detection_threshold))
            return result

    def load_detector(model: str, force_cpu: bool = False) -> Model:
        loaded.append((model, force_cpu))
        return Model()

    module = SimpleNamespace(load_detector=load_detector)
    monkeypatch.setitem(sys.modules, "megadetector", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "megadetector.detection", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "megadetector.detection.run_detector", module)
    return loaded


def test_megadetector_adapter_maps_categories(monkeypatch: pytest.MonkeyPatch) -> None:
    result = {
        "detections": [
            {"category": "1", "conf": 0.8, "bbox": [0.1, 0.2, 0.3, 0.4]},
            {"category": "3", "conf": 0.4, "bbox": [0.5, 0.5, 0.2, 0.1]},
            {"category": "9", "conf": 0.2, "bbox": [0.0, 0.0, 0.1, 0.1]},
        ]
    }
    calls = install_fake_megadetector(monkeypatch, result)
    detector = MegaDetector("MDV5A", threshold=0.15, force_cpu=True)
    boxes = detector.detect(Image.new("RGB", (8, 8)), path="cam/a.jpg")
    assert detector.name == "megadetector:MDV5A"
    assert [b["category"] for b in boxes] == ["animal", "vehicle", "unknown"]
    assert boxes[0]["bbox"] == [0.1, 0.2, 0.3, 0.4]
    assert calls == [("MDV5A", True), ("cam/a.jpg", 0.15)]


def test_megadetector_adapter_raises_on_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    install_fake_megadetector(monkeypatch, {"failure": "Failure image access"})
    detector = MegaDetector()
    with pytest.raises(ImageFailedError, match="Failure image access"):
        detector.detect(Image.new("RGB", (8, 8)), path="x.jpg")


def test_missing_megadetector_package_gives_install_hint(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "megadetector.detection.run_detector", None)
    with pytest.raises(ModelDependencyError, match=r"\.\[models\]"):
        MegaDetector()


@pytest.mark.integration
def test_real_megadetector_runs_on_a_blank_frame() -> None:
    detector = MegaDetector(force_cpu=True)
    boxes = detector.detect(Image.new("RGB", (640, 480), (90, 110, 60)), path="blank.jpg")
    assert all(b["category"] in ("animal", "person", "vehicle") for b in boxes)
