from __future__ import annotations

import json
import sys
from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest
from PIL import Image

from camera_trap_pipeline._types import LabelScores
from camera_trap_pipeline.classify import (
    SpeciesNetClassifier,
    box_key,
    run_classify,
    select_boxes,
)
from camera_trap_pipeline.detect import BOX_COLUMNS
from camera_trap_pipeline.errors import ImageFailedError, ModelDependencyError
from camera_trap_pipeline.settings import ClassificationSettings, DetectionSettings
from camera_trap_pipeline.taxonomy import UNRESOLVED

COYOTE = "id4;mammalia;carnivora;canidae;canis;latrans;coyote"
DOG = "id7;mammalia;carnivora;canidae;canis;familiaris;domestic dog"


class FakeClassifier:
    name = "fake-classifier"

    def __init__(self) -> None:
        self.calls: list[int] = []

    def classify(
        self, image: Image.Image, boxes: Sequence[Sequence[float]], *, path: str
    ) -> list[LabelScores]:
        self.calls.append(len(boxes))
        if Path(path).name == "bad.jpg":
            raise ImageFailedError("classifier failed")
        return [{"classes": [COYOTE, DOG], "scores": [0.5, 0.3]} for _ in boxes]


def box_table(rows: list[tuple[str, int, str, float]]) -> pd.DataFrame:
    records = [
        {
            "photo_id": pid,
            "box_idx": idx,
            "category": category,
            "conf": conf,
            "x": 0.1 * idx,
            "y": 0.2,
            "w": 0.1,
            "h": 0.1,
            "model": "m",
        }
        for pid, idx, category, conf in rows
    ]
    return pd.DataFrame(records, columns=BOX_COLUMNS)


def test_select_boxes_keeps_counted_animals_capped_per_photo() -> None:
    boxes = box_table(
        [
            ("p1", 0, "animal", 0.9),
            ("p1", 1, "animal", 0.8),
            ("p1", 2, "animal", 0.7),
            ("p1", 3, "animal", 0.1),
            ("p1", 4, "person", 0.95),
            ("p2", 0, "animal", 0.3),
        ]
    )
    selected = select_boxes(
        boxes, DetectionSettings(count_conf=0.2), ClassificationSettings(max_boxes_per_photo=2)
    )
    assert list(zip(selected["photo_id"], selected["box_idx"], strict=True)) == [
        ("p1", 0),
        ("p1", 1),
        ("p2", 0),
    ]
    assert selected["box_key"].iloc[0] == box_key("p1", (0.0, 0.2, 0.1, 0.1))
    assert box_key("p", (0.12345, 0, 1, 0.5)) == "p:0.1235,0.0000,1.0000,0.5000"


def media_for(root: Path, names: list[str]) -> pd.DataFrame:
    (root / "cam").mkdir(parents=True, exist_ok=True)
    for name in names:
        Image.new("RGB", (40, 30)).save(root / "cam" / name)
    return pd.DataFrame(
        {"photo_id": [n.split(".")[0] for n in names], "rel_path": [f"cam/{n}" for n in names]}
    )


def test_run_classify_rolls_up_and_resumes(tmp_path: Path) -> None:
    media = media_for(tmp_path, ["a.jpg", "b.jpg"])
    boxes = box_table([("a", 0, "animal", 0.9), ("a", 1, "animal", 0.6), ("b", 0, "animal", 0.5)])
    checkpoint = tmp_path / "ck.jsonl"
    classifier = FakeClassifier()
    table = run_classify(
        boxes,
        media,
        tmp_path,
        lambda: classifier,
        checkpoint,
        DetectionSettings(),
        ClassificationSettings(),
    )
    assert classifier.calls == [2, 1]
    assert set(table["species"]) == {"canis (genus)"}
    assert table["species_level"].eq("genus").all()
    assert table["top1_label"].eq("coyote").all()
    assert json.loads(table["top5_json"].iloc[0]) == [["coyote", 0.5], ["domestic dog", 0.3]]

    def must_not_load() -> FakeClassifier:
        raise AssertionError("classifier loaded with nothing to do")

    again = run_classify(
        boxes,
        media,
        tmp_path,
        must_not_load,
        checkpoint,
        DetectionSettings(),
        ClassificationSettings(),
    )
    pd.testing.assert_frame_equal(again, table)


def test_classifier_errors_are_recorded_and_left_unresolved(tmp_path: Path) -> None:
    media = media_for(tmp_path, ["bad.jpg"])
    boxes = box_table([("bad", 0, "animal", 0.9)])
    table = run_classify(
        boxes,
        media,
        tmp_path,
        FakeClassifier,
        tmp_path / "ck.jsonl",
        DetectionSettings(),
        ClassificationSettings(),
    )
    assert table["species"].tolist() == [UNRESOLVED]
    assert table["error"].tolist() == ["ImageFailedError"]


def test_without_classifier_every_box_is_unresolved(tmp_path: Path) -> None:
    boxes = box_table([("a", 0, "animal", 0.9)])
    table = run_classify(
        boxes,
        pd.DataFrame({"photo_id": ["a"], "rel_path": ["cam/a.jpg"]}),
        tmp_path,
        None,
        tmp_path / "ck.jsonl",
        DetectionSettings(),
        ClassificationSettings(),
    )
    assert table["species"].tolist() == [UNRESOLVED]
    assert not (tmp_path / "ck.jsonl").exists()


def install_fake_speciesnet(monkeypatch: pytest.MonkeyPatch) -> list[Any]:
    seen: list[Any] = []

    class BBox:
        def __init__(self, xmin: float, ymin: float, width: float, height: float) -> None:
            self.values = (xmin, ymin, width, height)

    class Classifier:
        def __init__(self, model_name: str, device: str | None = None) -> None:
            seen.append((model_name, device))

        def preprocess(self, image: Image.Image, bboxes: list[BBox]) -> tuple[float, ...]:
            return bboxes[0].values

        def batch_predict(self, names: list[str], images: list[Any]) -> list[dict[str, Any]]:
            seen.append((names, images))
            return [
                {"classifications": {"classes": [COYOTE] * 6, "scores": [0.9] + [0.02] * 5}},
                {"failures": ["CLASSIFIER"]},
            ]

    monkeypatch.setitem(sys.modules, "speciesnet", SimpleNamespace(DEFAULT_MODEL="sn-default"))
    monkeypatch.setitem(
        sys.modules, "speciesnet.classifier", SimpleNamespace(SpeciesNetClassifier=Classifier)
    )
    monkeypatch.setitem(sys.modules, "speciesnet.utils", SimpleNamespace(BBox=BBox))
    return seen


def test_speciesnet_adapter_classifies_each_box(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = install_fake_speciesnet(monkeypatch)
    classifier = SpeciesNetClassifier(device="cpu")
    results = classifier.classify(
        Image.new("RGB", (8, 8)), [[0.1, 0.2, 0.3, 0.4], [0.5, 0.5, 0.1, 0.1]], path="cam/a.jpg"
    )
    assert classifier.name == "speciesnet:sn-default"
    assert seen[0] == ("sn-default", "cpu")
    assert seen[1] == (["cam/a.jpg#0", "cam/a.jpg#1"], [(0.1, 0.2, 0.3, 0.4), (0.5, 0.5, 0.1, 0.1)])
    assert len(results[0]["classes"]) == 5
    assert results[0]["scores"][0] == 0.9
    assert results[1] == {"classes": [], "scores": [], "error": "CLASSIFIER"}


def test_missing_speciesnet_package_gives_install_hint(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "speciesnet.classifier", None)
    with pytest.raises(ModelDependencyError, match=r"\.\[models\]"):
        SpeciesNetClassifier()


@pytest.mark.integration
def test_real_speciesnet_classifies_a_crop() -> None:
    classifier = SpeciesNetClassifier(device="cpu")
    image = Image.new("RGB", (640, 480), (90, 110, 60))
    results = classifier.classify(image, [[0.25, 0.25, 0.5, 0.5]], path="blank.jpg")
    assert len(results) == 1
    assert results[0]["classes"]
