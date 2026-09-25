from __future__ import annotations

from typing import Any

import pytest
from PIL import Image

from camera_trap_pipeline.mock_models import (
    CrashingDetector,
    MockClassifier,
    MockDetector,
    SimulatedCrash,
    iou,
    mock_label,
)
from camera_trap_pipeline.synthetic import GroundTruth, SyntheticDataset
from camera_trap_pipeline.taxonomy import parse_label

BLANK_IMAGE = Image.new("RGB", (4, 4))


def first_animal_image(truth: GroundTruth) -> tuple[str, dict[str, Any]]:
    for rel, entry in truth.images.items():
        objects = entry["objects"]
        if objects and all(o["category"] == "animal" for o in objects) and not entry["night"]:
            return rel, entry
    raise AssertionError("dataset has no daytime animal photo")


def test_iou() -> None:
    assert iou([0, 0, 1, 1], [0, 0, 1, 1]) == 1.0
    assert iou([0, 0, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5]) == 0.0
    assert iou([0, 0, 0.5, 1], [0.25, 0, 0.5, 1]) == pytest.approx(1 / 3)
    assert iou([0, 0, 0, 0], [0, 0, 0, 0]) == 0.0


def test_mock_label_parses_like_speciesnet() -> None:
    label = parse_label(mock_label("coyote", "mammalia;carnivora;canidae;canis;latrans"))
    assert label.common_name == "coyote"
    assert label.depth == 5
    assert parse_label(mock_label("animal", "")).depth == 0
    assert mock_label("coyote", "x") == mock_label("coyote", "x")


def test_detector_returns_every_truth_box_above_counting_threshold(
    tiny_dataset: SyntheticDataset, tiny_truth: GroundTruth
) -> None:
    rel, entry = first_animal_image(tiny_truth)
    detector = MockDetector(tiny_truth, seed=1)
    boxes = detector.detect(BLANK_IMAGE, path=str(tiny_dataset.root / rel))
    counted = [b for b in boxes if b["conf"] >= 0.2]
    assert len(counted) == len(entry["objects"])
    assert boxes == detector.detect(BLANK_IMAGE, path=rel)
    assert detector.detect(BLANK_IMAGE, path="elsewhere/unknown.jpg") == []


def test_classifier_puts_true_species_first(
    tiny_dataset: SyntheticDataset, tiny_truth: GroundTruth
) -> None:
    rel, entry = first_animal_image(tiny_truth)
    objects = entry["objects"]
    classifier = MockClassifier(tiny_truth, seed=1)
    results = classifier.classify(BLANK_IMAGE, [o["bbox"] for o in objects], path=rel)
    for obj, result in zip(objects, results, strict=True):
        assert parse_label(result["classes"][0]).common_name == obj["species"]
        assert result["scores"] == sorted(result["scores"], reverse=True)
        assert sum(result["scores"]) == pytest.approx(1.0, abs=1e-3)
    unmatched = classifier.classify(BLANK_IMAGE, [[0.0, 0.0, 0.01, 0.01]], path=rel)
    assert parse_label(unmatched[0]["classes"][0]).common_name == "blank"


def test_crashing_detector_stops_after_limit(tiny_truth: GroundTruth) -> None:
    detector = CrashingDetector(tiny_truth, seed=1, crash_after=2)
    detector.detect(BLANK_IMAGE, path="a.jpg")
    detector.detect(BLANK_IMAGE, path="b.jpg")
    with pytest.raises(SimulatedCrash):
        detector.detect(BLANK_IMAGE, path="c.jpg")
    assert not isinstance(SimulatedCrash(), Exception)
