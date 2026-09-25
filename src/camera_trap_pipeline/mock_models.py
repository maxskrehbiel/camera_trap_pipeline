"""Stand-in detector and classifier that read synthetic ground truth instead of pixels.

Their output is noisy on purpose so thresholds and taxonomic roll-up are exercised offline.
"""

from __future__ import annotations

import hashlib
import uuid
from collections.abc import Sequence

import numpy as np
from PIL import Image

from ._types import Box, LabelScores
from .synthetic import GroundTruth
from .synthetic_schedule import SPECIES_BY_NAME

TAXONOMY_FIELDS = 5
SPURIOUS_BOX_RATE = 0.15
SMALL_BOX_AREA = 0.01
MATCH_IOU = 0.3
MIN_TRUE_BOX_CONF = 0.21
CONFUSER_WEIGHTS = (6.0, 2.5, 1.2, 0.6)
_LABEL_NAMESPACE = uuid.uuid5(uuid.NAMESPACE_URL, "camera_trap_pipeline/mock_labels")


class SimulatedCrash(KeyboardInterrupt):
    """Raised by ``CrashingDetector`` to imitate a killed process or lost session."""


def mock_label(common_name: str, lineage: str) -> str:
    """Build a SpeciesNet-style label string with a deterministic, obviously synthetic id.

    Args:
        common_name: Common name.
        lineage: Up to five taxonomy fields joined with ``;``.

    Returns:
        ``<id>;<class>;<order>;<family>;<genus>;<species>;<common name>``.
    """
    fields = lineage.split(";") if lineage else []
    fields = (fields + [""] * TAXONOMY_FIELDS)[:TAXONOMY_FIELDS]
    label_id = uuid.uuid5(_LABEL_NAMESPACE, common_name)
    return ";".join([str(label_id), *fields, common_name])


def _rng_for(seed: int, *parts: str) -> np.random.Generator:
    digest = hashlib.blake2b("|".join(parts).encode("utf-8"), digest_size=8).digest()
    return np.random.default_rng([seed, int.from_bytes(digest, "big")])


def iou(a: Sequence[float], b: Sequence[float]) -> float:
    """Intersection over union of two ``[x, y, w, h]`` boxes.

    Args:
        a: First box.
        b: Second box.

    Returns:
        Overlap area divided by combined area (0 when both are empty).
    """
    ax1, ay1, bx1, by1 = a[0] + a[2], a[1] + a[3], b[0] + b[2], b[1] + b[3]
    inter_w = max(0.0, min(ax1, bx1) - max(a[0], b[0]))
    inter_h = max(0.0, min(ay1, by1) - max(a[1], b[1]))
    inter = inter_w * inter_h
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


class MockDetector:
    """Returns the ground-truth boxes with jitter and confidence noise."""

    name = "mock-detector"

    def __init__(self, truth: GroundTruth, seed: int) -> None:
        """Create a detector over one synthetic dataset.

        Args:
            truth: Ground truth of the dataset being processed.
            seed: Base seed; each image gets its own generator derived from it.
        """
        self.truth = truth
        self.seed = seed

    def detect(self, image: Image.Image, *, path: str) -> list[Box]:
        """Return noisy ground-truth boxes for the image at ``path``.

        Args:
            image: Unused; present to match the detector interface.
            path: Image path, used to look up the ground truth.

        Returns:
            One box per drawn object, sometimes plus a spurious low-confidence box; none for
            files outside the dataset.
        """
        entry = self.truth.entry_for(path)
        if entry is None:
            return []
        rng = _rng_for(self.seed, "detect", entry["time"], entry["camera_id"])
        low, high = (0.45, 0.92) if entry["night"] else (0.7, 0.98)
        boxes: list[Box] = []
        for obj in entry["objects"]:
            x, y, w, h = obj["bbox"]
            conf = float(rng.uniform(low, high)) - (0.15 if w * h < SMALL_BOX_AREA else 0.0)
            jitter = rng.normal(0, 0.006, 4)
            boxes.append(
                {
                    "category": obj["category"],
                    "conf": max(conf, MIN_TRUE_BOX_CONF),
                    "bbox": [x + jitter[0], y + jitter[1], w + jitter[2], h + jitter[3]],
                }
            )
        if rng.random() < SPURIOUS_BOX_RATE:
            boxes.append(
                {
                    "category": "animal",
                    "conf": float(rng.uniform(0.02, 0.15)),
                    "bbox": [float(rng.uniform(0, 0.8)), float(rng.uniform(0.3, 0.7)), 0.08, 0.06],
                }
            )
        return boxes


class CrashingDetector(MockDetector):
    """A ``MockDetector`` that raises ``SimulatedCrash`` after a set number of photos."""

    def __init__(self, truth: GroundTruth, seed: int, crash_after: int) -> None:
        """Create the detector.

        Args:
            truth: Ground truth of the dataset.
            seed: Base seed.
            crash_after: Photos to process before crashing.
        """
        super().__init__(truth, seed)
        self.crash_after = crash_after
        self.calls = 0

    def detect(self, image: Image.Image, *, path: str) -> list[Box]:
        """Detect normally until the crash point, then raise.

        Args:
            image: Unused.
            path: Image path.

        Returns:
            The same boxes as ``MockDetector`` before the crash point.

        Raises:
            SimulatedCrash: On call number ``crash_after + 1``.
        """
        if self.calls >= self.crash_after:
            raise SimulatedCrash(f"simulated crash after {self.calls} photos")
        self.calls += 1
        return super().detect(image, path=path)


class MockClassifier:
    """Labels boxes with the matching ground-truth species, spreading score to look-alikes."""

    name = "mock-classifier"

    def __init__(self, truth: GroundTruth, seed: int) -> None:
        """Create a classifier over one synthetic dataset.

        Args:
            truth: Ground truth of the dataset being processed.
            seed: Base seed.
        """
        self.truth = truth
        self.seed = seed

    def classify(
        self, image: Image.Image, boxes: Sequence[Sequence[float]], *, path: str
    ) -> list[LabelScores]:
        """Label each box.

        Args:
            image: Unused.
            boxes: Normalized boxes to label.
            path: Image path, used to look up the ground truth.

        Returns:
            Top-5 labels and scores per box; ``blank`` for boxes that match no animal.
        """
        entry = self.truth.entry_for(path)
        animals = [o for o in entry["objects"] if o["category"] == "animal"] if entry else []
        night = bool(entry["night"]) if entry else False
        image_key = entry["time"] if entry else path
        results: list[LabelScores] = []
        for index, box in enumerate(boxes):
            rng = _rng_for(self.seed, "classify", image_key, str(index))
            match = max(animals, key=lambda o: iou(box, o["bbox"]), default=None)
            if match is None or match["species"] is None or iou(box, match["bbox"]) < MATCH_IOU:
                score = round(float(rng.uniform(0.5, 0.9)), 4)
                results.append({"classes": [mock_label("blank", "")], "scores": [score]})
                continue
            area = float(box[2]) * float(box[3])
            results.append(self._scores(match["species"], night, area, rng))
        return results

    @staticmethod
    def _scores(species: str, night: bool, area: float, rng: np.random.Generator) -> LabelScores:
        profile = SPECIES_BY_NAME[species]
        low, high = (0.42, 0.8) if night else (0.78, 0.96)
        top = float(rng.uniform(low, high)) - (0.12 if area < SMALL_BOX_AREA else 0.0)
        shares = rng.dirichlet(CONFUSER_WEIGHTS)
        labels = [mock_label(profile.name, profile.lineage)]
        labels += [mock_label(name, lineage) for name, lineage in profile.confusers]
        labels.append(mock_label("animal", ""))
        scores = [top] + [(1.0 - top) * float(s) for s in shares]
        order = sorted(range(len(labels)), key=lambda i: -scores[i])
        return {
            "classes": [labels[i] for i in order],
            "scores": [round(scores[i], 4) for i in order],
        }
