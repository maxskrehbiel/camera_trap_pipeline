"""Classify stage: a species label for each counted animal box, checkpointed box by box.

Checkpoint keys include box coordinates, so redoing detection only relabels changed boxes.
"""

from __future__ import annotations

import importlib
import json
import logging
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import pandas as pd
from PIL import Image

from ._types import ClassificationRecord, LabelScores
from .errors import ImageFailedError, ModelDependencyError
from .images import IMAGE_DECODE_ERRORS, load_rgb
from .parallel import ordered_map
from .settings import ClassificationSettings, DetectionSettings
from .status import JsonlCheckpoint
from .taxonomy import parse_label, roll_up

logger = logging.getLogger(__name__)

SPECIESNET_FALLBACK_MODEL = "kaggle:google/speciesnet/pyTorch/v4.0.3a/1"
TOP_K = 5

CLASSIFICATION_COLUMNS = [
    "photo_id",
    "box_idx",
    "box_key",
    "species",
    "species_score",
    "species_level",
    "lineage",
    "top1_label",
    "top1_score",
    "top5_json",
    "model",
    "error",
]


class Classifier(Protocol):
    """Anything that labels boxes within an image.

    Attributes:
        name: Model identifier stored with every result.
    """

    name: str

    def classify(
        self, image: Image.Image, boxes: Sequence[Sequence[float]], *, path: str
    ) -> list[LabelScores]:
        """Label each box within one image.

        Args:
            image: RGB image.
            boxes: Normalized ``[x_min, y_min, width, height]`` boxes.
            path: Source file, for classifiers that key results by path.

        Returns:
            One entry per box with labels and scores, most likely first.

        Raises:
            ImageFailedError: If the model cannot process this particular image.
        """
        ...


ClassifierFactory = Callable[[], Classifier]


class SpeciesNetClassifier:
    """SpeciesNet's image classifier applied to each box crop (weights download on first use).

    Only the classifier component is used. SpeciesNet's ensemble step (which also applies a
    geographic filter) is not, because it labels whole images rather than boxes.
    """

    def __init__(self, model: str | None = None, device: str | None = None) -> None:
        """Load the classifier.

        Args:
            model: SpeciesNet model name; defaults to the package's default model.
            device: ``cpu`` or ``cuda``; None lets SpeciesNet choose.

        Raises:
            ModelDependencyError: If the ``speciesnet`` package is not installed.
        """
        try:
            classifier_module = importlib.import_module("speciesnet.classifier")
            utils_module = importlib.import_module("speciesnet.utils")
            package = importlib.import_module("speciesnet")
        except ImportError as exc:
            raise ModelDependencyError(
                "SpeciesNet is optional: pip install -e '.[models]'"
            ) from exc
        model_name = model or getattr(package, "DEFAULT_MODEL", SPECIESNET_FALLBACK_MODEL)
        self._classifier = classifier_module.SpeciesNetClassifier(model_name, device=device)
        self._bbox = utils_module.BBox
        self.name = f"speciesnet:{model_name}"

    def classify(
        self, image: Image.Image, boxes: Sequence[Sequence[float]], *, path: str
    ) -> list[LabelScores]:
        """Classify each box crop.

        Args:
            image: RGB image.
            boxes: Normalized ``[x_min, y_min, width, height]`` boxes.
            path: Source file (used only to label results).

        Returns:
            The top five labels and scores per box; boxes SpeciesNet could not process
            carry an ``error``.
        """
        prepared = [self._classifier.preprocess(image, bboxes=[self._bbox(*box)]) for box in boxes]
        names = [f"{path}#{i}" for i in range(len(boxes))]
        results = self._classifier.batch_predict(names, prepared)
        out = []
        for result in results:
            found = result.get("classifications") or {}
            entry: LabelScores = {
                "classes": list(found.get("classes", []))[:TOP_K],
                "scores": [float(s) for s in found.get("scores", [])][:TOP_K],
            }
            if result.get("failures"):
                entry["error"] = "; ".join(map(str, result["failures"]))
            out.append(entry)
        return out


def box_key(photo_id: str, bbox: Sequence[float]) -> str:
    """Checkpoint key for one box: photo id plus rounded coordinates.

    Args:
        photo_id: Photo id.
        bbox: Normalized ``[x_min, y_min, width, height]``.

    Returns:
        ``<photo_id>:<x>,<y>,<w>,<h>`` with four decimals.
    """
    return photo_id + ":" + ",".join(f"{v:.4f}" for v in bbox)


def select_boxes(
    boxes: pd.DataFrame, detection: DetectionSettings, classification: ClassificationSettings
) -> pd.DataFrame:
    """Pick the boxes to classify: counted animal boxes, most confident first, capped per photo.

    Args:
        boxes: Box table from the detect stage.
        detection: Provides the counting confidence.
        classification: Provides the per-photo cap.

    Returns:
        The selected rows with an added ``box_key`` column.
    """
    selected = boxes[(boxes["category"] == "animal") & (boxes["conf"] >= detection.count_conf)]
    selected = selected.sort_values(["photo_id", "conf"], ascending=[True, False])
    selected = selected.groupby("photo_id", sort=False).head(classification.max_boxes_per_photo)
    selected = selected.copy()
    selected["box_key"] = [
        box_key(pid, (x, y, w, h))
        for pid, x, y, w, h in zip(
            selected["photo_id"],
            selected["x"],
            selected["y"],
            selected["w"],
            selected["h"],
            strict=True,
        )
    ]
    return selected.reset_index(drop=True)


def classifications_table(
    selected: pd.DataFrame, records: Mapping[str, ClassificationRecord], threshold: float
) -> pd.DataFrame:
    """Turn raw classifier output into one labeled row per selected box.

    Args:
        selected: Output of ``select_boxes``.
        records: Checkpoint records keyed by ``box_key``; missing keys become unresolved.
        threshold: Roll-up threshold.

    Returns:
        One row per selected box with the columns in ``CLASSIFICATION_COLUMNS``.
    """
    rows = []
    for pid, idx, key in zip(
        selected["photo_id"], selected["box_idx"], selected["box_key"], strict=True
    ):
        record = records.get(key)
        classes = list(record["classes"]) if record else []
        scores = [float(v) for v in record["scores"]] if record else []
        call = roll_up(classes, scores, threshold)
        rows.append(
            {
                "photo_id": pid,
                "box_idx": int(idx),
                "box_key": key,
                "species": call.species,
                "species_score": round(call.score, 4),
                "species_level": call.level,
                "lineage": call.lineage,
                "top1_label": parse_label(classes[0]).common_name if classes else None,
                "top1_score": round(scores[0], 4) if scores else None,
                "top5_json": json.dumps(
                    [
                        [parse_label(c).common_name, round(s, 4)]
                        for c, s in zip(classes, scores, strict=False)
                    ]
                ),
                "model": record["model"] if record else None,
                "error": record.get("error") if record else None,
            }
        )
    return pd.DataFrame(rows, columns=CLASSIFICATION_COLUMNS)


@dataclass(frozen=True)
class PhotoBoxes:
    """The boxes of one photo that still need a label.

    Attributes:
        photo_id: Photo id.
        keys: Checkpoint key of each box.
        bboxes: Normalized ``[x_min, y_min, width, height]`` of each box.
    """

    photo_id: str
    keys: list[str]
    bboxes: list[list[float]]


def _box_key_of(record: ClassificationRecord) -> str:
    return record["box_key"]


def group_by_photo(pending: pd.DataFrame) -> list[PhotoBoxes]:
    """Group the boxes that need a label by photo, so each image is decoded once.

    Args:
        pending: Rows of ``select_boxes`` output.

    Returns:
        One entry per photo, in first-seen order.
    """
    grouped: dict[str, PhotoBoxes] = {}
    for pid, key, x, y, w, h in zip(
        pending["photo_id"],
        pending["box_key"],
        pending["x"],
        pending["y"],
        pending["w"],
        pending["h"],
        strict=True,
    ):
        entry = grouped.setdefault(str(pid), PhotoBoxes(str(pid), [], []))
        entry.keys.append(str(key))
        entry.bboxes.append([float(x), float(y), float(w), float(h)])
    return list(grouped.values())


def run_classify(
    boxes: pd.DataFrame,
    media: pd.DataFrame,
    root: Path,
    classifier_factory: ClassifierFactory | None,
    checkpoint_path: Path,
    detection: DetectionSettings,
    classification: ClassificationSettings,
    workers: int = 4,
) -> pd.DataFrame:
    """Classify every selected box that is not in the checkpoint yet.

    Args:
        boxes: Box table from the detect stage.
        media: Media table (for file paths).
        root: Input folder.
        classifier_factory: Builds the classifier; None labels every box
            ``animal (unresolved)``.
        checkpoint_path: JSONL checkpoint of classification records.
        detection: Detection settings.
        classification: Classification settings.
        workers: Decoding threads.

    Returns:
        One row per selected box with the columns in ``CLASSIFICATION_COLUMNS``.
    """
    selected = select_boxes(boxes, detection, classification)
    if classifier_factory is None:
        logger.info("no classifier configured; %d box(es) left unresolved", len(selected))
        return classifications_table(selected, {}, classification.species_threshold)
    rel_of = {str(k): str(v) for k, v in zip(media["photo_id"], media["rel_path"], strict=True)}
    checkpoint: JsonlCheckpoint[ClassificationRecord]
    with JsonlCheckpoint(checkpoint_path, _box_key_of) as checkpoint:
        pending = selected[selected["box_key"].map(checkpoint.needs).astype(bool)]
        photos = group_by_photo(pending)
        logger.info(
            "%d box(es) in %d photo(s) to classify, %d already in checkpoint",
            len(pending),
            len(photos),
            len(checkpoint),
        )
        if photos:
            _classify_all(photos, rel_of, root, classifier_factory(), checkpoint, workers)
        records = dict(checkpoint.records)
    return classifications_table(selected, records, classification.species_threshold)


def _classify_all(
    photos: Iterable[PhotoBoxes],
    rel_of: Mapping[str, str],
    root: Path,
    classifier: Classifier,
    checkpoint: JsonlCheckpoint[ClassificationRecord],
    workers: int,
) -> None:
    images = ordered_map(
        lambda item: load_rgb(root / rel_of[item.photo_id]),
        photos,
        workers,
        catch=IMAGE_DECODE_ERRORS,
    )
    for photo, image in images:
        results: list[LabelScores]
        if isinstance(image, Exception):
            results = [_failed(type(image).__name__) for _ in photo.keys]
        else:
            try:
                results = classifier.classify(
                    image, photo.bboxes, path=str(root / rel_of[photo.photo_id])
                )
            except ImageFailedError:
                results = [_failed("ImageFailedError") for _ in photo.keys]
        for key, result in zip(photo.keys, results, strict=True):
            record: ClassificationRecord = {
                "box_key": key,
                "photo_id": photo.photo_id,
                "model": classifier.name,
                "classes": result["classes"],
                "scores": result["scores"],
            }
            if "error" in result:
                record["error"] = result["error"]
            checkpoint.append(record, flush=False)
        checkpoint.flush()


def _failed(error: str) -> LabelScores:
    return {"classes": [], "scores": [], "error": error}
