"""Detect stage: animal, person and vehicle boxes per photo, checkpointed photo by photo.

Boxes are normalized ``[x_min, y_min, width, height]``, MegaDetector's convention.
"""

from __future__ import annotations

import importlib
import logging
import time
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Protocol

import pandas as pd
from PIL import Image

from ._types import Box, DetectionRecord
from .errors import ImageFailedError, ModelDependencyError
from .images import IMAGE_DECODE_ERRORS, load_rgb
from .parallel import ordered_map
from .status import JsonlCheckpoint

logger = logging.getLogger(__name__)

MEGADETECTOR_CATEGORIES = {"1": "animal", "2": "person", "3": "vehicle"}
DEFAULT_DETECTION_THRESHOLD = 0.1
DEFAULT_MEGADETECTOR_MODEL = "MDV5A"
PROGRESS_EVERY_S = 30.0

BOX_COLUMNS = ["photo_id", "box_idx", "category", "conf", "x", "y", "w", "h", "model"]
RUN_COLUMNS = ["photo_id", "model", "n_boxes", "error"]


class Detector(Protocol):
    """Anything that turns an image into boxes.

    Attributes:
        name: Model identifier stored with every result.
    """

    name: str

    def detect(self, image: Image.Image, *, path: str) -> list[Box]:
        """Find objects in one image.

        Args:
            image: RGB image.
            path: Source file, for detectors that key results by path.

        Returns:
            Boxes with ``category``, ``conf`` and normalized ``bbox``.

        Raises:
            ImageFailedError: If the model cannot process this particular image.
        """
        ...


DetectorFactory = Callable[[], Detector]


class MegaDetector:
    """MegaDetector through the ``megadetector`` package (weights download on first use).

    Uses the package's ``load_detector`` and ``generate_detections_one_image``, so the
    output matches the reference implementation. Runs on a GPU when PyTorch can see one.
    """

    def __init__(
        self,
        model: str = DEFAULT_MEGADETECTOR_MODEL,
        threshold: float = DEFAULT_DETECTION_THRESHOLD,
        force_cpu: bool = False,
    ) -> None:
        """Load the model.

        Args:
            model: A model name ``load_detector`` knows (such as ``MDV5A``) or a weights file.
            threshold: Boxes below this confidence are not returned.
            force_cpu: Run on the CPU even if a GPU is available.

        Raises:
            ModelDependencyError: If the ``megadetector`` package is not installed.
        """
        try:
            run_detector = importlib.import_module("megadetector.detection.run_detector")
        except ImportError as exc:
            raise ModelDependencyError(
                "MegaDetector is optional: pip install -e '.[models]'"
            ) from exc
        self._model = run_detector.load_detector(model, force_cpu=force_cpu)
        self.threshold = threshold
        self.name = f"megadetector:{model}"

    def detect(self, image: Image.Image, *, path: str) -> list[Box]:
        """Run the detector on one image.

        Args:
            image: RGB image.
            path: Source file (used as the image id in the model's result).

        Returns:
            Boxes with categories mapped to ``animal``, ``person`` and ``vehicle``.

        Raises:
            ImageFailedError: If the model reports a failure for this image.
        """
        result = self._model.generate_detections_one_image(
            image, path, detection_threshold=self.threshold
        )
        if result.get("failure"):
            raise ImageFailedError(str(result["failure"]))
        return [
            {
                "category": MEGADETECTOR_CATEGORIES.get(str(d["category"]), "unknown"),
                "conf": float(d["conf"]),
                "bbox": [float(v) for v in d["bbox"]],
            }
            for d in result.get("detections") or []
        ]


def clean_boxes(raw: Iterable[Box]) -> list[Box]:
    """Clip boxes to the frame, round them and sort them by confidence.

    Args:
        raw: Boxes as returned by a detector.

    Returns:
        New boxes, most confident first.
    """
    boxes: list[Box] = []
    for box in raw:
        x, y, w, h = (min(max(float(v), 0.0), 1.0) for v in box["bbox"])
        boxes.append(
            {
                "category": str(box["category"]),
                "conf": round(float(box["conf"]), 4),
                "bbox": [
                    round(x, 4),
                    round(y, 4),
                    round(min(w, 1.0 - x), 4),
                    round(min(h, 1.0 - y), 4),
                ],
            }
        )
    boxes.sort(key=lambda b: -b["conf"])
    return boxes


def boxes_table(records: Iterable[DetectionRecord]) -> pd.DataFrame:
    """Flatten detection records into a table.

    Args:
        records: Detection checkpoint records.

    Returns:
        One row per box with the columns in ``BOX_COLUMNS``.
    """
    rows = [
        {
            "photo_id": record["photo_id"],
            "box_idx": idx,
            "category": box["category"],
            "conf": box["conf"],
            "x": box["bbox"][0],
            "y": box["bbox"][1],
            "w": box["bbox"][2],
            "h": box["bbox"][3],
            "model": record["model"],
        }
        for record in records
        for idx, box in enumerate(record["boxes"])
    ]
    return pd.DataFrame(rows, columns=BOX_COLUMNS)


def runs_table(records: Iterable[DetectionRecord]) -> pd.DataFrame:
    """Summarize detection records per photo.

    Args:
        records: Detection checkpoint records.

    Returns:
        One row per processed photo with the columns in ``RUN_COLUMNS``.
    """
    rows = [
        {
            "photo_id": r["photo_id"],
            "model": r["model"],
            "n_boxes": len(r["boxes"]),
            "error": r.get("error"),
        }
        for r in records
    ]
    return pd.DataFrame(rows, columns=RUN_COLUMNS)


def _photo_key(record: DetectionRecord) -> str:
    return record["photo_id"]


def run_detect(
    media: pd.DataFrame,
    quality: pd.DataFrame,
    root: Path,
    detector_factory: DetectorFactory,
    checkpoint_path: Path,
    workers: int = 4,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Detect on every decodable photo that is not in the checkpoint yet.

    A photo that cannot be decoded, or that the model reports it cannot process, is
    recorded with its error and retried on the next run. Any other exception stops the
    stage; everything detected so far is already in the checkpoint.

    Args:
        media: Media table.
        quality: Quality table (``decode_ok`` selects the photos to process).
        root: Input folder.
        detector_factory: Builds the detector; called only if there is work to do.
        checkpoint_path: JSONL checkpoint of detection records.
        workers: Decoding threads.

    Returns:
        ``(boxes, runs)``: one row per box, and one row per processed photo.
    """
    decodable = set(quality.loc[quality["decode_ok"].astype(bool), "photo_id"])
    eligible = media[media["photo_id"].isin(decodable)]
    checkpoint: JsonlCheckpoint[DetectionRecord]
    with JsonlCheckpoint(checkpoint_path, _photo_key) as checkpoint:
        todo = [
            (str(pid), str(rel))
            for pid, rel in zip(eligible["photo_id"], eligible["rel_path"], strict=True)
            if checkpoint.needs(str(pid))
        ]
        logger.info("%d photo(s) to detect, %d already in checkpoint", len(todo), len(checkpoint))
        if todo:
            _detect_all(todo, root, detector_factory(), checkpoint, workers)
        records = [checkpoint.records[p] for p in eligible["photo_id"] if p in checkpoint]
    return boxes_table(records), runs_table(records)


def _detect_all(
    todo: list[tuple[str, str]],
    root: Path,
    detector: Detector,
    checkpoint: JsonlCheckpoint[DetectionRecord],
    workers: int,
) -> None:
    started = last_report = time.monotonic()
    images = ordered_map(
        lambda item: load_rgb(root / item[1]), todo, workers, catch=IMAGE_DECODE_ERRORS
    )
    for done, ((pid, rel), image) in enumerate(images, start=1):
        record: DetectionRecord = {"photo_id": pid, "model": detector.name, "boxes": []}
        if isinstance(image, Exception):
            record["error"] = type(image).__name__
        else:
            try:
                record["boxes"] = clean_boxes(detector.detect(image, path=str(root / rel)))
            except ImageFailedError as exc:
                # Keep the input folder's absolute path out of stored outputs.
                record["error"] = f"ImageFailedError: {exc}".replace(str(root), "<input>")
        checkpoint.append(record)
        now = time.monotonic()
        if now - last_report >= PROGRESS_EVERY_S or done == len(todo):
            logger.info("detected %d/%d (%.2f s/photo)", done, len(todo), (now - started) / done)
            last_report = now
