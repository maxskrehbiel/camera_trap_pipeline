"""Shared type aliases for arrays and for the JSON records kept in checkpoints."""

from __future__ import annotations

from typing import NotRequired, TypedDict

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
Float32Array = NDArray[np.float32]
UInt8Array = NDArray[np.uint8]


class Box(TypedDict):
    """One detector box: category, confidence and normalized ``[x_min, y_min, w, h]``."""

    category: str
    conf: float
    bbox: list[float]


class DetectionRecord(TypedDict):
    """Detection checkpoint line for one photo."""

    photo_id: str
    model: str
    boxes: list[Box]
    error: NotRequired[str]


class LabelScores(TypedDict):
    """Classifier output for one box: labels and scores, most likely first."""

    classes: list[str]
    scores: list[float]
    error: NotRequired[str]


class ClassificationRecord(TypedDict):
    """Classification checkpoint line for one box."""

    box_key: str
    photo_id: str
    model: str
    classes: list[str]
    scores: list[float]
    error: NotRequired[str]


class QualityMetrics(TypedDict):
    """Quality measurements of one decoded photo."""

    sharpness: float
    brightness: float
    contrast: float
    ir_score: float
    dhash: str


class QualityRecord(TypedDict):
    """Quality checkpoint line for one photo: metrics, or the reason decoding failed."""

    photo_id: str
    sharpness: NotRequired[float]
    brightness: NotRequired[float]
    contrast: NotRequired[float]
    ir_score: NotRequired[float]
    dhash: NotRequired[str]
    error: NotRequired[str]
