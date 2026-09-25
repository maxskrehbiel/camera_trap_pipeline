"""Quality stage: sharpness, exposure, infrared flag, difference hash and near-duplicates.

Flags are relative to each camera, so one scene's texture does not set the bar for another.
"""

from __future__ import annotations

import importlib
import logging
from collections.abc import Callable
from functools import cache
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from ._types import FloatArray, QualityMetrics, QualityRecord, UInt8Array
from .images import IMAGE_DECODE_ERRORS, load_rgb
from .parallel import ordered_map
from .settings import QualitySettings
from .status import JsonlCheckpoint

logger = logging.getLogger(__name__)

HASH_SIZE = 8

QUALITY_COLUMNS = [
    "photo_id",
    "decode_ok",
    "quality_error",
    "sharpness",
    "sharpness_rel",
    "brightness",
    "contrast",
    "ir_score",
    "dhash",
    "is_ir",
    "is_dark",
    "is_overexposed",
    "is_blurry",
    "dup_group",
    "is_near_duplicate",
]


def crop_strips(image: Image.Image, top: float, bottom: float) -> Image.Image:
    """Remove burned-in info bars from the top and bottom of a frame.

    Args:
        image: Frame to crop.
        top: Fraction of the height to drop from the top.
        bottom: Fraction of the height to drop from the bottom.

    Returns:
        The cropped frame, or the original if the crop would leave almost nothing.
    """
    first = round(image.height * top)
    last = image.height - round(image.height * bottom)
    if last - first < HASH_SIZE:
        return image
    return image.crop((0, first, image.width, last))


def shrink(image: Image.Image, max_side: int) -> Image.Image:
    """Shrink a frame so its longest side is at most ``max_side``.

    Args:
        image: Frame to shrink.
        max_side: Longest allowed side in pixels.

    Returns:
        A resized copy, or the original frame if it is already small enough.
    """
    if max(image.size) <= max_side:
        return image
    small = image.copy()
    small.thumbnail((max_side, max_side), Image.Resampling.BILINEAR)
    return small


def laplacian_variance_numpy(gray: FloatArray) -> float:
    """Variance of the 4-neighbour Laplacian, with mirrored borders.

    Sharp frames have strong local intensity changes, so the Laplacian has high variance;
    blur (motion, fog, a dirty lens) flattens it.

    Args:
        gray: 2-D grayscale array.

    Returns:
        The variance of the Laplacian over the frame.
    """
    padded = np.pad(gray, 1, mode="reflect")
    lap = (
        padded[:-2, 1:-1]
        + padded[2:, 1:-1]
        + padded[1:-1, :-2]
        + padded[1:-1, 2:]
        - 4.0 * padded[1:-1, 1:-1]
    )
    return float(lap.var())


@cache
def _opencv_laplacian() -> Callable[[FloatArray], float] | None:
    try:
        cv2 = importlib.import_module("cv2")
    except ImportError:
        return None

    def variance(gray: FloatArray) -> float:
        return float(cv2.Laplacian(gray, cv2.CV_64F, ksize=1).var())

    return variance


def laplacian_variance(gray: FloatArray) -> float:
    """Laplacian variance, using OpenCV when installed and NumPy otherwise.

    Both use the same 3x3 kernel and mirrored borders, so results agree to rounding.

    Args:
        gray: 2-D grayscale array.

    Returns:
        The variance of the Laplacian over the frame.
    """
    fast = _opencv_laplacian()
    return fast(gray) if fast is not None else laplacian_variance_numpy(gray)


def ir_score(rgb: UInt8Array) -> float:
    """Mean absolute difference between neighbouring color channels.

    Infrared night frames are monochrome, so the channels are nearly equal and the score
    is close to zero; daylight frames score well above it.

    Args:
        rgb: ``H x W x 3`` array.

    Returns:
        The mean channel spread in gray levels.
    """
    channels = rgb.astype(np.int16)
    red_green = np.abs(channels[..., 0] - channels[..., 1]).mean()
    green_blue = np.abs(channels[..., 1] - channels[..., 2]).mean()
    return float(red_green + green_blue)


def dhash(image: Image.Image, hash_size: int = HASH_SIZE) -> int:
    """64-bit difference hash: does brightness rise or fall between neighbouring pixels.

    The frame is shrunk to ``(hash_size + 1) x hash_size`` grayscale, so small changes
    (noise, a moving leaf, a new timestamp) flip few bits while a new scene flips many.

    Args:
        image: Frame to hash.
        hash_size: Rows of the hash grid; the hash has ``hash_size ** 2`` bits.

    Returns:
        The hash as an integer.
    """
    small = image.convert("L").resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)
    pixels = np.asarray(small, dtype=np.int16)
    bits = (pixels[:, 1:] > pixels[:, :-1]).flatten()
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return value


def hamming(a: int, b: int) -> int:
    """Number of differing bits between two hashes.

    Args:
        a: First hash.
        b: Second hash.

    Returns:
        The Hamming distance.
    """
    return (a ^ b).bit_count()


def measure_image(path: Path, settings: QualitySettings) -> QualityMetrics:
    """Compute the quality metrics of one photo.

    Args:
        path: Image file.
        settings: Quality settings (strip crop and working size).

    Returns:
        Sharpness, brightness, contrast, infrared score and difference hash (hex).
    """
    body = crop_strips(load_rgb(path), settings.strip_top, settings.strip_bottom)
    small = shrink(body, settings.max_side)
    gray = np.asarray(small.convert("L"), dtype=np.float64)
    return {
        "sharpness": round(laplacian_variance(gray), 3),
        "brightness": round(float(gray.mean()), 3),
        "contrast": round(float(gray.std()), 3),
        "ir_score": round(ir_score(np.asarray(small)), 3),
        "dhash": f"{dhash(small):016x}",
    }


def near_duplicate_groups(
    frame: pd.DataFrame, max_bits: int, window: int, max_gap_s: float
) -> pd.Series:
    """Assign each photo to a near-duplicate group within its camera.

    Photos are walked in time order. A photo joins the group of the closest earlier photo
    (among the previous ``window``, taken at most ``max_gap_s`` earlier) whose hash differs
    by at most ``max_bits``; otherwise it starts a new group named after itself.

    Args:
        frame: Needs ``photo_id``, ``camera_id``, ``timestamp``, ``rel_path`` and ``dhash``
            (hex string, or missing for undecodable photos).
        max_bits: Largest Hamming distance that counts as a near-duplicate.
        window: Number of earlier photos to compare against.
        max_gap_s: Largest time difference that still counts.

    Returns:
        Group id (a ``photo_id``) per row, aligned to ``frame.index``.
    """
    # Copy: on pandas without copy-on-write, writing into a view would rename the photos.
    groups = pd.Series(frame["photo_id"].to_numpy(copy=True), index=frame.index, dtype=object)
    hashed = frame[frame["dhash"].notna() & frame["timestamp"].notna()].sort_values(
        ["camera_id", "timestamp", "rel_path"]
    )
    for _, camera in hashed.groupby("camera_id", sort=False):
        values = [int(h, 16) for h in camera["dhash"]]
        seconds = (camera["timestamp"] - camera["timestamp"].iloc[0]).dt.total_seconds().tolist()
        assigned = list(camera["photo_id"])
        for i in range(len(values)):
            for j in range(i - 1, max(-1, i - window - 1), -1):
                if seconds[i] - seconds[j] > max_gap_s:
                    break
                if hamming(values[i], values[j]) <= max_bits:
                    assigned[i] = assigned[j]
                    break
        groups.loc[camera.index] = assigned
    return groups


def add_flags(frame: pd.DataFrame, settings: QualitySettings) -> pd.DataFrame:
    """Add boolean quality flags and near-duplicate groups to measured photos.

    Args:
        frame: Media rows joined with raw metrics; rows that failed to decode have
            missing metrics and get False flags.
        settings: Thresholds.

    Returns:
        A copy with ``is_ir``, ``is_dark``, ``is_overexposed``, ``sharpness_rel``,
        ``is_blurry``, ``dup_group`` and ``is_near_duplicate`` columns.
    """
    out = frame.copy()
    ok = out["decode_ok"]
    out["is_ir"] = ok & (out["ir_score"] < settings.ir_max_channel_spread)
    out["is_dark"] = ok & (out["brightness"] < settings.dark_brightness)
    out["is_overexposed"] = ok & (out["brightness"] > settings.overexposed_brightness)
    median = out[ok].groupby(["camera_id", "is_ir"])["sharpness"].transform("median")
    out["sharpness_rel"] = (out.loc[ok, "sharpness"] / median.where(median > 0)).round(4)
    out["is_blurry"] = ok & (out["sharpness_rel"] < settings.blur_ratio).fillna(False)
    out["dup_group"] = near_duplicate_groups(
        out, settings.duplicate_max_bits, settings.duplicate_window, settings.duplicate_max_gap_s
    )
    out["is_near_duplicate"] = ok & (out["dup_group"] != out["photo_id"])
    return out


def _quality_record(photo_id: str, result: QualityMetrics | Exception) -> QualityRecord:
    if isinstance(result, Exception):
        return {"photo_id": photo_id, "error": type(result).__name__}
    return {
        "photo_id": photo_id,
        "sharpness": result["sharpness"],
        "brightness": result["brightness"],
        "contrast": result["contrast"],
        "ir_score": result["ir_score"],
        "dhash": result["dhash"],
    }


def _photo_key(record: QualityRecord) -> str:
    return record["photo_id"]


def run_quality(
    media: pd.DataFrame,
    root: Path,
    checkpoint_path: Path,
    settings: QualitySettings,
    workers: int = 4,
) -> pd.DataFrame:
    """Measure every readable photo not yet in the checkpoint, then flag all photos.

    Args:
        media: Media table from the ingest stage.
        root: Input folder.
        checkpoint_path: JSONL checkpoint keyed by ``photo_id``.
        settings: Quality settings.
        workers: Decoding threads.

    Returns:
        One row per photo with the columns in ``QUALITY_COLUMNS``.
    """
    readable = media.loc[media["readable"].astype(bool)]
    checkpoint: JsonlCheckpoint[QualityRecord]
    with JsonlCheckpoint(checkpoint_path, _photo_key) as checkpoint:
        todo = [
            (pid, rel)
            for pid, rel in zip(readable["photo_id"], readable["rel_path"], strict=True)
            if checkpoint.needs(pid)
        ]
        logger.info("%d photo(s) to measure, %d already in checkpoint", len(todo), len(checkpoint))
        for (pid, _), result in ordered_map(
            lambda item: measure_image(root / item[1], settings),
            todo,
            workers,
            catch=IMAGE_DECODE_ERRORS,
        ):
            checkpoint.append(_quality_record(pid, result))
        records = [checkpoint.records[p] for p in media["photo_id"] if p in checkpoint]
    metrics = pd.DataFrame(
        records, columns=["photo_id", "sharpness", "brightness", "contrast", "ir_score", "dhash"]
    )
    errors = {r["photo_id"]: r.get("error") for r in records if "error" in r}
    frame = media[["photo_id", "rel_path", "camera_id", "timestamp"]].merge(
        metrics, on="photo_id", how="left"
    )
    frame["decode_ok"] = frame["sharpness"].notna()
    readable_ids = set(readable["photo_id"])
    frame["quality_error"] = [
        errors.get(pid) if pid in readable_ids else "unreadable" for pid in frame["photo_id"]
    ]
    flagged = add_flags(frame, settings)
    return flagged[QUALITY_COLUMNS].reset_index(drop=True)
