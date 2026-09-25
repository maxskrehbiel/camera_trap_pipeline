"""Ingest stage: list the photos under a folder and resolve each one's camera and timestamp.

Sources in priority order: ``metadata.csv``, the camera folder, then EXIF.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path, PurePosixPath
from typing import TypedDict

import pandas as pd
from PIL import Image

from .errors import InputError
from .images import IMAGE_DECODE_ERRORS
from .parallel import ordered_map

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".tif", ".tiff"})
SIDECAR_NAME = "metadata.csv"

TAG_MAKE = 0x010F
TAG_MODEL = 0x0110
TAG_DATETIME = 0x0132
TAG_EXIF_IFD = 0x8769
TAG_DATETIME_ORIGINAL = 0x9003
TAG_BODY_SERIAL = 0xA431

_TIMESTAMP_FORMATS = (
    "%Y:%m:%d %H:%M:%S",
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%dT%H:%M:%S",
    "%Y/%m/%d %H:%M:%S",
)
_FUTURE_TOLERANCE = timedelta(days=1)

MEDIA_COLUMNS = [
    "photo_id",
    "rel_path",
    "camera_id",
    "camera_source",
    "timestamp",
    "timestamp_source",
    "timestamp_valid",
    "width",
    "height",
    "file_bytes",
    "exif_make",
    "exif_model",
    "readable",
    "error",
]


class HeaderInfo(TypedDict, total=False):
    """Fields read from an image header; only ``readable`` is always present."""

    readable: bool
    width: int
    height: int
    datetime_original: str | None
    datetime: str | None
    make: str | None
    model: str | None
    serial: str | None
    error: str | None


@dataclass(frozen=True)
class SidecarRow:
    """Per-file overrides read from ``metadata.csv``.

    Attributes:
        camera_id: Camera identifier, or None to keep the default resolution.
        timestamp: Capture time, or None to fall back to EXIF.
    """

    camera_id: str | None
    timestamp: datetime | None


def photo_id_for(rel_path: str) -> str:
    """Derive a stable id from a photo's path relative to the input folder.

    Args:
        rel_path: POSIX-style relative path.

    Returns:
        16 hexadecimal characters.
    """
    return hashlib.blake2b(rel_path.encode("utf-8"), digest_size=8).hexdigest()


def find_images(root: Path) -> list[str]:
    """List the image files under a folder, skipping hidden files and folders.

    Args:
        root: Input folder.

    Returns:
        Sorted POSIX paths relative to ``root``.
    """
    found = []
    for path in root.rglob("*"):
        rel = path.relative_to(root)
        if any(part.startswith(".") for part in rel.parts):
            continue
        if path.suffix.lower() in IMAGE_EXTENSIONS and path.is_file():
            found.append(rel.as_posix())
    return sorted(found)


def _clean(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, bytes):
        value = value.decode("utf-8", "ignore")
    text = str(value).replace("\x00", "").strip()
    return text or None


def parse_timestamp(value: object) -> datetime | None:
    """Parse an EXIF or ISO-like timestamp into a naive local datetime.

    Time-zone offsets are dropped: camera clocks record local wall-clock time and the
    pipeline keeps it that way, since hour-of-day activity is the quantity of interest.

    Args:
        value: String, bytes or None.

    Returns:
        The parsed datetime, or None for empty or unparseable input (including the
        ``0000:00:00 00:00:00`` placeholder some cameras write).
    """
    text = _clean(value)
    if text is None or text.startswith("0000"):
        return None
    for fmt in _TIMESTAMP_FORMATS:
        try:
            return datetime.strptime(text[:19], fmt)
        except ValueError:
            continue
    try:
        parsed = pd.Timestamp(text)
    except (ValueError, TypeError):
        return None
    if pd.isna(parsed):
        return None
    if parsed.tzinfo is not None:
        parsed = parsed.tz_localize(None)
    result: datetime = parsed.to_pydatetime()
    return result


def load_sidecar(root: Path) -> dict[str, SidecarRow]:
    """Read ``metadata.csv`` from the input folder if present.

    The file needs a ``file`` column (path relative to the input folder) and may have
    ``camera_id`` and ``timestamp`` columns. It is the way to bring in public datasets whose
    capture times live in a metadata file rather than in EXIF.

    Args:
        root: Input folder.

    Returns:
        Overrides keyed by POSIX relative path; empty when there is no sidecar.

    Raises:
        InputError: If the sidecar has no ``file`` column.
    """
    path = root / SIDECAR_NAME
    if not path.exists():
        return {}
    table = pd.read_csv(path, dtype=str, keep_default_na=False)
    if "file" not in table.columns:
        raise InputError(f"{SIDECAR_NAME} needs a 'file' column")
    rows: dict[str, SidecarRow] = {}
    for record in table.to_dict("records"):
        key = PurePosixPath(str(record["file"]).replace("\\", "/")).as_posix()
        rows[key] = SidecarRow(
            camera_id=_clean(record.get("camera_id")),
            timestamp=parse_timestamp(record.get("timestamp")),
        )
    return rows


def read_header(path: Path) -> HeaderInfo:
    """Read size and EXIF fields without decoding pixel data.

    Args:
        path: Image file.

    Returns:
        A dict with ``readable`` and, when readable, size and EXIF fields; otherwise the
        error type and message.
    """
    try:
        with Image.open(path) as image:
            width, height = image.size
            exif = image.getexif()
            sub = exif.get_ifd(TAG_EXIF_IFD)
            return {
                "readable": True,
                "width": width,
                "height": height,
                "datetime_original": _clean(sub.get(TAG_DATETIME_ORIGINAL)),
                "datetime": _clean(exif.get(TAG_DATETIME)),
                "make": _clean(exif.get(TAG_MAKE)),
                "model": _clean(exif.get(TAG_MODEL)),
                "serial": _clean(sub.get(TAG_BODY_SERIAL)),
                "error": None,
            }
    except IMAGE_DECODE_ERRORS as exc:
        return {"readable": False, "error": type(exc).__name__}


def _camera_for(rel: PurePosixPath, header: HeaderInfo, side: SidecarRow | None) -> tuple[str, str]:
    if side is not None and side.camera_id:
        return side.camera_id, "sidecar"
    if len(rel.parts) > 1:
        return rel.parts[0], "folder"
    if header.get("serial"):
        return str(header["serial"]), "exif_serial"
    make_model = " ".join(p for p in (header.get("make"), header.get("model")) if p)
    if make_model:
        return make_model, "exif_make_model"
    return "unknown", "none"


def _timestamp_for(header: HeaderInfo, side: SidecarRow | None) -> tuple[datetime | None, str]:
    if side is not None and side.timestamp is not None:
        return side.timestamp, "sidecar"
    original = parse_timestamp(header.get("datetime_original"))
    if original is not None:
        return original, "exif_original"
    fallback = parse_timestamp(header.get("datetime"))
    if fallback is not None:
        return fallback, "exif_datetime"
    return None, "missing"


def ingest(
    root: Path, valid_from: str, workers: int = 4, now: datetime | None = None
) -> pd.DataFrame:
    """Build the media table: one row per image file under ``root``.

    Args:
        root: Input folder.
        valid_from: Timestamps on or before this date are flagged invalid (unset clock).
        workers: Threads used to read file headers.
        now: Reference time for the future-timestamp check; defaults to the current time.

    Returns:
        A DataFrame with the columns in ``MEDIA_COLUMNS``, sorted by camera, time and path.

    Raises:
        InputError: If the folder holds no images or its sidecar is malformed.
    """
    sidecar = load_sidecar(root)
    rel_paths = find_images(root)
    if not rel_paths:
        raise InputError(f"no images found under {root}")
    floor = pd.Timestamp(valid_from).to_pydatetime()
    ceiling = (now or datetime.now()) + _FUTURE_TOLERANCE
    rows: list[dict[str, object]] = []
    for rel_path, result in ordered_map(lambda p: read_header(root / p), rel_paths, workers):
        header: HeaderInfo = (
            {"readable": False, "error": type(result).__name__}
            if isinstance(result, Exception)
            else result
        )
        timestamp, timestamp_source = _timestamp_for(header, sidecar.get(rel_path))
        camera_id, camera_source = _camera_for(
            PurePosixPath(rel_path), header, sidecar.get(rel_path)
        )
        rows.append(
            {
                "photo_id": photo_id_for(rel_path),
                "rel_path": rel_path,
                "camera_id": camera_id,
                "camera_source": camera_source,
                "timestamp": timestamp,
                "timestamp_source": timestamp_source,
                "timestamp_valid": timestamp is not None and floor < timestamp <= ceiling,
                "width": header.get("width"),
                "height": header.get("height"),
                "file_bytes": (root / rel_path).stat().st_size,
                "exif_make": header.get("make"),
                "exif_model": header.get("model"),
                "readable": bool(header.get("readable")),
                "error": header.get("error"),
            }
        )
    return _media_frame(rows)


def _media_frame(rows: list[dict[str, object]]) -> pd.DataFrame:
    media = pd.DataFrame(rows, columns=MEDIA_COLUMNS)
    media["timestamp"] = pd.to_datetime(media["timestamp"])
    media["width"] = media["width"].astype("Int64")
    media["height"] = media["height"].astype("Int64")
    media = media.sort_values(["camera_id", "timestamp", "rel_path"], na_position="last")
    logger.info(
        "found %d image(s) from %d camera(s); %d unreadable, %d with invalid timestamps",
        len(media),
        media["camera_id"].nunique(),
        int((~media["readable"]).sum()),
        int((~media["timestamp_valid"]).sum()),
    )
    return media.reset_index(drop=True)


def find_new_images(root: Path, media: pd.DataFrame) -> list[str]:
    """Find images added to the input folder after an ingest.

    Args:
        root: Input folder.
        media: Media table from a previous ingest.

    Returns:
        Relative paths that are not in ``media``.
    """
    known = set(media["rel_path"])
    return [p for p in find_images(root) if p not in known]
