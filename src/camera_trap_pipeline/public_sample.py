"""Download a small sample of a public COCO Camera Traps dataset (default: LILA BC).

Images go to a local folder at run time and capture times to ``metadata.csv``.
"""

from __future__ import annotations

import csv
import io
import json
import logging
import re
import zipfile
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import IO, Any
from urllib.error import URLError
from urllib.parse import quote, urlparse
from urllib.request import Request, urlopen

import numpy as np

from .atomic_io import atomic_write_bytes, atomic_write_text
from .errors import InputError
from .ingest import IMAGE_EXTENSIONS, SIDECAR_NAME

logger = logging.getLogger(__name__)

LILA_CCT_METADATA_URL = (
    "https://storage.googleapis.com/public-datasets-lila/caltechcameratraps/labels/"
    "caltech_bboxes_20200316.json"
)
LILA_CCT_IMAGE_BASE_URL = (
    "https://storage.googleapis.com/public-datasets-lila/caltech-unzipped/cct_images/"
)
LILA_CCT_ATTRIBUTION = (
    "Images: Caltech Camera Traps, distributed by LILA BC (https://lila.science).\n"
    "License: Community Data License Agreement - Permissive, Version 1.0.\n"
    "Citation: Beery, S., Van Horn, G., Perona, P. Recognition in Terra Incognita. "
    "ECCV 2018.\n"
)
USER_AGENT = "camera_trap_pipeline (public sample fetch)"
MAX_METADATA_BYTES = 1_000_000_000
MAX_IMAGE_BYTES = 50_000_000
DOWNLOAD_CHUNK_BYTES = 1 << 20
_SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_TIME_FIELDS = ("datetime", "date_captured")
_CAMERA_FIELDS = ("location", "camera", "site")

Opener = Callable[..., IO[bytes]]


@dataclass(frozen=True)
class SampleSettings:
    """How much to download.

    Attributes:
        n_cameras: Number of camera locations to sample.
        photos_per_camera: Photos per location, taken as whole sequences in time order.
        timeout_s: Network timeout per request.
    """

    n_cameras: int = 3
    photos_per_camera: int = 60
    timeout_s: float = 60.0


@dataclass(frozen=True)
class SampleImage:
    """One image entry from the metadata.

    Attributes:
        file_name: Path of the image within the dataset.
        camera_id: Camera (location) id.
        timestamp: Capture time string, as given.
        seq_id: Sequence id, if the dataset groups bursts.
    """

    file_name: str
    camera_id: str
    timestamp: str | None
    seq_id: str | None


def require_https(url: str) -> str:
    """Check that a URL uses HTTPS.

    Args:
        url: URL to check.

    Returns:
        The URL unchanged.

    Raises:
        InputError: For any other scheme.
    """
    if urlparse(url).scheme != "https":
        raise InputError(f"only https URLs are allowed, got {url!r}")
    return url


def safe_file_name(name: str) -> str:
    """Return the base name of a dataset path after checking it is safe to write locally.

    Metadata comes from a remote file, so names are validated before they touch the disk:
    no absolute paths, no ``..``, no backslashes, image extensions only.

    Args:
        name: Dataset-relative path.

    Returns:
        The file's base name.

    Raises:
        InputError: If the name is unsafe or not an image.
    """
    if "\\" in name or name.startswith("/"):
        raise InputError(f"unsafe file name {name!r}")
    # Check the raw segments: PurePosixPath would silently collapse "a/./b" and "a//b".
    if any(part in ("..", ".", "") for part in name.split("/")):
        raise InputError(f"unsafe file name {name!r}")
    path = PurePosixPath(name)
    base = path.name
    if not _SAFE_NAME.match(base) or path.suffix.lower() not in IMAGE_EXTENSIONS:
        raise InputError(f"unsupported file name {name!r}")
    return base


def camera_folder(camera_id: str) -> str:
    """Turn a dataset location id into a snake_case folder name.

    Args:
        camera_id: Location id from the metadata.

    Returns:
        ``location_<id>`` with anything but letters and digits replaced by ``_``.
    """
    cleaned = re.sub(r"[^a-z0-9]+", "_", camera_id.lower()).strip("_")
    return f"location_{cleaned or 'unknown'}"


def parse_metadata(raw: bytes) -> list[SampleImage]:
    """Read image entries from COCO Camera Traps JSON (optionally inside a zip).

    Args:
        raw: File contents.

    Returns:
        Entries that have a file name and a camera; entries without either are dropped.

    Raises:
        InputError: If the file is not JSON, or a zip without a JSON member.
    """
    try:
        if raw[:2] == b"PK":
            with zipfile.ZipFile(io.BytesIO(raw)) as archive:
                member = next(n for n in archive.namelist() if n.lower().endswith(".json"))
                raw = archive.read(member)
        data: dict[str, Any] = json.loads(raw)
    except (StopIteration, zipfile.BadZipFile, ValueError) as exc:
        raise InputError(f"metadata is not COCO Camera Traps JSON: {exc}") from exc
    entries = []
    for image in data.get("images", []):
        camera = next((image[f] for f in _CAMERA_FIELDS if image.get(f) not in (None, "")), None)
        if not image.get("file_name") or camera is None:
            continue
        stamp = next((str(image[f]) for f in _TIME_FIELDS if image.get(f)), None)
        seq = image.get("seq_id")
        entries.append(
            SampleImage(str(image["file_name"]), str(camera), stamp, str(seq) if seq else None)
        )
    return entries


def choose_sample(
    entries: list[SampleImage], settings: SampleSettings, rng: np.random.Generator
) -> list[SampleImage]:
    """Pick cameras and a contiguous run of photos from each.

    Whole sequences are kept together so bursts survive and the events stage has real
    bursts to group.

    Args:
        entries: Output of ``parse_metadata``.
        settings: Sample size.
        rng: Random generator for choosing cameras and starting points.

    Returns:
        The chosen entries, grouped by camera and in time order within each.

    Raises:
        InputError: If no entry has a capture time.
    """
    by_camera: dict[str, list[SampleImage]] = defaultdict(list)
    for entry in entries:
        if entry.timestamp:
            by_camera[entry.camera_id].append(entry)
    eligible = sorted(
        c for c, items in by_camera.items() if len(items) >= settings.photos_per_camera
    )
    if not by_camera:
        raise InputError("the metadata lists no images with both a camera and a capture time")
    if not eligible:
        eligible = sorted(by_camera)
    count = min(settings.n_cameras, len(eligible))
    chosen = sorted(rng.choice(eligible, size=count, replace=False).tolist()) if count else []
    sample: list[SampleImage] = []
    for camera in chosen:
        items = sorted(by_camera[camera], key=lambda e: (e.timestamp or "", e.file_name))
        start = int(rng.integers(0, max(1, len(items) - settings.photos_per_camera + 1)))
        window = items[start : start + settings.photos_per_camera]
        if window and window[-1].seq_id:
            tail = [e for e in items[start + len(window) :] if e.seq_id == window[-1].seq_id]
            window += tail
        sample += window
    return sample


def _download(url: str, limit: int, timeout_s: float, opener: Opener) -> bytes:
    request = Request(require_https(url), headers={"User-Agent": USER_AGENT})
    buffer = bytearray()
    try:
        with opener(request, timeout=timeout_s) as response:
            while chunk := response.read(DOWNLOAD_CHUNK_BYTES):
                buffer += chunk
                if len(buffer) > limit:
                    raise InputError(f"download from {url} exceeded {limit:,} bytes")
    except URLError as exc:
        raise InputError(f"download from {url} failed: {exc.reason}") from exc
    return bytes(buffer)


def _download_images(
    sample: list[SampleImage],
    out_dir: Path,
    image_base_url: str,
    settings: SampleSettings,
    opener: Opener,
) -> list[dict[str, str | None]]:
    """Fetch the chosen images that are not on disk yet; return their sidecar rows."""
    base = image_base_url if image_base_url.endswith("/") else image_base_url + "/"
    rows: list[dict[str, str | None]] = []
    for entry in sample:
        try:
            name = safe_file_name(entry.file_name)
        except InputError as exc:
            logger.warning("skipping %s", exc)
            continue
        folder = camera_folder(entry.camera_id)
        target = out_dir / folder / name
        if not target.exists():
            url = base + quote(entry.file_name)
            atomic_write_bytes(target, _download(url, MAX_IMAGE_BYTES, settings.timeout_s, opener))
        rows.append({"file": f"{folder}/{name}", "camera_id": folder, "timestamp": entry.timestamp})
    return rows


def _write_sidecar(out_dir: Path, rows: list[dict[str, str | None]], metadata_url: str) -> None:
    """Write ``metadata.csv`` for the ingest stage and ``attribution.txt`` for the license."""
    buffer = io.StringIO()
    writer = csv.DictWriter(
        buffer, fieldnames=["file", "camera_id", "timestamp"], lineterminator="\n"
    )
    writer.writeheader()
    writer.writerows(rows)
    atomic_write_text(out_dir / SIDECAR_NAME, buffer.getvalue())
    attribution = (
        LILA_CCT_ATTRIBUTION
        if metadata_url == LILA_CCT_METADATA_URL
        else f"Source: {metadata_url}\n"
    )
    atomic_write_text(out_dir / "attribution.txt", attribution)


def fetch_public_sample(
    out_dir: Path,
    settings: SampleSettings,
    rng: np.random.Generator,
    metadata_url: str = LILA_CCT_METADATA_URL,
    image_base_url: str = LILA_CCT_IMAGE_BASE_URL,
    opener: Opener = urlopen,
) -> Path:
    """Download a small sample into ``out_dir/<location>/`` with a ``metadata.csv`` sidecar.

    The metadata file is cached in ``out_dir/.cache`` so repeated calls only fetch images.
    Images already on disk are skipped.

    Args:
        out_dir: Destination folder.
        settings: Sample size and timeout.
        rng: Random generator.
        metadata_url: HTTPS URL of COCO Camera Traps JSON (or zipped JSON).
        image_base_url: HTTPS URL that image file names are appended to.
        opener: ``urllib.request.urlopen`` or a compatible callable (for tests).

    Returns:
        ``out_dir``.

    Raises:
        InputError: For a non-HTTPS URL, a failed or oversized download, or unusable
            metadata.
    """
    require_https(metadata_url)
    require_https(image_base_url)
    cache = out_dir / ".cache" / "metadata.bin"
    if not cache.exists():
        logger.info("downloading metadata from %s", metadata_url)
        metadata = _download(metadata_url, MAX_METADATA_BYTES, settings.timeout_s, opener)
        atomic_write_bytes(cache, metadata)
    sample = choose_sample(parse_metadata(cache.read_bytes()), settings, rng)
    rows = _download_images(sample, out_dir, image_base_url, settings, opener)
    _write_sidecar(out_dir, rows, metadata_url)
    logger.info("wrote %d image(s) to %s", len(rows), out_dir.name)
    return out_dir
