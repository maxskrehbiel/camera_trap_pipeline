from __future__ import annotations

import csv
import io
import json
import zipfile
from collections.abc import Callable
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.parse import quote
from urllib.request import Request

import numpy as np
import pytest

from camera_trap_pipeline import public_sample
from camera_trap_pipeline.errors import InputError
from camera_trap_pipeline.public_sample import (
    LILA_CCT_IMAGE_BASE_URL,
    LILA_CCT_METADATA_URL,
    SampleImage,
    SampleSettings,
    camera_folder,
    choose_sample,
    fetch_public_sample,
    parse_metadata,
    require_https,
    safe_file_name,
)

IMAGES = [
    {"file_name": "a1.jpg", "location": "46", "datetime": "2013-10-04 13:31:53", "seq_id": "s1"},
    {"file_name": "a2.jpg", "location": "46", "datetime": "2013-10-04 13:31:54", "seq_id": "s1"},
    {"file_name": "../evil.jpg", "location": "46", "datetime": "2013-10-04 13:31:55"},
    {
        "file_name": "b1.jpg",
        "location": "7",
        "date_captured": "2012-01-01 01:00:00",
        "seq_id": "t1",
    },
    {
        "file_name": "b2.jpg",
        "location": "7",
        "date_captured": "2012-01-01 01:00:01",
        "seq_id": "t1",
    },
    {
        "file_name": "b3.jpg",
        "location": "7",
        "date_captured": "2012-01-01 01:00:02",
        "seq_id": "t1",
    },
    {
        "file_name": "b4.jpg",
        "location": "7",
        "date_captured": "2012-01-01 01:00:03",
        "seq_id": "t1",
    },
    {"file_name": "no_camera.jpg", "datetime": "2012-01-01 01:00:00"},
]


def metadata_bytes() -> bytes:
    return json.dumps({"images": IMAGES}).encode("utf-8")


def fake_opener(files: dict[str, bytes]) -> tuple[Callable[..., Any], list[str]]:
    calls: list[str] = []

    def opener(request: Request, timeout: float) -> io.BytesIO:
        calls.append(request.full_url)
        assert request.get_header("User-agent", "").startswith("camera_trap_pipeline")
        return io.BytesIO(files[request.full_url])

    return opener, calls


def test_parse_metadata_reads_json_and_zip() -> None:
    entries = parse_metadata(metadata_bytes())
    assert len(entries) == 7
    assert entries[0] == SampleImage("a1.jpg", "46", "2013-10-04 13:31:53", "s1")
    assert entries[3].timestamp == "2012-01-01 01:00:00"

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("readme.txt", "x")
        archive.writestr("labels/meta.json", metadata_bytes())
    assert parse_metadata(buffer.getvalue()) == entries


@pytest.mark.parametrize(
    "name",
    ["../evil.jpg", "/abs.jpg", "a\\b.jpg", "a/./b.jpg", "run.exe", ".hidden.jpg", "a//b.jpg"],
)
def test_unsafe_names_are_rejected(name: str) -> None:
    with pytest.raises(ValueError, match=r"unsafe|unsupported"):
        safe_file_name(name)


def test_safe_names_keep_only_the_base_name() -> None:
    assert safe_file_name("cct_images/5968c0f9-23d2.jpg") == "5968c0f9-23d2.jpg"
    assert camera_folder("Loc 46/B") == "location_loc_46_b"
    assert camera_folder("!!") == "location_unknown"


def test_only_https_is_allowed() -> None:
    assert require_https("https://example.org/x") == "https://example.org/x"
    with pytest.raises(ValueError, match="https"):
        require_https("http://example.org/x")
    with pytest.raises(ValueError, match="https"):
        require_https("file:///etc/passwd")


def test_choose_sample_keeps_whole_sequences() -> None:
    entries = parse_metadata(metadata_bytes())
    sample = choose_sample(
        entries, SampleSettings(n_cameras=2, photos_per_camera=2), np.random.default_rng(0)
    )
    by_camera: dict[str, list[str]] = {}
    for entry in sample:
        by_camera.setdefault(entry.camera_id, []).append(entry.file_name)
    assert set(by_camera) == {"46", "7"}
    assert by_camera["7"] == sorted(by_camera["7"])
    assert by_camera["7"][-1] == "b4.jpg", "the open sequence is completed"


def test_fetch_writes_images_sidecar_and_attribution(tmp_path: Path) -> None:
    files = {LILA_CCT_METADATA_URL: metadata_bytes()}
    for image in IMAGES:
        files[LILA_CCT_IMAGE_BASE_URL + quote(image["file_name"])] = b"jpeg-bytes"
    opener, calls = fake_opener(files)
    settings = SampleSettings(n_cameras=2, photos_per_camera=3)
    fetch_public_sample(tmp_path, settings, np.random.default_rng(1), opener=opener)

    with (tmp_path / "metadata.csv").open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    files_written = {r["file"] for r in rows}
    assert "location_46/a1.jpg" in files_written
    assert not any("evil" in f for f in files_written)
    assert all((tmp_path / f).read_bytes() == b"jpeg-bytes" for f in files_written)
    assert {r["camera_id"] for r in rows} == {"location_46", "location_7"}
    assert "Community Data License Agreement" in (tmp_path / "attribution.txt").read_text(
        encoding="utf-8"
    )
    assert calls[0] == LILA_CCT_METADATA_URL

    calls.clear()
    fetch_public_sample(tmp_path, settings, np.random.default_rng(1), opener=opener)
    assert calls == [], "metadata is cached and existing images are not downloaded again"


def test_custom_source_and_size_limit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    meta_url, base = "https://example.org/meta.json", "https://example.org/img"
    safe_images = [i for i in IMAGES if "evil" not in i["file_name"]]
    files = {meta_url: json.dumps({"images": safe_images}).encode("utf-8")}
    files.update({f"{base}/{quote(i['file_name'])}": b"x" * 10 for i in IMAGES})
    opener, _ = fake_opener(files)
    monkeypatch.setattr(public_sample, "MAX_IMAGE_BYTES", 5)
    monkeypatch.setattr(public_sample, "DOWNLOAD_CHUNK_BYTES", 4)
    with pytest.raises(ValueError, match="exceeded"):
        fetch_public_sample(
            tmp_path,
            SampleSettings(n_cameras=1, photos_per_camera=1),
            np.random.default_rng(0),
            metadata_url=meta_url,
            image_base_url=base,
            opener=opener,
        )
    monkeypatch.setattr(public_sample, "MAX_IMAGE_BYTES", 50)
    fetch_public_sample(
        tmp_path,
        SampleSettings(n_cameras=1, photos_per_camera=1),
        np.random.default_rng(0),
        metadata_url=meta_url,
        image_base_url=base,
        opener=opener,
    )
    assert (tmp_path / "attribution.txt").read_text(encoding="utf-8") == f"Source: {meta_url}\n"


@pytest.mark.integration
def test_default_metadata_url_is_reachable(tmp_path: Path) -> None:
    fetch_public_sample(
        tmp_path, SampleSettings(n_cameras=1, photos_per_camera=3), np.random.default_rng(0)
    )
    assert (tmp_path / "metadata.csv").exists()


def test_network_failures_become_input_errors(tmp_path: Path) -> None:
    def offline(request: Request, timeout: float) -> io.BytesIO:
        raise URLError("no route to host")

    with pytest.raises(InputError, match="no route to host"):
        fetch_public_sample(tmp_path, SampleSettings(), np.random.default_rng(0), opener=offline)


@pytest.mark.parametrize("raw", [b"not json", b"PK not a zip"])
def test_unreadable_metadata_is_an_input_error(raw: bytes) -> None:
    with pytest.raises(InputError, match="COCO Camera Traps"):
        parse_metadata(raw)


def test_metadata_without_timestamps_cannot_be_sampled() -> None:
    entries = [SampleImage("a.jpg", "1", None, None)]
    with pytest.raises(InputError, match="capture time"):
        choose_sample(entries, SampleSettings(), np.random.default_rng(0))
