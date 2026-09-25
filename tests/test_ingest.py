from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest
from PIL import Image

from camera_trap_pipeline.errors import InputError
from camera_trap_pipeline.ingest import (
    find_images,
    find_new_images,
    ingest,
    parse_timestamp,
    photo_id_for,
)

NOW = datetime(2026, 5, 1, 12, 0)


def save_photo(
    path: Path,
    original: str | None = None,
    plain: str | None = None,
    make: str | None = None,
    model: str | None = None,
    serial: str | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exif = Image.Exif()
    if make:
        exif[0x010F] = make
    if model:
        exif[0x0110] = model
    if plain:
        exif[0x0132] = plain
    sub = {}
    if original:
        sub[0x9003] = original
    if serial:
        sub[0xA431] = serial
    if sub:
        exif[0x8769] = sub
    Image.new("RGB", (32, 24), (90, 120, 60)).save(path, "JPEG", exif=exif)


def row(media: pd.DataFrame, rel_path: str) -> pd.Series:
    return media.set_index("rel_path").loc[rel_path]


def test_camera_from_folder_and_timestamp_from_exif(tmp_path: Path) -> None:
    save_photo(tmp_path / "cam_a" / "img_1.jpg", original="2026:04:06 06:30:00")
    save_photo(tmp_path / "cam_a" / "sub" / "img_2.jpg", plain="2026:04:06 06:31:00")
    media = ingest(tmp_path, "2001-01-01", workers=2, now=NOW)

    first = row(media, "cam_a/img_1.jpg")
    assert first["camera_id"] == "cam_a"
    assert first["camera_source"] == "folder"
    assert first["timestamp"] == pd.Timestamp("2026-04-06 06:30:00")
    assert first["timestamp_source"] == "exif_original"
    assert bool(first["timestamp_valid"])
    assert first["photo_id"] == photo_id_for("cam_a/img_1.jpg")
    assert row(media, "cam_a/sub/img_2.jpg")["timestamp_source"] == "exif_datetime"


def test_camera_falls_back_to_exif_then_unknown(tmp_path: Path) -> None:
    save_photo(tmp_path / "serial.jpg", original="2026:04:06 06:30:00", serial="SN-9")
    save_photo(tmp_path / "model.jpg", original="2026:04:06 06:30:00", make="Acme", model="T1")
    save_photo(tmp_path / "bare.jpg")
    media = ingest(tmp_path, "2001-01-01", now=NOW)
    assert row(media, "serial.jpg")["camera_id"] == "SN-9"
    assert row(media, "model.jpg")["camera_id"] == "Acme T1"
    bare = row(media, "bare.jpg")
    assert bare["camera_id"] == "unknown"
    assert bare["timestamp_source"] == "missing"
    assert not bool(bare["timestamp_valid"])


def test_unset_and_future_clocks_are_flagged_invalid(tmp_path: Path) -> None:
    save_photo(tmp_path / "cam" / "reset.jpg", original="2000:01:01 00:05:00")
    save_photo(tmp_path / "cam" / "future.jpg", original="2031:01:01 00:00:00")
    save_photo(tmp_path / "cam" / "ok.jpg", original="2026:04:30 23:00:00")
    media = ingest(tmp_path, "2001-01-01", now=NOW)
    valid = dict(zip(media["rel_path"], media["timestamp_valid"], strict=True))
    assert valid == {"cam/reset.jpg": False, "cam/future.jpg": False, "cam/ok.jpg": True}


def test_sidecar_overrides_camera_and_timestamp(tmp_path: Path) -> None:
    save_photo(tmp_path / "a" / "x.jpg", original="2026:04:06 06:30:00")
    (tmp_path / "metadata.csv").write_text(
        "file,camera_id,timestamp\na/x.jpg,site_7,2013-10-04 13:31:53\n", encoding="utf-8"
    )
    media = ingest(tmp_path, "2001-01-01", now=NOW)
    only = row(media, "a/x.jpg")
    assert (only["camera_id"], only["camera_source"]) == ("site_7", "sidecar")
    assert only["timestamp"] == pd.Timestamp("2013-10-04 13:31:53")
    assert only["timestamp_source"] == "sidecar"


def test_sidecar_without_file_column_is_an_error(tmp_path: Path) -> None:
    save_photo(tmp_path / "a" / "x.jpg")
    (tmp_path / "metadata.csv").write_text("path,camera_id\na/x.jpg,s\n", encoding="utf-8")
    with pytest.raises(ValueError, match="'file' column"):
        ingest(tmp_path, "2001-01-01", now=NOW)


def test_unreadable_file_is_kept_and_flagged(tmp_path: Path) -> None:
    (tmp_path / "cam").mkdir()
    (tmp_path / "cam" / "broken.jpg").write_bytes(b"not a jpeg at all")
    media = ingest(tmp_path, "2001-01-01", now=NOW)
    broken = row(media, "cam/broken.jpg")
    assert not bool(broken["readable"])
    assert broken["error"] == "UnidentifiedImageError"


def test_hidden_entries_and_other_files_are_ignored(tmp_path: Path) -> None:
    save_photo(tmp_path / "cam" / "a.JPG")
    save_photo(tmp_path / ".cache" / "b.jpg")
    save_photo(tmp_path / "cam" / ".c.jpg")
    (tmp_path / "cam" / "notes.txt").write_text("x", encoding="utf-8")
    assert find_images(tmp_path) == ["cam/a.JPG"]


def test_find_new_images_lists_files_added_after_ingest(tmp_path: Path) -> None:
    save_photo(tmp_path / "cam" / "a.jpg")
    media = ingest(tmp_path, "2001-01-01", now=NOW)
    save_photo(tmp_path / "cam" / "b.jpg")
    assert find_new_images(tmp_path, media) == ["cam/b.jpg"]


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("2026:04:06 06:30:00", datetime(2026, 4, 6, 6, 30)),
        (b"2026:04:06 06:30:00\x00", datetime(2026, 4, 6, 6, 30)),
        ("2026-04-06T06:30:00+02:00", datetime(2026, 4, 6, 6, 30)),
        ("6 April 2026 06:30", datetime(2026, 4, 6, 6, 30)),
        ("2026-04-06 06:30:00Z", datetime(2026, 4, 6, 6, 30)),
        ("0000:00:00 00:00:00", None),
        ("not a date", None),
        ("", None),
        (None, None),
    ],
)
def test_parse_timestamp(value: object, expected: datetime | None) -> None:
    assert parse_timestamp(value) == expected


def test_photo_ids_are_stable_hex() -> None:
    assert photo_id_for("cam/a.jpg") == photo_id_for("cam/a.jpg")
    assert photo_id_for("cam/a.jpg") != photo_id_for("cam/b.jpg")
    assert len(photo_id_for("x")) == 16
    int(photo_id_for("x"), 16)


def test_empty_folder_is_an_input_error(tmp_path: Path) -> None:
    (tmp_path / "cam").mkdir()
    (tmp_path / "cam" / "notes.txt").write_text("no photos here", encoding="utf-8")
    with pytest.raises(InputError, match="no images found"):
        ingest(tmp_path, "2001-01-01", now=NOW)
