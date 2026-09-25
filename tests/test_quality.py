from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest
from PIL import Image, ImageFilter

from camera_trap_pipeline import quality
from camera_trap_pipeline.quality import (
    add_flags,
    crop_strips,
    dhash,
    hamming,
    ir_score,
    laplacian_variance,
    laplacian_variance_numpy,
    measure_image,
    near_duplicate_groups,
    run_quality,
    shrink,
)
from camera_trap_pipeline.settings import QualitySettings


def textured(seed: int, size: tuple[int, int] = (160, 120)) -> Image.Image:
    rng = np.random.default_rng(seed)
    blocks = rng.integers(0, 255, (size[1] // 8, size[0] // 8, 3), dtype=np.uint8)
    return Image.fromarray(blocks).resize(size, Image.Resampling.NEAREST)


def gray(image: Image.Image) -> np.ndarray:
    return np.asarray(image.convert("L"), dtype=np.float64)


def test_laplacian_variance_of_single_impulse_is_exact() -> None:
    impulse = np.zeros((5, 5))
    impulse[2, 2] = 1.0
    # One -4, four +1 and twenty 0 values: mean 0, variance (16 + 4) / 25.
    assert laplacian_variance_numpy(impulse) == pytest.approx(0.8)
    assert laplacian_variance_numpy(np.full((6, 6), 7.0)) == 0.0


def test_blur_lowers_sharpness() -> None:
    sharp = textured(1)
    blurred = sharp.filter(ImageFilter.GaussianBlur(3))
    assert laplacian_variance(gray(blurred)) < 0.35 * laplacian_variance(gray(sharp))


def test_opencv_path_is_used_when_available(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_laplacian(image: np.ndarray, depth: int, ksize: int) -> np.ndarray:
        assert ksize == 1
        return np.full_like(image, 2.0) + np.arange(image.size).reshape(image.shape)

    monkeypatch.setitem(sys.modules, "cv2", SimpleNamespace(Laplacian=fake_laplacian, CV_64F=6))
    quality._opencv_laplacian.cache_clear()
    try:
        data = np.zeros((2, 2))
        assert laplacian_variance(data) == pytest.approx(np.arange(4).var())
    finally:
        quality._opencv_laplacian.cache_clear()


def test_numpy_path_is_used_without_opencv(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "cv2", None)
    quality._opencv_laplacian.cache_clear()
    try:
        data = gray(textured(3))
        assert laplacian_variance(data) == laplacian_variance_numpy(data)
    finally:
        quality._opencv_laplacian.cache_clear()


def test_opencv_matches_numpy_when_installed() -> None:
    cv2 = pytest.importorskip("cv2")
    data = gray(textured(4))
    expected = laplacian_variance_numpy(data)
    assert float(cv2.Laplacian(data, cv2.CV_64F, ksize=1).var()) == pytest.approx(expected)


def test_dhash_separates_near_duplicates_from_new_scenes() -> None:
    base = textured(5)
    noisy_pixels = np.asarray(base, dtype=np.int16) + np.random.default_rng(0).integers(
        -4, 5, (120, 160, 3)
    )
    noisy = Image.fromarray(np.clip(noisy_pixels, 0, 255).astype(np.uint8))
    other = textured(6)
    assert hamming(dhash(base), dhash(base)) == 0
    assert hamming(dhash(base), dhash(noisy)) <= 6
    assert hamming(dhash(base), dhash(other)) > 16
    assert 0 <= dhash(base) < 2**64


def test_ir_score_is_near_zero_for_monochrome() -> None:
    color = np.asarray(textured(7))
    mono = np.repeat(np.asarray(textured(7).convert("L"))[..., None], 3, axis=2)
    assert ir_score(mono) == 0.0
    assert ir_score(color) > 4.0


def test_crop_strips_and_shrink() -> None:
    image = Image.new("RGB", (200, 100))
    assert crop_strips(image, 0.1, 0.2).size == (200, 70)
    assert crop_strips(image, 0.5, 0.49).size == (200, 100)
    assert shrink(image, 50).size == (50, 25)
    assert shrink(image, 500) is image


def test_measure_image_ignores_info_strip(tmp_path: Path) -> None:
    pixels = np.full((100, 100, 3), 120, dtype=np.uint8)
    pixels[90:] = 0
    path = tmp_path / "a.png"
    Image.fromarray(pixels).save(path)
    with_strip = measure_image(path, QualitySettings())
    without_strip = measure_image(path, QualitySettings(strip_bottom=0.1))
    assert without_strip["brightness"] == pytest.approx(120.0)
    assert with_strip["brightness"] < without_strip["brightness"]
    assert len(without_strip["dhash"]) == 16


def frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    table = pd.DataFrame(rows)
    table["timestamp"] = pd.to_datetime(table["timestamp"])
    return table


def test_near_duplicates_respect_camera_window_and_time_gap() -> None:
    a = 0x0F0F0F0F0F0F0F0F
    table = frame(
        [
            {
                "photo_id": "p1",
                "camera_id": "c1",
                "timestamp": "2026-04-06 06:00:00",
                "dhash": f"{a:016x}",
            },
            {
                "photo_id": "p2",
                "camera_id": "c1",
                "timestamp": "2026-04-06 06:00:01",
                "dhash": f"{a ^ 0b111:016x}",
            },
            {
                "photo_id": "p3",
                "camera_id": "c1",
                "timestamp": "2026-04-06 07:00:00",
                "dhash": f"{a:016x}",
            },
            {
                "photo_id": "p4",
                "camera_id": "c2",
                "timestamp": "2026-04-06 06:00:01",
                "dhash": f"{a:016x}",
            },
            {
                "photo_id": "p5",
                "camera_id": "c1",
                "timestamp": "2026-04-06 07:00:02",
                "dhash": f"{~a & (2**64 - 1):016x}",
            },
            {
                "photo_id": "p6",
                "camera_id": "c1",
                "timestamp": "2026-04-06 07:00:03",
                "dhash": None,
            },
        ]
    )
    table["rel_path"] = table["photo_id"]
    before = table["photo_id"].tolist()
    groups = near_duplicate_groups(table, max_bits=6, window=8, max_gap_s=300)
    assert groups.tolist() == ["p1", "p1", "p3", "p4", "p5", "p6"]
    assert table["photo_id"].tolist() == before, "input frame must not be modified"


def test_add_flags_uses_per_camera_relative_blur() -> None:
    table = frame(
        [
            {
                "photo_id": f"p{i}",
                "camera_id": "c1",
                "timestamp": f"2026-04-06 0{i}:00:00",
                "sharpness": s,
                "brightness": b,
                "ir_score": 20.0,
                "dhash": f"{i:016x}",
            }
            for i, (s, b) in enumerate(
                [(100.0, 120.0), (110.0, 10.0), (90.0, 240.0), (20.0, 120.0)]
            )
        ]
    )
    table["rel_path"] = table["photo_id"]
    table["decode_ok"] = True
    flagged = add_flags(table, QualitySettings()).set_index("photo_id")
    assert flagged["is_blurry"].tolist() == [False, False, False, True]
    assert flagged["is_dark"].tolist() == [False, True, False, False]
    assert flagged["is_overexposed"].tolist() == [False, False, True, False]
    assert not flagged["is_ir"].any()


def test_run_quality_measures_once_and_retries_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "photos"
    (root / "cam").mkdir(parents=True)
    textured(8).save(root / "cam" / "a.jpg")
    textured(9).save(root / "cam" / "b.jpg")
    data = (root / "cam" / "b.jpg").read_bytes()
    (root / "cam" / "c.jpg").write_bytes(data[: len(data) // 3])
    media = pd.DataFrame(
        {
            "photo_id": ["a", "b", "c", "d"],
            "rel_path": ["cam/a.jpg", "cam/b.jpg", "cam/c.jpg", "cam/d.jpg"],
            "camera_id": "cam",
            "timestamp": pd.to_datetime(["2026-04-06 06:00:00"] * 4),
            "readable": [True, True, True, False],
        }
    )
    checkpoint = tmp_path / "quality.jsonl"
    first = run_quality(media, root, checkpoint, QualitySettings(), workers=2).set_index("photo_id")
    assert first["decode_ok"].tolist() == [True, True, False, False]
    assert first.loc["c", "quality_error"] == "OSError"
    assert first.loc["d", "quality_error"] == "unreadable"

    measured: list[Path] = []
    original = quality.measure_image

    def spy(path: Path, settings: QualitySettings) -> dict[str, Any]:
        measured.append(path)
        return original(path, settings)

    monkeypatch.setattr(quality, "measure_image", spy)
    run_quality(media, root, checkpoint, QualitySettings(), workers=1)
    assert [p.name for p in measured] == ["c.jpg"]


def test_run_quality_with_no_readable_photos(tmp_path: Path) -> None:
    media = pd.DataFrame(
        {
            "photo_id": ["a"],
            "rel_path": ["cam/a.jpg"],
            "camera_id": "cam",
            "timestamp": pd.to_datetime(["2026-04-06 06:00:00"]),
            "readable": [False],
        }
    )
    table = run_quality(media, tmp_path, tmp_path / "q.jsonl", QualitySettings(), workers=1)
    assert table["decode_ok"].tolist() == [False]
    assert table["quality_error"].tolist() == ["unreadable"]
