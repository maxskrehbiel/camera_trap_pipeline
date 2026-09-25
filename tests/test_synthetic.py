from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

from camera_trap_pipeline.errors import InputError
from camera_trap_pipeline.synthetic import (
    SyntheticDataset,
    SyntheticSettings,
    generate_dataset,
    load_ground_truth,
)
from camera_trap_pipeline.synthetic_scene import step_subject, subjects_for
from camera_trap_pipeline.synthetic_schedule import (
    SPECIES,
    Visit,
    diel_weights,
    is_night,
    plan_cameras,
)


def test_same_seed_gives_identical_dataset(tmp_path: Path) -> None:
    settings = SyntheticSettings(n_cameras=1, n_days=1, width=96, height=64, inject_problems=False)
    first = generate_dataset(tmp_path / "a", settings, np.random.default_rng(5))
    second = generate_dataset(tmp_path / "b", settings, np.random.default_rng(5))
    assert first.n_images == second.n_images > 0
    for path in sorted((tmp_path / "a").rglob("*.jpg")):
        twin = tmp_path / "b" / path.relative_to(tmp_path / "a")
        assert path.read_bytes() == twin.read_bytes()


def test_images_carry_exif_matching_the_camera_clock(tiny_dataset: SyntheticDataset) -> None:
    truth = load_ground_truth(tiny_dataset.root)
    rel, entry = next(iter(truth.images.items()))
    with Image.open(tiny_dataset.root / rel) as image:
        stamp = image.getexif().get_ifd(0x8769)[0x9003]
    assert datetime.strptime(stamp, "%Y:%m:%d %H:%M:%S") == datetime.fromisoformat(
        entry["camera_clock"]
    )


def test_ground_truth_is_consistent(tiny_dataset: SyntheticDataset) -> None:
    truth = load_ground_truth(tiny_dataset.root)
    assert len(truth.images) == tiny_dataset.n_images
    assert sum(v["n_photos"] for v in truth.visits) == tiny_dataset.n_images
    for entry in truth.images.values():
        for obj in entry["objects"]:
            x, y, w, h = obj["bbox"]
            assert 0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1
    kinds = {v["kind"] for v in truth.visits}
    assert "animal" in kinds and "empty" in kinds


def test_injected_problems(tiny_dataset: SyntheticDataset) -> None:
    truth = load_ground_truth(tiny_dataset.root)
    entries = list(truth.images.values())
    assert sum(e["truncated"] for e in entries) == 1
    assert any(e["blurred"] for e in entries)
    assert any(e["camera_clock"].startswith("2000") for e in entries)
    assert any(e["night"] for e in entries) and not all(e["night"] for e in entries)


def test_plan_cameras_spreads_problems() -> None:
    plans = plan_cameras(SyntheticSettings(n_cameras=3, n_days=14))
    assert plans[0].fog_day == 1
    assert plans[1].unset_clock_days == 2
    assert plans[2].outage_days == (6, 10)
    quiet = plan_cameras(SyntheticSettings(n_cameras=2, n_days=2, inject_problems=False))
    assert all(p.outage_days is None and p.fog_day is None for p in quiet)


def test_diel_weights_follow_activity_peaks() -> None:
    by_name = {p.name: p for p in SPECIES}
    raccoon = diel_weights(by_name["northern raccoon"])
    turkey = diel_weights(by_name["wild turkey"])
    assert raccoon.sum() == pytest.approx(1.0)
    assert raccoon[1] > 5 * raccoon[12]
    assert turkey[8] > 5 * turkey[1]


def test_is_night() -> None:
    assert is_night(datetime(2026, 4, 6, 5, 0))
    assert not is_night(datetime(2026, 4, 6, 12, 0))
    assert is_night(datetime(2026, 4, 6, 21, 0))


def test_missing_ground_truth_is_explained(tmp_path: Path) -> None:
    with pytest.raises(InputError, match="synthetic data"):
        load_ground_truth(tmp_path)


@pytest.mark.parametrize(
    "overrides",
    [
        {"n_cameras": 0},
        {"n_days": 0},
        {"burst_size": 0},
        {"width": 10},
        {"info_strip": 0.6},
        {"jpeg_quality": 0},
        {"rate_scale": -1.0},
    ],
)
def test_settings_reject_impossible_datasets(overrides: dict[str, Any]) -> None:
    with pytest.raises(InputError):
        SyntheticSettings(**overrides)


def test_person_and_vehicle_subjects_turn_at_the_edge() -> None:
    settings = SyntheticSettings(width=200, height=100)
    rng = np.random.default_rng(0)
    start = datetime(2026, 4, 6, 12)
    person = subjects_for(Visit(1, "cam", "person", start, None, 1), settings, rng)
    vehicle = subjects_for(Visit(2, "cam", "vehicle", start, None, 1), settings, rng)
    assert [s["kind"] for s in person + vehicle] == ["person", "vehicle"]
    assert subjects_for(Visit(3, "cam", "empty", start), settings, rng) == []
    car = vehicle[0]
    for _ in range(20):
        step_subject(car, settings.width)
        assert 0.1 * settings.width <= car["center"][0] <= 0.9 * settings.width
