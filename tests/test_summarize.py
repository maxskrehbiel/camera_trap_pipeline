from __future__ import annotations

import pandas as pd
import pytest

from camera_trap_pipeline.settings import HealthSettings
from camera_trap_pipeline.summarize import (
    activity_by_hour,
    build_summaries,
    camera_gaps,
    camera_health,
    daily_photos,
    overview,
    species_by_camera,
    species_totals,
)

DEER = "mammalia;cetartiodactyla;cervidae;odocoileus;virginianus"
COYOTE = "mammalia;carnivora;canidae;canis;latrans"


def event_species_table() -> pd.DataFrame:
    rows = [
        ("c1-e1", "c1", "2026-04-06 06:10", "white-tailed deer", DEER, 2),
        ("c1-e2", "c1", "2026-04-06 06:50", "white-tailed deer", DEER, 1),
        ("c1-e3", "c1", "2026-04-07 22:05", "coyote", COYOTE, 1),
        ("c2-e1", "c2", "2026-04-06 19:30", "white-tailed deer", DEER, 3),
        ("c2-e2", "c2", "2026-04-07 23:00", "mammalia (class)", "mammalia", 1),
    ]
    table = pd.DataFrame(
        rows, columns=["event_id", "camera_id", "start", "species", "lineage", "max_count"]
    )
    table["start"] = pd.to_datetime(table["start"])
    table["n_photos"] = 3
    table["max_score"] = 0.9
    return table


def photos_table() -> pd.DataFrame:
    times = [
        ("c1", "2026-04-06 06:10"),
        ("c1", "2026-04-06 06:50"),
        ("c1", "2026-04-07 22:05"),
        ("c1", "2026-04-10 10:05"),  # 60 h after the previous photo
        ("c2", "2026-04-06 19:30"),
        ("c2", "2026-04-07 23:00"),
        ("c2", "2000-01-01 00:00"),
    ]
    photos = pd.DataFrame(times, columns=["camera_id", "timestamp"])
    photos["timestamp"] = pd.to_datetime(photos["timestamp"])
    photos["photo_id"] = [f"p{i}" for i in range(len(photos))]
    photos["timestamp_valid"] = photos["timestamp"].dt.year > 2001
    photos["decode_ok"] = [True] * 6 + [False]
    photos["detected"] = photos["decode_ok"]
    photos["n_animals"] = [2, 1, 1, 0, 3, 1, 0]
    photos["n_people"] = 0
    photos["n_vehicles"] = 0
    photos["is_ir"] = [False, False, True, False, False, True, False]
    photos["is_dark"] = False
    photos["is_blurry"] = [False, True, False, False, False, False, False]
    photos["is_near_duplicate"] = False
    return photos


def events_table() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "event_id": ["c1-e1", "c1-e2", "c1-e3", "c1-e4", "c2-e1", "c2-e2"],
            "camera_id": ["c1", "c1", "c1", "c1", "c2", "c2"],
            "observation_type": ["animal", "animal", "animal", "blank", "animal", "animal"],
        }
    )


def test_species_totals_sorted_by_events() -> None:
    totals = species_totals(event_species_table())
    assert totals["species"].tolist()[0] == "white-tailed deer"
    deer = totals.set_index("species").loc["white-tailed deer"]
    assert (deer["n_events"], deer["n_cameras"], deer["animals"]) == (3, 2, 6)


def test_activity_by_hour_has_full_grid_and_right_counts() -> None:
    activity = activity_by_hour(event_species_table())
    assert len(activity) == 3 * 24
    deer = activity[activity["species"] == "white-tailed deer"].set_index("hour")["n_events"]
    assert deer[6] == 2
    assert deer[19] == 1
    assert deer.sum() == 3


def test_camera_gaps_and_health() -> None:
    photos = photos_table()
    gaps = camera_gaps(photos, HealthSettings(gap_hours=48))
    assert gaps["camera_id"].tolist() == ["c1"]
    assert gaps["gap_hours"].iloc[0] == pytest.approx(60.0)

    health = camera_health(photos, events_table(), gaps).set_index("camera_id")
    c1 = health.loc["c1"]
    span_days = (
        pd.Timestamp("2026-04-10 10:05") - pd.Timestamp("2026-04-06 06:10")
    ).total_seconds()
    assert c1["span_days"] == pytest.approx(round(span_days / 86400, 2))
    assert c1["effort_days"] == pytest.approx(round(span_days / 86400 - 60 / 24, 2))
    assert (c1["n_gaps"], c1["active_days"], c1["n_events"], c1["n_animal_events"]) == (1, 3, 4, 3)
    assert c1["pct_blurry"] == 25.0
    assert c1["pct_empty"] == 25.0
    c2 = health.loc["c2"]
    assert (c2["n_invalid_timestamp"], c2["n_unreadable"]) == (1, 1)
    assert c2["pct_ir"] == 50.0


def test_species_by_camera_relative_abundance() -> None:
    health = pd.DataFrame({"camera_id": ["c1", "c2"], "effort_days": [4.0, 0.0]})
    table = species_by_camera(event_species_table(), health)
    c1_deer = table[(table["camera_id"] == "c1") & (table["species"] == "white-tailed deer")].iloc[
        0
    ]
    assert (c1_deer["n_events"], c1_deer["animals"], c1_deer["max_group"]) == (2, 3, 2)
    assert c1_deer["events_per_100_days"] == 50.0
    assert table.loc[table["camera_id"] == "c2", "events_per_100_days"].isna().all()


def test_daily_photos_and_overview() -> None:
    photos = photos_table()
    daily = daily_photos(photos)
    assert daily["n_photos"].sum() == 6
    summary = overview(photos, events_table(), event_species_table())
    assert summary["n_photos"] == 7
    assert summary["n_species"] == 2
    assert summary["n_labels"] == 3
    assert summary["n_animal_events"] == 5
    assert summary["n_blank_events"] == 1
    assert summary["n_invalid_timestamp"] == 1


def test_empty_inputs_produce_empty_tables() -> None:
    empty = event_species_table().iloc[0:0]
    assert species_totals(empty).empty
    assert activity_by_hour(empty).empty
    assert species_by_camera(empty, pd.DataFrame({"camera_id": [], "effort_days": []})).empty
    tables = build_summaries(photos_table(), events_table(), empty, HealthSettings())
    assert set(tables) == {
        "species_totals",
        "activity_by_hour",
        "species_by_camera",
        "camera_health",
        "camera_gaps",
        "daily_photos",
    }
