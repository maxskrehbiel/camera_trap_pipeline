"""Summary tables: species totals, activity by hour, species by camera and camera health."""

from __future__ import annotations

from typing import Any

import pandas as pd

from .settings import HealthSettings
from .taxonomy import LEVELS

SECONDS_PER_DAY = 86_400.0
SECONDS_PER_HOUR = 3_600.0
HOURS = range(24)


def species_totals(event_species: pd.DataFrame) -> pd.DataFrame:
    """Total each label over all cameras.

    Args:
        event_species: Event-species table.

    Returns:
        One row per label with events, cameras, summed event counts and first and last
        sighting, most frequent first.
    """
    if event_species.empty:
        return pd.DataFrame(
            columns=["species", "n_events", "n_cameras", "animals", "first_seen", "last_seen"]
        )
    totals = event_species.groupby("species", as_index=False).agg(
        n_events=("event_id", "nunique"),
        n_cameras=("camera_id", "nunique"),
        animals=("max_count", "sum"),
        first_seen=("start", "min"),
        last_seen=("start", "max"),
    )
    return totals.sort_values(["n_events", "species"], ascending=[False, True]).reset_index(
        drop=True
    )


def activity_by_hour(event_species: pd.DataFrame) -> pd.DataFrame:
    """Events per label and hour of day (camera-local), including hours with none.

    Args:
        event_species: Event-species table.

    Returns:
        Long table with ``species``, ``hour`` (0-23) and ``n_events``.
    """
    if event_species.empty:
        return pd.DataFrame(columns=["species", "hour", "n_events"])
    hours = event_species.assign(hour=event_species["start"].dt.hour)
    counts = hours.groupby(["species", "hour"]).size()
    grid = pd.MultiIndex.from_product(
        [sorted(event_species["species"].unique()), HOURS], names=["species", "hour"]
    )
    return counts.reindex(grid, fill_value=0).rename("n_events").reset_index()


def camera_gaps(photos: pd.DataFrame, settings: HealthSettings) -> pd.DataFrame:
    """Silences longer than ``settings.gap_hours`` between consecutive photos of a camera.

    A trail camera only fires on motion, so a gap can be a quiet spell rather than an
    outage; long gaps are still the first thing to check when a camera goes dark.

    Args:
        photos: Photo table.
        settings: Health settings.

    Returns:
        ``camera_id``, ``gap_start``, ``gap_end`` and ``gap_hours`` per gap.
    """
    timed = photos[photos["timestamp_valid"]].sort_values(["camera_id", "timestamp"])
    previous = timed.groupby("camera_id")["timestamp"].shift()
    hours = (timed["timestamp"] - previous).dt.total_seconds() / SECONDS_PER_HOUR
    long_gap = hours > settings.gap_hours
    gaps = pd.DataFrame(
        {
            "camera_id": timed.loc[long_gap, "camera_id"],
            "gap_start": previous[long_gap],
            "gap_end": timed.loc[long_gap, "timestamp"],
            "gap_hours": hours[long_gap].round(1),
        }
    )
    return gaps.reset_index(drop=True)


def camera_health(photos: pd.DataFrame, events: pd.DataFrame, gaps: pd.DataFrame) -> pd.DataFrame:
    """Per-camera uptime, effort and image-quality indicators.

    ``effort_days`` is the span from first to last photo minus the reported gaps, the usual
    denominator for detection rates in camera-trap studies.

    Args:
        photos: Photo table.
        events: Event table.
        gaps: Output of ``camera_gaps``.

    Returns:
        One row per camera.
    """
    rows = []
    for camera_id, group in photos.groupby("camera_id", sort=True):
        timed = group[group["timestamp_valid"]].sort_values("timestamp")
        decoded = group[group["decode_ok"]]
        detected = decoded[decoded["detected"]]
        camera_gap_hours = gaps.loc[gaps["camera_id"] == camera_id, "gap_hours"]
        first, last = (timed["timestamp"].min(), timed["timestamp"].max())
        span_days = (last - first).total_seconds() / SECONDS_PER_DAY if len(timed) else 0.0
        steps = timed["timestamp"].diff().dt.total_seconds() / SECONDS_PER_HOUR
        camera_events = events[events["camera_id"] == camera_id]
        effort_days = max(span_days - float(camera_gap_hours.sum()) / 24.0, 0.0)
        empty = (detected[["n_animals", "n_people", "n_vehicles"]].sum(axis=1) == 0).mean()
        rows.append(
            {
                "camera_id": camera_id,
                "n_photos": len(group),
                "n_unreadable": int((~group["decode_ok"]).sum()),
                "n_invalid_timestamp": int((~group["timestamp_valid"]).sum()),
                "first_photo": first,
                "last_photo": last,
                "span_days": round(span_days, 2),
                "active_days": int(timed["timestamp"].dt.normalize().nunique()),
                "effort_days": round(effort_days, 2),
                "n_gaps": len(camera_gap_hours),
                "longest_silence_hours": round(float(steps.max()), 1)
                if steps.notna().any()
                else None,
                "n_events": len(camera_events),
                "n_animal_events": int((camera_events["observation_type"] == "animal").sum()),
                "pct_ir": _pct(decoded["is_ir"]),
                "pct_dark": _pct(decoded["is_dark"]),
                "pct_blurry": _pct(decoded["is_blurry"]),
                "pct_near_duplicate": _pct(decoded["is_near_duplicate"]),
                "pct_empty": round(100.0 * float(empty), 1) if len(detected) else None,
            }
        )
    return pd.DataFrame(rows)


def _pct(flags: pd.Series) -> float | None:
    return round(100.0 * float(flags.mean()), 1) if len(flags) else None


def species_by_camera(event_species: pd.DataFrame, health: pd.DataFrame) -> pd.DataFrame:
    """Events per label and camera, with a relative abundance index.

    The index is events per 100 effort days, a common way to compare how often a species
    passes different cameras that ran for different lengths of time.

    Args:
        event_species: Event-species table.
        health: Output of ``camera_health`` (for effort days).

    Returns:
        One row per camera and label, most frequent first within each camera.
    """
    columns = [
        "camera_id",
        "species",
        "n_events",
        "animals",
        "max_group",
        "events_per_100_days",
    ]
    if event_species.empty:
        return pd.DataFrame(columns=columns)
    table = event_species.groupby(["camera_id", "species"], as_index=False).agg(
        n_events=("event_id", "nunique"),
        animals=("max_count", "sum"),
        max_group=("max_count", "max"),
    )
    effort = table["camera_id"].map(health.set_index("camera_id")["effort_days"])
    table["events_per_100_days"] = (100.0 * table["n_events"] / effort.where(effort > 0)).round(1)
    return table.sort_values(["camera_id", "n_events"], ascending=[True, False])[
        columns
    ].reset_index(drop=True)


def daily_photos(photos: pd.DataFrame) -> pd.DataFrame:
    """Count photos per camera and calendar day (valid timestamps only).

    Args:
        photos: Photo table.

    Returns:
        ``camera_id``, ``date`` and ``n_photos``.
    """
    timed = photos[photos["timestamp_valid"]]
    counts = timed.groupby(["camera_id", timed["timestamp"].dt.normalize().rename("date")]).size()
    return counts.rename("n_photos").reset_index()


def overview(
    photos: pd.DataFrame, events: pd.DataFrame, event_species: pd.DataFrame
) -> dict[str, Any]:
    """Compute the headline numbers for the report.

    Args:
        photos: Photo table.
        events: Event table.
        event_species: Event-species table.

    Returns:
        Counts and percentages keyed by name.
    """
    timed = photos.loc[photos["timestamp_valid"], "timestamp"]
    decoded = photos[photos["decode_ok"]]
    species_depth = len(LEVELS)
    resolved = event_species[event_species["lineage"].str.count(";") == species_depth - 1]
    return {
        "n_photos": len(photos),
        "n_cameras": int(photos["camera_id"].nunique()),
        "first_photo": timed.min() if len(timed) else None,
        "last_photo": timed.max() if len(timed) else None,
        "n_unreadable": int((~photos["decode_ok"]).sum()),
        "n_invalid_timestamp": int((~photos["timestamp_valid"]).sum()),
        "n_events": len(events),
        "n_animal_events": int((events["observation_type"] == "animal").sum()),
        "n_person_events": int((events["observation_type"] == "person").sum()),
        "n_vehicle_events": int((events["observation_type"] == "vehicle").sum()),
        "n_blank_events": int((events["observation_type"] == "blank").sum()),
        "n_species": int(resolved["species"].nunique()),
        "n_labels": int(event_species["species"].nunique()),
        "pct_near_duplicate": _pct(decoded["is_near_duplicate"]),
        "pct_ir": _pct(decoded["is_ir"]),
    }


def build_summaries(
    photos: pd.DataFrame,
    events: pd.DataFrame,
    event_species: pd.DataFrame,
    settings: HealthSettings,
) -> dict[str, pd.DataFrame]:
    """Compute every summary table.

    Args:
        photos: Photo table.
        events: Event table.
        event_species: Event-species table.
        settings: Health settings.

    Returns:
        Tables keyed by their output file stem.
    """
    gaps = camera_gaps(photos, settings)
    health = camera_health(photos, events, gaps)
    return {
        "species_totals": species_totals(event_species),
        "activity_by_hour": activity_by_hour(event_species),
        "species_by_camera": species_by_camera(event_species, health),
        "camera_health": health,
        "camera_gaps": gaps,
        "daily_photos": daily_photos(photos),
    }
