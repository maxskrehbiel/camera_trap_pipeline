"""Events stage: join per-photo results and group each camera's bursts into sightings.

An event's count per species is its largest single-frame count, never a sum over frames.
"""

from __future__ import annotations

import pandas as pd

from .taxonomy import UNRESOLVED, common_lineage, fold_labels, label_for_lineage

PHOTO_SPECIES_COLUMNS = ["photo_id", "species", "lineage", "count", "max_score"]
EVENT_COLUMNS = [
    "event_id",
    "camera_id",
    "start",
    "end",
    "duration_s",
    "n_photos",
    "n_animal_photos",
    "max_animals",
    "has_person",
    "has_vehicle",
    "observation_type",
    "species",
    "species_count",
    "best_photo_id",
]
EVENT_SPECIES_COLUMNS = [
    "event_id",
    "camera_id",
    "start",
    "species",
    "lineage",
    "max_count",
    "n_photos",
    "max_score",
]


def build_photo_table(
    media: pd.DataFrame,
    quality: pd.DataFrame,
    boxes: pd.DataFrame,
    runs: pd.DataFrame,
    classifications: pd.DataFrame,
    count_conf: float,
) -> pd.DataFrame:
    """Join media, quality and detection results into one row per photo.

    Args:
        media: Media table.
        quality: Quality table.
        boxes: Box table from the detect stage.
        runs: Per-photo detection runs (model, errors).
        classifications: Box labels from the classify stage.
        count_conf: Boxes at or above this confidence are counted.

    Returns:
        Media and quality columns plus ``detected``, ``detect_error``, ``n_animals``,
        ``n_people``, ``n_vehicles``, ``max_animal_conf`` and ``species`` (label of the most
        confident animal box).
    """
    photos = media.merge(quality, on="photo_id", how="left")
    counted = boxes[boxes["conf"] >= count_conf]
    counts = (
        counted.pivot_table(
            index="photo_id", columns="category", values="conf", aggfunc="size", fill_value=0
        )
        .reindex(columns=["animal", "person", "vehicle"], fill_value=0)
        .rename(columns={"animal": "n_animals", "person": "n_people", "vehicle": "n_vehicles"})
    )
    animal_conf = counted[counted["category"] == "animal"].groupby("photo_id")["conf"].max()
    photos = photos.merge(counts, left_on="photo_id", right_index=True, how="left")
    for column in ("n_animals", "n_people", "n_vehicles"):
        photos[column] = photos[column].fillna(0).astype(int)
    photos["max_animal_conf"] = photos["photo_id"].map(animal_conf)
    finished = runs[runs["error"].isna()]
    photos["detected"] = photos["photo_id"].isin(finished["photo_id"])
    photos["detect_error"] = photos["photo_id"].map(runs.set_index("photo_id")["error"])
    top_label = classifications.sort_values(["photo_id", "box_idx"]).drop_duplicates("photo_id")
    photos["species"] = photos["photo_id"].map(top_label.set_index("photo_id")["species"])
    for column in ("decode_ok", "is_ir", "is_dark", "is_overexposed", "is_blurry"):
        photos[column] = photos[column].astype("boolean").fillna(False).astype(bool)
    photos["is_near_duplicate"] = (
        photos["is_near_duplicate"].astype("boolean").fillna(False).astype(bool)
    )
    return photos.reset_index(drop=True)


def photo_species(
    boxes: pd.DataFrame, classifications: pd.DataFrame, count_conf: float
) -> pd.DataFrame:
    """Count each label per photo over the counted animal boxes.

    Counted animal boxes that were not classified (beyond the per-photo cap, or no
    classifier configured) count as ``animal (unresolved)``.

    Args:
        boxes: Box table.
        classifications: Box labels.
        count_conf: Counting confidence.

    Returns:
        One row per photo and label with ``count`` and ``max_score``.
    """
    animals = boxes[(boxes["category"] == "animal") & (boxes["conf"] >= count_conf)]
    labeled = animals[["photo_id", "box_idx"]].merge(
        classifications[["photo_id", "box_idx", "species", "lineage", "species_score"]],
        on=["photo_id", "box_idx"],
        how="left",
    )
    unresolved = labeled["species"].isna()
    labeled.loc[unresolved, "species"] = UNRESOLVED
    labeled.loc[unresolved, "lineage"] = ""
    labeled["species_score"] = labeled["species_score"].astype(float)
    grouped = labeled.groupby(["photo_id", "species", "lineage"], as_index=False).agg(
        count=("box_idx", "size"), max_score=("species_score", "max")
    )
    return grouped[PHOTO_SPECIES_COLUMNS]


def assign_events(photos: pd.DataFrame, gap_s: float) -> pd.Series:
    """Give each photo with a valid timestamp an event id.

    Photos are sorted by camera and time; a new event starts at a camera's first photo and
    whenever the gap to the previous photo exceeds ``gap_s``. Gaps chain, so a long visit
    photographed every 30 s stays one event even if it lasts an hour.

    Args:
        photos: Needs ``camera_id``, ``timestamp``, ``timestamp_valid``, ``decode_ok`` and
            ``rel_path``.
        gap_s: Largest gap in seconds that keeps two photos in one event.

    Returns:
        ``<camera_id>-e<sequence>`` per row; missing for photos left out (invalid timestamp
        or undecodable).
    """
    eligible = photos["timestamp_valid"].astype(bool) & photos["decode_ok"].astype(bool)
    ordered = photos.loc[eligible, ["camera_id", "timestamp", "rel_path"]].sort_values(
        ["camera_id", "timestamp", "rel_path"]
    )
    gap = ordered.groupby("camera_id")["timestamp"].diff().dt.total_seconds()
    new_event = gap.isna() | (gap > gap_s)
    sequence = new_event.astype(int).groupby(ordered["camera_id"]).cumsum()
    ids = ordered["camera_id"].astype(str) + "-e" + sequence.astype(str).str.zfill(5)
    return pd.Series(ids, index=ordered.index, dtype=object).reindex(photos.index)


def _resolve_labels(species_rows: pd.DataFrame, single_animal: set[str]) -> pd.DataFrame:
    """Reconcile the labels given to the frames of each event.

    Vaguer labels fold into the single more specific label they contain. If labels still
    disagree in an event where no frame shows more than one animal, the frames must show
    the same animal, so every label is replaced by their closest common taxon.
    """
    mixed = species_rows.groupby("event_id")["species"].transform("nunique") > 1
    if not mixed.any():
        return species_rows
    out = species_rows.copy()
    for event_id, group in out[mixed].groupby("event_id"):
        lineages = dict(zip(group["species"], group["lineage"], strict=True))
        mapping = fold_labels(lineages)
        if event_id in single_animal and len(set(mapping.values())) > 1:
            shared = common_lineage(lineages.values())
            name = next((n for n, lin in lineages.items() if lin == shared), None)
            name = name or label_for_lineage(shared)
            lineages[name] = shared
            mapping = dict.fromkeys(lineages, name)
        out.loc[group.index, "species"] = group["species"].map(mapping)
        out.loc[group.index, "lineage"] = group["species"].map(mapping).map(lineages)
    return out


def build_events(
    photos: pd.DataFrame, species_per_photo: pd.DataFrame, gap_s: float
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Group photos into events and summarize each event and each species within it.

    Args:
        photos: Photo table from ``build_photo_table``.
        species_per_photo: Output of ``photo_species``.
        gap_s: Event gap in seconds.

    Returns:
        ``(photos_with_event_id, events, event_species)``.
    """
    photos = photos.copy()
    photos["event_id"] = assign_events(photos, gap_s)
    in_events = photos[photos["event_id"].notna()].copy()
    in_events["has_animal"] = in_events["n_animals"] > 0
    grouped = in_events.groupby("event_id", sort=True)
    events = grouped.agg(
        camera_id=("camera_id", "first"),
        start=("timestamp", "min"),
        end=("timestamp", "max"),
        n_photos=("photo_id", "size"),
        n_animal_photos=("has_animal", "sum"),
        max_animals=("n_animals", "max"),
        max_people=("n_people", "max"),
        max_vehicles=("n_vehicles", "max"),
    ).reset_index()
    events["duration_s"] = (events["end"] - events["start"]).dt.total_seconds()
    events["has_person"] = events["max_people"] > 0
    events["has_vehicle"] = events["max_vehicles"] > 0
    events["observation_type"] = "blank"
    events.loc[events["has_vehicle"], "observation_type"] = "vehicle"
    events.loc[events["has_person"], "observation_type"] = "person"
    events.loc[events["n_animal_photos"] > 0, "observation_type"] = "animal"
    best = in_events.sort_values(
        ["event_id", "n_animals", "max_animal_conf", "timestamp"],
        ascending=[True, False, False, True],
        na_position="last",
    ).drop_duplicates("event_id")
    events["best_photo_id"] = events["event_id"].map(best.set_index("event_id")["photo_id"])

    event_species = _event_species(in_events, species_per_photo)
    primary = event_species.sort_values(
        ["event_id", "max_count", "n_photos", "max_score", "species"],
        ascending=[True, False, False, False, True],
    ).drop_duplicates("event_id")
    primary = primary.set_index("event_id")
    events["species"] = events["event_id"].map(primary["species"])
    events["species_count"] = events["event_id"].map(primary["max_count"]).fillna(0).astype(int)
    return photos, events[EVENT_COLUMNS], event_species


def _event_species(in_events: pd.DataFrame, species_per_photo: pd.DataFrame) -> pd.DataFrame:
    rows = species_per_photo.merge(
        in_events[["photo_id", "event_id", "camera_id", "timestamp"]], on="photo_id"
    )
    if rows.empty:
        return pd.DataFrame(columns=EVENT_SPECIES_COLUMNS)
    most_animals = in_events.groupby("event_id")["n_animals"].max()
    rows = _resolve_labels(rows, set(most_animals[most_animals <= 1].index))
    per_photo = rows.groupby(["event_id", "photo_id", "species", "lineage"], as_index=False).agg(
        count=("count", "sum"), max_score=("max_score", "max")
    )
    summary = per_photo.groupby(["event_id", "species", "lineage"], as_index=False).agg(
        max_count=("count", "max"),
        n_photos=("photo_id", "nunique"),
        max_score=("max_score", "max"),
    )
    starts = in_events.groupby("event_id").agg(
        camera_id=("camera_id", "first"), start=("timestamp", "min")
    )
    summary = summary.merge(starts, left_on="event_id", right_index=True)
    summary["max_count"] = summary["max_count"].astype(int)
    return summary.sort_values(["event_id", "species"])[EVENT_SPECIES_COLUMNS].reset_index(
        drop=True
    )
