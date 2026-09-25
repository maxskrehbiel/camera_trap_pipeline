from __future__ import annotations

from typing import Any

import pandas as pd

from camera_trap_pipeline.classify import CLASSIFICATION_COLUMNS
from camera_trap_pipeline.detect import BOX_COLUMNS, RUN_COLUMNS
from camera_trap_pipeline.events import (
    assign_events,
    build_events,
    build_photo_table,
    photo_species,
)
from camera_trap_pipeline.taxonomy import UNRESOLVED

DEER = "mammalia;cetartiodactyla;cervidae;odocoileus;virginianus"
FAMILY = "mammalia;cetartiodactyla;cervidae"
COYOTE = "mammalia;carnivora;canidae;canis;latrans"


def photos_frame(rows: list[tuple[str, str, str]], valid: bool = True) -> pd.DataFrame:
    frame = pd.DataFrame(rows, columns=["photo_id", "camera_id", "timestamp"])
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    frame["rel_path"] = frame["photo_id"]
    frame["timestamp_valid"] = valid
    frame["decode_ok"] = True
    return frame


def test_gap_boundary_is_inclusive_and_gaps_chain() -> None:
    photos = photos_frame(
        [
            ("a", "c1", "2026-04-06 06:00:00"),
            ("b", "c1", "2026-04-06 06:01:00"),
            ("c", "c1", "2026-04-06 06:02:00"),
            ("d", "c1", "2026-04-06 06:03:01"),
        ]
    )
    ids = assign_events(photos, gap_s=60).tolist()
    assert ids == ["c1-e00001", "c1-e00001", "c1-e00001", "c1-e00002"]


def test_cameras_never_share_events_and_order_is_by_time() -> None:
    photos = photos_frame(
        [
            ("late", "c1", "2026-04-06 07:00:00"),
            ("early", "c1", "2026-04-06 06:00:00"),
            ("other", "c2", "2026-04-06 06:00:10"),
        ]
    )
    ids = dict(zip(photos["photo_id"], assign_events(photos, gap_s=60), strict=True))
    assert ids == {"early": "c1-e00001", "late": "c1-e00002", "other": "c2-e00001"}


def test_invalid_or_undecodable_photos_get_no_event() -> None:
    photos = photos_frame([("a", "c1", "2000-01-01 00:00:00"), ("b", "c1", "2026-04-06 06:00:00")])
    photos.loc[0, "timestamp_valid"] = False
    photos.loc[1, "decode_ok"] = False
    assert assign_events(photos, gap_s=60).isna().all()


def boxes(rows: list[tuple[str, int, str, float]]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "photo_id": p,
                "box_idx": i,
                "category": c,
                "conf": conf,
                "x": 0.1 * i,
                "y": 0.1,
                "w": 0.1,
                "h": 0.1,
                "model": "m",
            }
            for p, i, c, conf in rows
        ],
        columns=BOX_COLUMNS,
    )


def labels(rows: list[tuple[str, int, str, str, float]]) -> pd.DataFrame:
    records: list[dict[str, Any]] = [
        {"photo_id": p, "box_idx": i, "species": s, "lineage": lin, "species_score": score}
        for p, i, s, lin, score in rows
    ]
    frame = pd.DataFrame(records, columns=CLASSIFICATION_COLUMNS)
    return frame


def build(
    photo_rows: list[tuple[str, str, str]],
    box_rows: list[tuple[str, int, str, float]],
    label_rows: list[tuple[str, int, str, str, float]],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    media = photos_frame(photo_rows)
    quality = pd.DataFrame(
        {
            "photo_id": media["photo_id"],
            "decode_ok": True,
            "is_ir": False,
            "is_dark": False,
            "is_overexposed": False,
            "is_blurry": False,
            "is_near_duplicate": False,
        }
    )
    runs = pd.DataFrame(
        {"photo_id": media["photo_id"], "model": "m", "n_boxes": 0, "error": None},
        columns=RUN_COLUMNS,
    )
    box_table, label_table = boxes(box_rows), labels(label_rows)
    photos = build_photo_table(
        media.drop(columns=["decode_ok"]), quality, box_table, runs, label_table, 0.2
    )
    per_photo = photo_species(box_table, label_table, 0.2)
    return build_events(photos, per_photo, gap_s=60)


def test_event_count_is_max_per_frame_not_sum_over_frames() -> None:
    photos, events, event_species = build(
        [
            ("a", "c1", "2026-04-06 06:00:00"),
            ("b", "c1", "2026-04-06 06:00:01"),
            ("c", "c1", "2026-04-06 06:00:02"),
        ],
        [
            ("a", 0, "animal", 0.9),
            ("b", 0, "animal", 0.9),
            ("b", 1, "animal", 0.8),
            ("c", 0, "animal", 0.9),
            ("c", 1, "animal", 0.05),
        ],
        [
            ("a", 0, "white-tailed deer", DEER, 0.9),
            ("b", 0, "white-tailed deer", DEER, 0.8),
            ("b", 1, "white-tailed deer", DEER, 0.7),
            ("c", 0, "white-tailed deer", DEER, 0.95),
        ],
    )
    assert photos["n_animals"].tolist() == [1, 2, 1]
    assert len(events) == 1
    event = events.iloc[0]
    assert (event["species"], event["species_count"], event["max_animals"]) == (
        "white-tailed deer",
        2,
        2,
    )
    assert event["best_photo_id"] == "b"
    assert event["observation_type"] == "animal"
    assert event["duration_s"] == 2.0
    assert event_species[["species", "max_count", "n_photos"]].values.tolist() == [
        ["white-tailed deer", 2, 3]
    ]


def test_vaguer_labels_fold_into_the_single_species_of_the_event() -> None:
    _, _, event_species = build(
        [("a", "c1", "2026-04-06 06:00:00"), ("b", "c1", "2026-04-06 06:00:01")],
        [("a", 0, "animal", 0.9), ("b", 0, "animal", 0.9), ("b", 1, "animal", 0.6)],
        [("a", 0, "white-tailed deer", DEER, 0.9), ("b", 0, "cervidae (family)", FAMILY, 0.7)],
    )
    # b's second box was not classified; it counts as unresolved and folds into the deer too.
    assert event_species[["species", "max_count"]].values.tolist() == [["white-tailed deer", 2]]


def test_two_species_in_one_event_stay_separate() -> None:
    _, events, event_species = build(
        [("a", "c1", "2026-04-06 06:00:00"), ("b", "c1", "2026-04-06 06:00:05")],
        [("a", 0, "animal", 0.9), ("a", 1, "animal", 0.8), ("b", 0, "animal", 0.9)],
        [
            ("a", 0, "coyote", COYOTE, 0.9),
            ("a", 1, "white-tailed deer", DEER, 0.8),
            ("b", 0, "coyote", COYOTE, 0.9),
        ],
    )
    assert sorted(event_species["species"]) == ["coyote", "white-tailed deer"]
    assert events["species"].tolist() == ["coyote"]


def test_blank_person_and_vehicle_events() -> None:
    photos, events, event_species = build(
        [
            ("a", "c1", "2026-04-06 06:00:00"),
            ("b", "c1", "2026-04-06 08:00:00"),
            ("c", "c1", "2026-04-06 10:00:00"),
        ],
        [("b", 0, "person", 0.9), ("c", 0, "vehicle", 0.9), ("c", 1, "animal", 0.1)],
        [],
    )
    assert events["observation_type"].tolist() == ["blank", "person", "vehicle"]
    assert events["species"].isna().all()
    assert events["species_count"].tolist() == [0, 0, 0]
    assert event_species.empty
    assert photos["n_animals"].sum() == 0


def test_unclassified_animals_count_as_unresolved() -> None:
    _, events, _ = build([("a", "c1", "2026-04-06 06:00:00")], [("a", 0, "animal", 0.9)], [])
    assert events["species"].tolist() == [UNRESOLVED]


def test_single_animal_with_conflicting_labels_gets_their_common_taxon() -> None:
    opossum = "mammalia;didelphimorphia;didelphidae;didelphis;virginiana"
    _, events, event_species = build(
        [
            ("a", "c1", "2026-04-06 22:00:00"),
            ("b", "c1", "2026-04-06 22:00:01"),
            ("c", "c1", "2026-04-06 22:00:02"),
        ],
        [("a", 0, "animal", 0.9), ("b", 0, "animal", 0.9), ("c", 0, "animal", 0.9)],
        [
            ("a", 0, "virginia opossum", opossum, 0.7),
            ("b", 0, "carnivora (order)", "mammalia;carnivora", 0.7),
            ("c", 0, "mammalia (class)", "mammalia", 0.9),
        ],
    )
    assert event_species[["species", "lineage", "max_count"]].values.tolist() == [
        ["mammalia (class)", "mammalia", 1]
    ]
    assert events["species"].tolist() == ["mammalia (class)"]


def test_single_animal_consensus_can_name_a_taxon_not_given_by_any_frame() -> None:
    _, _, event_species = build(
        [("a", "c1", "2026-04-06 22:00:00"), ("b", "c1", "2026-04-06 22:00:01")],
        [("a", 0, "animal", 0.9), ("b", 0, "animal", 0.9)],
        [
            ("a", 0, "white-tailed deer", DEER, 0.7),
            ("b", 0, "elk", "mammalia;cetartiodactyla;cervidae;cervus;canadensis", 0.7),
        ],
    )
    assert event_species["species"].tolist() == ["cervidae (family)"]
