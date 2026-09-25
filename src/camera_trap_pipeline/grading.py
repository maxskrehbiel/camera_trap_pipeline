"""Self-check for synthetic runs: compare a finished pipeline run with the ground truth."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .synthetic import GroundTruth
from .synthetic_schedule import SPECIES_BY_NAME

KIND_TO_OBSERVATION = {
    "animal": "animal",
    "person": "person",
    "vehicle": "vehicle",
    "empty": "blank",
}


@dataclass(frozen=True)
class GradingThresholds:
    """Pass marks for the checks that allow some error.

    Attributes:
        species_agreement: Share of animal events whose species call equals the true species
            or one of its ancestors. The stand-in classifier errs at night on purpose.
        blur_recall: Share of fogged frames that must be flagged blurry.
        blur_false_rate: Largest share of clear frames that may be flagged blurry.
        duplicate_recall: Share of repeated frames of empty triggers that must be flagged as
            near-duplicates.
    """

    species_agreement: float = 0.9
    blur_recall: float = 0.8
    blur_false_rate: float = 0.05
    duplicate_recall: float = 0.9


@dataclass(frozen=True)
class Check:
    """One self-check result.

    Attributes:
        name: What was checked.
        passed: Whether the output met the expectation.
        detail: Counts behind the verdict.
    """

    name: str
    passed: bool
    detail: str


def _share(numerator: int, denominator: int) -> str:
    percent = 100.0 * numerator / denominator if denominator else 100.0
    return f"{numerator}/{denominator} ({percent:.1f}%)"


def _photos_with_truth(run_dir: Path, truth: GroundTruth) -> pd.DataFrame:
    photos = pd.read_parquet(run_dir / "photos.parquet")
    entries = photos["rel_path"].map(truth.images.__getitem__)
    for field in ("visit_id", "kind", "night", "blurred", "truncated", "camera_clock"):
        photos[f"truth_{field}"] = entries.map(lambda entry, name=field: entry[name])
    return photos


def _check_events(photos: pd.DataFrame, events: pd.DataFrame, truth: GroundTruth) -> list[Check]:
    in_events = photos.dropna(subset=["event_id"])
    visits_per_event = in_events.groupby("event_id")["truth_visit_id"].nunique()
    events_per_visit = in_events.groupby("truth_visit_id")["event_id"].nunique()
    expected = {v["visit_id"] for v in truth.visits if v["valid_clock"]}
    one_to_one = (
        bool((visits_per_event == 1).all())
        and bool((events_per_visit == 1).all())
        and set(in_events["truth_visit_id"]) == expected
    )
    visit_of_event = in_events.groupby("event_id")["truth_visit_id"].first()
    kinds = {v["visit_id"]: v["kind"] for v in truth.visits}
    observed = dict(zip(events["event_id"], events["observation_type"], strict=True))
    matching_types = sum(
        observed.get(str(event)) == KIND_TO_OBSERVATION[kinds[int(visit)]]
        for event, visit in visit_of_event.items()
    )
    return [
        Check(
            "events match visits",
            one_to_one,
            f"{len(events)} events for {len(expected)} visits with a valid clock",
        ),
        Check(
            "event types",
            matching_types == len(visit_of_event),
            _share(matching_types, len(visit_of_event)) + " match the visit kind",
        ),
    ]


def _check_animals(
    photos: pd.DataFrame,
    events: pd.DataFrame,
    event_species: pd.DataFrame,
    truth: GroundTruth,
    thresholds: GradingThresholds,
) -> list[Check]:
    in_events = photos.dropna(subset=["event_id"])
    visit_of_event = dict(zip(in_events["event_id"], in_events["truth_visit_id"], strict=True))
    visits = {v["visit_id"]: v for v in truth.visits}
    lineage_of = event_species.set_index(["event_id", "species"])["lineage"]
    counted = agreed = n_animal = 0
    for event in events[events["observation_type"] == "animal"].itertuples(index=False):
        n_animal += 1
        visit = visits.get(int(visit_of_event.get(event.event_id, -1)))
        if visit is None or visit["species"] is None:
            continue
        counted += int(event.max_animals == visit["count"])
        true_lineage = SPECIES_BY_NAME[visit["species"]].lineage
        called = str(lineage_of.get((event.event_id, event.species), ""))
        agreed += int(true_lineage == called or true_lineage.startswith(called + ";"))
    return [
        Check("animal counts", counted == n_animal, _share(counted, n_animal) + " exact"),
        Check(
            "species calls",
            n_animal > 0 and agreed / n_animal >= thresholds.species_agreement,
            _share(agreed, n_animal) + " consistent with the true species",
        ),
    ]


def _check_quality(photos: pd.DataFrame, thresholds: GradingThresholds) -> list[Check]:
    unset = photos["truth_camera_clock"] < "2001"
    wrong_clock = int((photos["timestamp_valid"] == unset).sum())
    truncated = photos["truth_truncated"].astype(bool)
    wrong_decode = int((photos["decode_ok"] == truncated).sum())
    decoded = photos[photos["decode_ok"]]
    wrong_ir = int((decoded["is_ir"] != decoded["truth_night"]).sum())
    fogged = decoded["truth_blurred"].astype(bool)
    found = int(decoded.loc[fogged, "is_blurry"].sum())
    false_blur = int(decoded.loc[~fogged, "is_blurry"].sum())
    return [
        Check(
            "unset clocks",
            wrong_clock == 0,
            f"{int(unset.sum())} unset-clock photos, {wrong_clock} misflagged",
        ),
        Check(
            "truncated uploads",
            wrong_decode == 0,
            f"{int(truncated.sum())} truncated, {wrong_decode} misflagged",
        ),
        Check("night frames", wrong_ir == 0, f"{wrong_ir} of {len(decoded)} IR flags wrong"),
        Check(
            "fogged frames",
            found >= thresholds.blur_recall * int(fogged.sum())
            and false_blur <= thresholds.blur_false_rate * int((~fogged).sum()),
            f"{_share(found, int(fogged.sum()))} found, {false_blur} clear frames flagged",
        ),
    ]


def _check_duplicates(photos: pd.DataFrame, thresholds: GradingThresholds) -> list[Check]:
    decoded = photos[photos["decode_ok"]].sort_values(["camera_id", "timestamp", "rel_path"])
    repeat = decoded["truth_visit_id"] == decoded.groupby("camera_id")["truth_visit_id"].shift()
    empty_repeats = decoded[repeat & (decoded["truth_kind"] == "empty")]
    found = int(empty_repeats["is_near_duplicate"].sum())
    visit_of_photo = dict(zip(decoded["photo_id"], decoded["truth_visit_id"], strict=True))
    crossing = sum(
        visit_of_photo.get(group) != visit
        for group, visit in zip(decoded["dup_group"], decoded["truth_visit_id"], strict=True)
    )
    return [
        Check(
            "near-duplicates",
            found >= thresholds.duplicate_recall * len(empty_repeats) and crossing == 0,
            f"{_share(found, len(empty_repeats))} repeated empty frames found, "
            f"{crossing} groups span two visits",
        )
    ]


def _check_outage(run_dir: Path, truth: GroundTruth) -> list[Check]:
    gaps = pd.read_parquet(run_dir / "summary" / "camera_gaps.parquet")
    expected = sorted(c["camera_id"] for c in truth.cameras if c["outage_days"])
    found = sorted(gaps["camera_id"])
    detail = (
        f"long silences on {', '.join(found) or 'no camera'}; "
        f"outage planned on {', '.join(expected) or 'no camera'}"
    )
    return [Check("camera outage", found == expected, detail)]


def grade_run(
    run_dir: Path, truth: GroundTruth, thresholds: GradingThresholds | None = None
) -> list[Check]:
    """Check a finished run against the synthetic ground truth it was made from.

    Args:
        run_dir: Pipeline output folder.
        truth: Ground truth of the input dataset.
        thresholds: Pass marks; defaults to ``GradingThresholds()``.

    Returns:
        One result per check, in a fixed order.
    """
    limits = thresholds or GradingThresholds()
    photos = _photos_with_truth(run_dir, truth)
    events = pd.read_parquet(run_dir / "events.parquet")
    event_species = pd.read_parquet(run_dir / "event_species.parquet")
    return [
        *_check_events(photos, events, truth),
        *_check_animals(photos, events, event_species, truth, limits),
        *_check_quality(photos, limits),
        *_check_duplicates(photos, limits),
        *_check_outage(run_dir, truth),
    ]
