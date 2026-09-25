"""Synthetic field schedule: species behaviour, camera deployments and when visits happen.

Visits follow each species' daily activity curve; injected problems are planned per camera.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta

import numpy as np

from ._types import FloatArray
from .errors import InputError

SUNRISE_HOUR = 6.5
SUNSET_HOUR = 19.5
MIN_FRAME_SIDE = 64
MAX_JPEG_QUALITY = 95
SECONDS_PER_HOUR = 3600
EXTRA_BURSTS_MEAN = 0.6
MAX_EXTRA_BURSTS = 2


@dataclass(frozen=True)
class SpeciesProfile:
    """How one synthetic species looks and behaves.

    Attributes:
        name: Common name.
        lineage: Class, order, family, genus and species joined with ``;``.
        color: Daylight body color.
        size: Body width as a fraction of frame width at mid-distance.
        group: Smallest and largest group size.
        visits_per_day: Mean visits per camera per day before habitat multipliers.
        peaks: Activity peaks as ``(hour, spread_hours)``; weights are circular Gaussians.
        confusers: Other labels (name, lineage) a classifier might give it, most likely first.
    """

    name: str
    lineage: str
    color: tuple[int, int, int]
    size: float
    group: tuple[int, int]
    visits_per_day: float
    peaks: tuple[tuple[float, float], ...]
    confusers: tuple[tuple[str, str], ...]


SPECIES: tuple[SpeciesProfile, ...] = (
    SpeciesProfile(
        "white-tailed deer",
        "mammalia;cetartiodactyla;cervidae;odocoileus;virginianus",
        (150, 105, 65),
        0.22,
        (1, 3),
        1.4,
        ((6.8, 1.5), (19.3, 1.5), (1.0, 3.0)),
        (
            ("mule deer", "mammalia;cetartiodactyla;cervidae;odocoileus;hemionus"),
            ("elk", "mammalia;cetartiodactyla;cervidae;cervus;canadensis"),
            ("domestic cattle", "mammalia;cetartiodactyla;bovidae;bos;taurus"),
        ),
    ),
    SpeciesProfile(
        "northern raccoon",
        "mammalia;carnivora;procyonidae;procyon;lotor",
        (120, 115, 110),
        0.11,
        (1, 2),
        1.0,
        ((0.5, 2.5), (22.5, 2.0)),
        (
            ("striped skunk", "mammalia;carnivora;mephitidae;mephitis;mephitis"),
            ("virginia opossum", "mammalia;didelphimorphia;didelphidae;didelphis;virginiana"),
            ("domestic cat", "mammalia;carnivora;felidae;felis;catus"),
        ),
    ),
    SpeciesProfile(
        "coyote",
        "mammalia;carnivora;canidae;canis;latrans",
        (160, 130, 95),
        0.15,
        (1, 2),
        0.7,
        ((22.0, 2.5), (5.0, 1.5)),
        (
            ("domestic dog", "mammalia;carnivora;canidae;canis;familiaris"),
            ("gray fox", "mammalia;carnivora;canidae;urocyon;cinereoargenteus"),
            ("bobcat", "mammalia;carnivora;felidae;lynx;rufus"),
        ),
    ),
    SpeciesProfile(
        "wild turkey",
        "aves;galliformes;phasianidae;meleagris;gallopavo",
        (70, 55, 45),
        0.10,
        (1, 5),
        0.8,
        ((8.0, 1.5), (16.5, 1.8)),
        (
            ("northern bobwhite", "aves;galliformes;odontophoridae;colinus;virginianus"),
            ("american crow", "aves;passeriformes;corvidae;corvus;brachyrhynchos"),
            ("bird", "aves"),
        ),
    ),
    SpeciesProfile(
        "virginia opossum",
        "mammalia;didelphimorphia;didelphidae;didelphis;virginiana",
        (185, 180, 170),
        0.10,
        (1, 1),
        0.7,
        ((23.0, 2.5),),
        (
            ("northern raccoon", "mammalia;carnivora;procyonidae;procyon;lotor"),
            ("domestic cat", "mammalia;carnivora;felidae;felis;catus"),
            ("striped skunk", "mammalia;carnivora;mephitidae;mephitis;mephitis"),
        ),
    ),
    SpeciesProfile(
        "bobcat",
        "mammalia;carnivora;felidae;lynx;rufus",
        (165, 125, 85),
        0.13,
        (1, 1),
        0.2,
        ((6.0, 1.5), (20.0, 2.0)),
        (
            ("domestic cat", "mammalia;carnivora;felidae;felis;catus"),
            ("mountain lion", "mammalia;carnivora;felidae;puma;concolor"),
            ("coyote", "mammalia;carnivora;canidae;canis;latrans"),
        ),
    ),
    SpeciesProfile(
        "eastern gray squirrel",
        "mammalia;rodentia;sciuridae;sciurus;carolinensis",
        (135, 130, 125),
        0.05,
        (1, 2),
        0.9,
        ((9.0, 1.5), (16.0, 1.5)),
        (
            ("eastern fox squirrel", "mammalia;rodentia;sciuridae;sciurus;niger"),
            ("eastern chipmunk", "mammalia;rodentia;sciuridae;tamias;striatus"),
            ("rodent", "mammalia;rodentia"),
        ),
    ),
)
SPECIES_BY_NAME = {profile.name: profile for profile in SPECIES}

# Habitat presets cycle across cameras so each camera sees a different mix.
HABITATS: tuple[dict[str, float], ...] = (
    {"white-tailed deer": 1.5, "wild turkey": 1.4, "eastern gray squirrel": 1.3},
    {"northern raccoon": 1.6, "virginia opossum": 1.5, "white-tailed deer": 0.6},
    {"coyote": 1.6, "bobcat": 2.5, "wild turkey": 0.4},
)


@dataclass(frozen=True)
class SyntheticSettings:
    """Size and content of a synthetic dataset.

    Attributes:
        n_cameras: Number of cameras (one folder each).
        n_days: Days of deployment.
        start: First deployment day (ISO date).
        width: Frame width in pixels.
        height: Frame height in pixels.
        burst_size: Photos per trigger.
        rate_scale: Multiplier on every visit rate.
        wind_triggers_per_day: Empty daytime triggers per camera per day.
        people_per_day: Person visits per camera per day.
        vehicles_per_day: Vehicle visits per camera per day.
        min_visit_spacing_s: Minimum time between visits at one camera, so ground-truth
            visits map one-to-one onto events.
        info_strip: Height of the burned-in info bar as a fraction of the frame.
        jpeg_quality: JPEG quality for saved frames.
        inject_problems: Add the unset clock, outage, fog and truncated file.
    """

    n_cameras: int = 3
    n_days: int = 10
    start: str = "2026-04-06"
    width: int = 384
    height: int = 256
    burst_size: int = 3
    rate_scale: float = 1.0
    wind_triggers_per_day: float = 1.5
    people_per_day: float = 0.12
    vehicles_per_day: float = 0.06
    min_visit_spacing_s: int = 600
    info_strip: float = 0.07
    jpeg_quality: int = 80
    inject_problems: bool = True

    def __post_init__(self) -> None:
        """Reject settings that cannot produce a dataset.

        Raises:
            InputError: If a count or size is out of range.
        """
        problems = [
            (self.n_cameras < 1, "n_cameras must be at least 1"),
            (self.n_days < 1, "n_days must be at least 1"),
            (self.burst_size < 1, "burst_size must be at least 1"),
            (min(self.width, self.height) < MIN_FRAME_SIDE, f"frames need {MIN_FRAME_SIDE}+ px"),
            (not 0 <= self.info_strip < 0.5, "info_strip must be in [0, 0.5)"),
            (not 1 <= self.jpeg_quality <= MAX_JPEG_QUALITY, "jpeg_quality must be 1-95"),
            (self.rate_scale < 0, "rate_scale must not be negative"),
        ]
        for failed, message in problems:
            if failed:
                raise InputError(message)


@dataclass(frozen=True)
class CameraPlan:
    """Per-camera deployment facts, including injected problems.

    Attributes:
        camera_id: Folder name.
        habitat: Species rate multipliers.
        outage_days: ``(first_day, last_day_exclusive)`` with no photos, or None.
        unset_clock_days: Leading days during which the clock reads from 2000-01-01.
        fog_day: Day whose early-morning frames are blurred, or None.
    """

    camera_id: str
    habitat: dict[str, float] = field(hash=False)
    outage_days: tuple[int, int] | None = None
    unset_clock_days: int = 0
    fog_day: int | None = None


@dataclass
class Visit:
    """One ground-truth visit (becomes one event).

    Attributes:
        visit_id: Sequential id across the dataset.
        camera_id: Camera.
        kind: ``animal``, ``person``, ``vehicle`` or ``empty``.
        start: True start time.
        species: Species name for animal visits.
        count: Number of animals.
        n_photos: Photos written for the visit.
        valid_clock: False when the camera clock was unset during the visit.
    """

    visit_id: int
    camera_id: str
    kind: str
    start: datetime
    species: str | None = None
    count: int = 0
    n_photos: int = 0
    valid_clock: bool = True


def plan_cameras(settings: SyntheticSettings) -> list[CameraPlan]:
    """Assign habitats and injected problems to cameras.

    Args:
        settings: Dataset settings.

    Returns:
        One plan per camera, named ``cam_01``, ``cam_02`` and so on.
    """
    n_cameras, n_days = settings.n_cameras, settings.n_days
    problems = settings.inject_problems
    # Problems go to different cameras when there are enough of them.
    fog_camera, clock_camera, outage_camera = 0, 1 % n_cameras, 2 % n_cameras
    outage_start = max(1, round(n_days * 0.4))
    outage_end = outage_start + max(2, round(n_days * 0.3))
    has_outage = problems and outage_end < n_days
    unset_days = min(2, max(1, n_days // 4))
    return [
        CameraPlan(
            camera_id=f"cam_{index + 1:02d}",
            habitat=HABITATS[index % len(HABITATS)],
            outage_days=(
                (outage_start, outage_end) if has_outage and index == outage_camera else None
            ),
            unset_clock_days=unset_days if problems and index == clock_camera else 0,
            fog_day=1 if problems and index == fog_camera and n_days > 1 else None,
        )
        for index in range(n_cameras)
    ]


def diel_weights(profile: SpeciesProfile) -> FloatArray:
    """Compute the chance of a visit starting in each hour from the activity peaks.

    Args:
        profile: Species profile.

    Returns:
        24 probabilities that sum to 1.
    """
    hours = np.arange(24, dtype=np.float64) + 0.5
    weights = np.full(24, 0.02)
    for peak, spread in profile.peaks:
        distance = np.minimum(np.abs(hours - peak), 24 - np.abs(hours - peak))
        weights += np.exp(-0.5 * (distance / spread) ** 2)
    normalized: FloatArray = weights / weights.sum()
    return normalized


def is_night(moment: datetime) -> bool:
    """Tell whether a moment falls between the fixed synthetic sunset and sunrise.

    Args:
        moment: Time of day.

    Returns:
        True at night.
    """
    hour = moment.hour + moment.minute / 60
    return hour < SUNRISE_HOUR or hour >= SUNSET_HOUR


def burst_count(visit: Visit, rng: np.random.Generator) -> int:
    """Decide how many trigger bursts a visit produces.

    Args:
        visit: The visit.
        rng: Random generator.

    Returns:
        1 for empty, person and vehicle visits; 1 to 3 for animals that linger.
    """
    if visit.kind != "animal":
        return 1
    return 1 + min(int(rng.poisson(EXTRA_BURSTS_MEAN)), MAX_EXTRA_BURSTS)


def _visits_for_day(
    plan: CameraPlan,
    day_start: datetime,
    settings: SyntheticSettings,
    rng: np.random.Generator,
) -> list[Visit]:
    visits = []
    for profile in SPECIES:
        rate = profile.visits_per_day * plan.habitat.get(profile.name, 1.0) * settings.rate_scale
        for _ in range(int(rng.poisson(rate))):
            hour = int(rng.choice(24, p=diel_weights(profile))) + float(rng.random())
            count = int(rng.integers(profile.group[0], profile.group[1] + 1))
            start = day_start + timedelta(seconds=round(hour * SECONDS_PER_HOUR))
            visits.append(Visit(0, plan.camera_id, "animal", start, profile.name, count))
    extras = (
        ("empty", settings.wind_triggers_per_day * settings.rate_scale, (8.0, 18.0)),
        ("person", settings.people_per_day, (9.0, 17.0)),
        ("vehicle", settings.vehicles_per_day, (8.0, 18.0)),
    )
    for kind, rate, (first, last) in extras:
        for _ in range(int(rng.poisson(rate))):
            offset = round(float(rng.uniform(first, last)) * SECONDS_PER_HOUR)
            start = day_start + timedelta(seconds=offset)
            visits.append(Visit(0, plan.camera_id, kind, start, None, 0 if kind == "empty" else 1))
    return visits


def schedule_visits(
    plan: CameraPlan, settings: SyntheticSettings, rng: np.random.Generator
) -> list[Visit]:
    """Draw every visit to one camera, skipping outage days and keeping visits apart.

    Args:
        plan: The camera's plan.
        settings: Dataset settings.
        rng: Random generator.

    Returns:
        Visits in time order, at least ``settings.min_visit_spacing_s`` apart.
    """
    start = datetime.fromisoformat(settings.start)
    visits: list[Visit] = []
    for day in range(settings.n_days):
        if plan.outage_days and plan.outage_days[0] <= day < plan.outage_days[1]:
            continue
        visits += _visits_for_day(plan, start + timedelta(days=day), settings, rng)
    kept: list[Visit] = []
    for visit in sorted(visits, key=lambda v: v.start):
        if (
            not kept
            or (visit.start - kept[-1].start).total_seconds() >= settings.min_visit_spacing_s
        ):
            kept.append(visit)
    return kept
