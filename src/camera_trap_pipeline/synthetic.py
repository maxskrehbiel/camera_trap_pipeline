"""Synthetic trail-camera datasets with EXIF and ground truth, for tests and the demo.

Field problems are injected on purpose: unset clock, outage, fog, wind triggers, truncation.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from functools import cached_property
from pathlib import Path
from typing import TypedDict, cast

import numpy as np

from .atomic_io import atomic_write_json
from .errors import InputError
from .synthetic_scene import (
    Scene,
    TruthObject,
    exif_for,
    make_scene,
    render_frame,
    step_subject,
    subjects_for,
)
from .synthetic_schedule import (
    CameraPlan,
    SyntheticSettings,
    Visit,
    burst_count,
    is_night,
    plan_cameras,
    schedule_visits,
)

__all__ = [
    "GROUND_TRUTH_NAME",
    "GroundTruth",
    "SyntheticDataset",
    "SyntheticSettings",
    "generate_dataset",
    "load_ground_truth",
]

GROUND_TRUTH_NAME = "ground_truth.json"
CLOCK_RESET = datetime(2000, 1, 1)
FOG_BEFORE_HOUR = 10
TRUNCATED_FRACTION = 0.4


class TruthImage(TypedDict):
    """Ground truth for one image file."""

    camera_id: str
    time: str
    camera_clock: str
    visit_id: int
    kind: str
    night: bool
    blurred: bool
    truncated: bool
    objects: list[TruthObject]


class TruthVisit(TypedDict):
    """Ground truth for one visit, as stored in ``ground_truth.json``."""

    visit_id: int
    camera_id: str
    kind: str
    start: str
    species: str | None
    count: int
    n_photos: int
    valid_clock: bool


class TruthCamera(TypedDict):
    """Ground truth for one camera's deployment and injected problems."""

    camera_id: str
    habitat: dict[str, float]
    outage_days: list[int] | None
    unset_clock_days: int
    fog_day: int | None


@dataclass(frozen=True)
class SyntheticDataset:
    """What ``generate_dataset`` wrote.

    Attributes:
        root: Folder containing one sub-folder per camera and ``ground_truth.json``.
        n_images: Number of image files written.
        visits: Ground-truth visits.
    """

    root: Path
    n_images: int
    visits: tuple[Visit, ...]


@dataclass
class _CameraWriter:
    """Writes the frames of one camera and collects their ground truth."""

    root: Path
    settings: SyntheticSettings
    plan: CameraPlan
    scene: Scene
    images: dict[str, TruthImage]
    frames_written: int = 0

    def write_visit(self, visit: Visit, rng: np.random.Generator) -> None:
        """Render and save every frame of one visit, recording its ground truth.

        Args:
            visit: The visit; its photo count and clock validity are updated in place.
            rng: Random generator.
        """
        start = datetime.fromisoformat(self.settings.start)
        subjects = subjects_for(visit, self.settings, rng)
        moment = visit.start
        for burst in range(burst_count(visit, rng)):
            if burst:
                moment += timedelta(seconds=int(rng.integers(10, 41)))
            for shot in range(self.settings.burst_size):
                frame_time = moment + timedelta(seconds=shot)
                day_index = (frame_time.date() - start.date()).days
                unset = day_index < self.plan.unset_clock_days
                clock = CLOCK_RESET + (frame_time - start) if unset else frame_time
                blurred = self.plan.fog_day == day_index and frame_time.hour < FOG_BEFORE_HOUR
                image, objects = render_frame(
                    self.scene,
                    self.settings,
                    frame_time,
                    clock,
                    self.plan.camera_id,
                    subjects,
                    rng,
                    blurred,
                )
                for subject in subjects:
                    step_subject(subject, self.settings.width)
                self.frames_written += 1
                rel_path = f"{self.plan.camera_id}/img_{self.frames_written:05d}.jpg"
                exif = exif_for(clock, self.plan.camera_id)
                image.save(
                    self.root / rel_path, "JPEG", quality=self.settings.jpeg_quality, exif=exif
                )
                visit.n_photos += 1
                visit.valid_clock = visit.valid_clock and not unset
                self.images[rel_path] = {
                    "camera_id": self.plan.camera_id,
                    "time": frame_time.isoformat(),
                    "camera_clock": clock.isoformat(),
                    "visit_id": visit.visit_id,
                    "kind": visit.kind,
                    "night": is_night(frame_time),
                    "blurred": blurred,
                    "truncated": False,
                    "objects": objects,
                }


def generate_dataset(
    root: Path, settings: SyntheticSettings, rng: np.random.Generator
) -> SyntheticDataset:
    """Write a synthetic dataset: ``<root>/<camera_id>/img_NNNNN.jpg`` plus ground truth.

    Args:
        root: Output folder (created if missing).
        settings: Dataset settings.
        rng: Random generator; the same seed gives the same dataset.

    Returns:
        Summary of what was written.
    """
    root.mkdir(parents=True, exist_ok=True)
    images: dict[str, TruthImage] = {}
    all_visits: list[Visit] = []
    plans = plan_cameras(settings)
    for plan in plans:
        writer = _CameraWriter(root, settings, plan, make_scene(settings, rng), images)
        (root / plan.camera_id).mkdir(parents=True, exist_ok=True)
        for visit in schedule_visits(plan, settings, rng):
            visit.visit_id = len(all_visits) + 1
            all_visits.append(visit)
            writer.write_visit(visit, rng)
    if settings.inject_problems:
        _truncate_last_image(root, images, plans[0].camera_id)
    atomic_write_json(
        root / GROUND_TRUTH_NAME,
        {
            "generator": "camera_trap_pipeline.synthetic",
            "settings": asdict(settings),
            "cameras": [asdict(p) for p in plans],
            "visits": [{**asdict(v), "start": v.start.isoformat()} for v in all_visits],
            "images": images,
        },
    )
    return SyntheticDataset(root=root, n_images=len(images), visits=tuple(all_visits))


def _truncate_last_image(root: Path, images: dict[str, TruthImage], camera_id: str) -> None:
    """Cut the camera's last file short, like an upload that stopped midway."""
    own = [p for p in images if p.startswith(f"{camera_id}/")]
    if not own:
        return
    last = max(own)
    path = root / last
    data = path.read_bytes()
    path.write_bytes(data[: int(len(data) * TRUNCATED_FRACTION)])
    images[last]["truncated"] = True


@dataclass(frozen=True)
class GroundTruth:
    """Parsed ``ground_truth.json``.

    Attributes:
        root: Dataset folder.
        images: Per-image truth keyed by POSIX path relative to ``root``.
        visits: Ground-truth visits.
        cameras: Ground-truth camera plans.
    """

    root: Path
    images: dict[str, TruthImage]
    visits: list[TruthVisit]
    cameras: list[TruthCamera]

    def entry_for(self, path: str | Path) -> TruthImage | None:
        """Look up the truth for an image path (absolute or relative to ``root``).

        Args:
            path: Image path.

        Returns:
            The image's truth, or None for a file that is not part of the dataset.
        """
        candidate = Path(path)
        if candidate.is_relative_to(self.root):
            return self.images.get(candidate.relative_to(self.root).as_posix())
        if candidate.is_absolute() and candidate.is_relative_to(self.resolved_root):
            return self.images.get(candidate.relative_to(self.resolved_root).as_posix())
        return self.images.get(candidate.as_posix())

    @cached_property
    def resolved_root(self) -> Path:
        """Absolute form of ``root``, computed once.

        Returns:
            ``root`` with links resolved.
        """
        return self.root.resolve()


def load_ground_truth(root: Path) -> GroundTruth:
    """Read the ground truth written by ``generate_dataset``.

    Args:
        root: Dataset folder.

    Returns:
        The parsed ground truth.

    Raises:
        InputError: If the folder has no ``ground_truth.json``.
    """
    path = root / GROUND_TRUTH_NAME
    if not path.exists():
        raise InputError(
            f"{GROUND_TRUTH_NAME} not found in {root}; mock models only work on synthetic data"
        )
    data = json.loads(path.read_text(encoding="utf-8"))
    return GroundTruth(
        root=root,
        images=cast(dict[str, TruthImage], data["images"]),
        visits=cast(list[TruthVisit], data["visits"]),
        cameras=cast(list[TruthCamera], data["cameras"]),
    )
