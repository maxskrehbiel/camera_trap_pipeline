"""End-to-end demo: synthetic photos, stand-in models, the full pipeline and a self-check."""

from __future__ import annotations

import logging
import shutil
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .errors import InputError
from .grading import Check, grade_run
from .mock_models import CrashingDetector, MockClassifier, MockDetector, SimulatedCrash
from .pipeline import Pipeline
from .settings import PipelineSettings, QualitySettings
from .synthetic import SyntheticSettings, generate_dataset, load_ground_truth

logger = logging.getLogger(__name__)

WORK_MARKER = ".camera_trap_pipeline_demo"
DEMO_TITLE = "Camera-trap pipeline report (synthetic demo)"
EXAMPLE_TABLES = (
    "species_totals",
    "activity_by_hour",
    "species_by_camera",
    "camera_health",
    "camera_gaps",
)


@dataclass(frozen=True)
class DemoResult:
    """What a demo run produced.

    Attributes:
        n_images: Synthetic photos generated.
        crashed_after: Photos detected before the simulated crash, or None.
        published: Files written to the output folder.
        checks: Self-check results against the synthetic ground truth.
    """

    n_images: int
    crashed_after: int | None
    published: tuple[Path, ...]
    checks: tuple[Check, ...]

    @property
    def passed(self) -> bool:
        """Whether every self-check passed.

        Returns:
            True when no check failed.
        """
        return all(check.passed for check in self.checks)


def prepare_work_dir(work_dir: Path) -> None:
    """Empty a previous demo work folder, refusing to touch anything else.

    Args:
        work_dir: Folder for generated photos and the full run.

    Raises:
        InputError: If the folder exists, is not empty and was not made by the demo.
    """
    if work_dir.exists():
        if not (work_dir / WORK_MARKER).exists() and any(work_dir.iterdir()):
            raise InputError(
                f"{work_dir} exists and was not created by the demo; choose another --work folder"
            )
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True)
    (work_dir / WORK_MARKER).write_text("created by camera_trap_pipeline demo\n", encoding="utf-8")


def demo_settings() -> PipelineSettings:
    """Build pipeline settings for the synthetic frames.

    Returns:
        Default settings that also ignore the burned-in info bar at the bottom of each frame.
    """
    return PipelineSettings(report_title=DEMO_TITLE, quality=QualitySettings(strip_bottom=0.07))


def publish(run_dir: Path, out_dir: Path) -> tuple[Path, ...]:
    """Copy the report, charts and summary tables (as CSV) to ``out_dir``.

    Args:
        run_dir: Pipeline output folder.
        out_dir: Destination, ``demo_output`` by default or ``examples`` for the repository.

    Returns:
        The written files.
    """
    written = [out_dir / "report.html"]
    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(run_dir / "report.html", written[0])
    for png in sorted((run_dir / "plots").glob("*.png")):
        target = out_dir / "plots" / png.name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(png, target)
        written.append(target)
    for name in EXAMPLE_TABLES:
        target = out_dir / "summary" / f"{name}.csv"
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(run_dir / "summary" / f"{name}.csv", target)
        written.append(target)
    events = pd.read_parquet(run_dir / "events.parquet")
    target = out_dir / "summary" / "events.csv"
    events.to_csv(target, index=False, lineterminator="\n")
    written.append(target)
    return tuple(written)


@contextmanager
def _work_dir(requested: Path | None) -> Iterator[Path]:
    if requested is not None:
        prepare_work_dir(requested)
        yield requested
        return
    with tempfile.TemporaryDirectory(prefix="camera_trap_pipeline_demo_") as scratch:
        yield Path(scratch)


def run_demo(
    out_dir: Path,
    synthetic: SyntheticSettings,
    seed: int,
    simulate_crash: bool = False,
    work_dir: Path | None = None,
) -> DemoResult:
    """Generate synthetic photos, run the pipeline with stand-in models, publish and self-check.

    Args:
        out_dir: Where the small publishable outputs go.
        synthetic: Dataset size and content.
        seed: Seed for the dataset and the stand-in models.
        simulate_crash: Crash detection halfway, then resume, to show checkpointing.
        work_dir: Keep the photos and the full run here; a temporary folder by default.

    Returns:
        A summary of the run, including the self-check results.
    """
    with _work_dir(work_dir) as work:
        photos_dir, run_dir = work / "photos", work / "run"
        dataset = generate_dataset(photos_dir, synthetic, np.random.default_rng(seed))
        logger.info("generated %d synthetic photos", dataset.n_images)
        truth = load_ground_truth(photos_dir)
        crashed_after = None
        if simulate_crash:
            crashed_after = dataset.n_images // 2
            crashing = Pipeline(
                photos_dir,
                run_dir,
                demo_settings(),
                lambda: CrashingDetector(truth, seed, dataset.n_images // 2),
                lambda: MockClassifier(truth, seed),
                synthetic=True,
            )
            try:
                crashing.run()
            except SimulatedCrash:
                logger.warning("simulated crash after %d photos; resuming", crashed_after)
        Pipeline(
            photos_dir,
            run_dir,
            demo_settings(),
            lambda: MockDetector(truth, seed),
            lambda: MockClassifier(truth, seed),
            synthetic=True,
        ).run()
        checks = tuple(grade_run(run_dir, truth))
        return DemoResult(dataset.n_images, crashed_after, publish(run_dir, out_dir), checks)
