"""Command-line interface: ``run``, ``status``, ``demo``, ``synth`` and ``fetch_sample``.

Exit codes: 0 success, 1 a self-check failed, 2 usage or input error, 3 missing model package.
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Callable, Sequence
from dataclasses import replace
from datetime import date
from functools import partial
from pathlib import Path

import numpy as np

from . import __version__
from .classify import ClassifierFactory, SpeciesNetClassifier
from .demo import run_demo
from .detect import (
    DEFAULT_DETECTION_THRESHOLD,
    DEFAULT_MEGADETECTOR_MODEL,
    DetectorFactory,
    MegaDetector,
)
from .errors import ModelDependencyError, PipelineError
from .mock_models import MockClassifier, MockDetector
from .pipeline import STAGES, Pipeline
from .public_sample import SampleSettings, fetch_public_sample
from .settings import (
    ClassificationSettings,
    DetectionSettings,
    EventSettings,
    HealthSettings,
    PipelineSettings,
    QualitySettings,
)
from .status import RunStatus
from .synthetic import SyntheticSettings, generate_dataset, load_ground_truth

EXIT_OK = 0
EXIT_CHECK_FAILED = 1
EXIT_INPUT_ERROR = 2
EXIT_MISSING_DEPENDENCY = 3
DEFAULT_SEED = 7
DEFAULT_DEMO_OUT = Path("demo_output")


def positive_int(text: str) -> int:
    """Parse an integer that must be at least 1.

    Args:
        text: Command-line value.

    Returns:
        The integer.

    Raises:
        argparse.ArgumentTypeError: If the value is not a positive integer.
    """
    try:
        value = int(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{text!r} is not an integer") from exc
    if value < 1:
        raise argparse.ArgumentTypeError(f"{value} must be at least 1")
    return value


def non_negative_int(text: str) -> int:
    """Parse an integer that must be 0 or more.

    Args:
        text: Command-line value.

    Returns:
        The integer.

    Raises:
        argparse.ArgumentTypeError: If the value is not a non-negative integer.
    """
    try:
        value = int(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{text!r} is not an integer") from exc
    if value < 0:
        raise argparse.ArgumentTypeError(f"{value} must not be negative")
    return value


def positive_float(text: str) -> float:
    """Parse a number that must be greater than 0.

    Args:
        text: Command-line value.

    Returns:
        The number.

    Raises:
        argparse.ArgumentTypeError: If the value is not a positive number.
    """
    try:
        value = float(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{text!r} is not a number") from exc
    if not value > 0:
        raise argparse.ArgumentTypeError(f"{value} must be greater than 0")
    return value


def fraction(text: str) -> float:
    """Parse a number between 0 and 1 inclusive, such as a confidence.

    Args:
        text: Command-line value.

    Returns:
        The number.

    Raises:
        argparse.ArgumentTypeError: If the value is outside [0, 1].
    """
    try:
        value = float(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{text!r} is not a number") from exc
    if not 0.0 <= value <= 1.0:
        raise argparse.ArgumentTypeError(f"{value} must be between 0 and 1")
    return value


def iso_date(text: str) -> str:
    """Check that a value is an ISO date such as ``2024-06-01``.

    Args:
        text: Command-line value.

    Returns:
        The value unchanged.

    Raises:
        argparse.ArgumentTypeError: If the value is not an ISO date.
    """
    try:
        date.fromisoformat(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{text!r} is not an ISO date (YYYY-MM-DD)") from exc
    return text


def _add_run_parser(commands: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    run = commands.add_parser("run", help="run (or resume) the pipeline on a photo folder")
    run.add_argument("input", type=Path, help="folder of photos, one sub-folder per camera")
    run.add_argument("--out", type=Path, required=True, help="output folder for the run")
    _add_model_arguments(run)
    _add_threshold_arguments(run)
    run.add_argument(
        "--workers",
        type=positive_int,
        default=PipelineSettings.workers,
        help="threads for reading and decoding images",
    )
    run.add_argument("--title", default=PipelineSettings.report_title, help="report heading")
    run.add_argument(
        "--redo",
        action="append",
        default=[],
        choices=[*STAGES, "all"],
        help="recompute a stage and invalidate later ones (repeatable)",
    )


def _add_model_arguments(run: argparse.ArgumentParser) -> None:
    run.add_argument(
        "--detector",
        choices=["megadetector", "mock"],
        default="megadetector",
        help="megadetector, or mock for synthetic folders with ground_truth.json",
    )
    run.add_argument(
        "--detector-model",
        default=DEFAULT_MEGADETECTOR_MODEL,
        help="MegaDetector model name or weights file",
    )
    run.add_argument(
        "--detect-threshold",
        type=fraction,
        default=DEFAULT_DETECTION_THRESHOLD,
        help="lowest box confidence the detector keeps",
    )
    run.add_argument(
        "--classifier",
        choices=["speciesnet", "mock", "none"],
        default="speciesnet",
        help="speciesnet, mock for synthetic folders, or none to leave animals unresolved",
    )
    run.add_argument(
        "--classifier-model", default=None, help="SpeciesNet model name (package default)"
    )
    run.add_argument(
        "--device", choices=["auto", "cpu", "cuda"], default="auto", help="where models run"
    )
    run.add_argument(
        "--seed", type=non_negative_int, default=DEFAULT_SEED, help="seed for mock models"
    )


def _add_threshold_arguments(run: argparse.ArgumentParser) -> None:
    run.add_argument(
        "--count-conf",
        type=fraction,
        default=DetectionSettings.count_conf,
        help="boxes at or above this confidence are counted and classified",
    )
    run.add_argument(
        "--species-threshold",
        type=fraction,
        default=ClassificationSettings.species_threshold,
        help="minimum label score before rolling up the taxonomy",
    )
    run.add_argument(
        "--event-gap",
        type=positive_float,
        default=EventSettings.gap_s,
        help="seconds between photos that still belong to one event",
    )
    run.add_argument(
        "--gap-hours",
        type=positive_float,
        default=HealthSettings.gap_hours,
        help="silence (hours) reported as a possible outage",
    )
    run.add_argument(
        "--valid-from",
        type=iso_date,
        default=PipelineSettings.valid_from,
        help="ISO date; timestamps on or before it are treated as an unset clock",
    )
    run.add_argument(
        "--strip-top",
        type=fraction,
        default=QualitySettings.strip_top,
        help="fraction of the frame height to ignore at the top (info bar)",
    )
    run.add_argument(
        "--strip-bottom",
        type=fraction,
        default=QualitySettings.strip_bottom,
        help="fraction of the frame height to ignore at the bottom (info bar)",
    )


def _add_other_parsers(commands: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    status = commands.add_parser("status", help="show stage progress for an output folder")
    status.add_argument("out", type=Path, help="output folder of a run")

    demo = commands.add_parser("demo", help="synthetic end-to-end run with a self-check")
    demo.add_argument(
        "--out", type=Path, default=DEFAULT_DEMO_OUT, help="folder for the report, charts, CSVs"
    )
    demo.add_argument("--work", type=Path, default=None, help="keep photos and the full run here")
    demo.add_argument("--days", type=positive_int, default=SyntheticSettings.n_days, help="days")
    demo.add_argument(
        "--cameras", type=positive_int, default=SyntheticSettings.n_cameras, help="cameras"
    )
    demo.add_argument("--seed", type=non_negative_int, default=DEFAULT_SEED, help="random seed")
    demo.add_argument(
        "--simulate-crash", action="store_true", help="crash detection halfway, then resume"
    )

    synth = commands.add_parser("synth", help="write a synthetic photo folder only")
    synth.add_argument("out", type=Path, help="destination folder")
    synth.add_argument("--days", type=positive_int, default=SyntheticSettings.n_days, help="days")
    synth.add_argument(
        "--cameras", type=positive_int, default=SyntheticSettings.n_cameras, help="cameras"
    )
    synth.add_argument("--seed", type=non_negative_int, default=DEFAULT_SEED, help="random seed")

    fetch = commands.add_parser(
        "fetch_sample", help="download a small public camera-trap sample (network)"
    )
    fetch.add_argument("out", type=Path, help="destination folder")
    fetch.add_argument(
        "--cameras", type=positive_int, default=SampleSettings.n_cameras, help="camera locations"
    )
    fetch.add_argument(
        "--per-camera",
        type=positive_int,
        default=SampleSettings.photos_per_camera,
        help="photos per location, whole sequences kept together",
    )
    fetch.add_argument("--metadata-url", default=None, help="COCO Camera Traps JSON (https)")
    fetch.add_argument("--image-base-url", default=None, help="prefix for image files (https)")
    fetch.add_argument("--seed", type=non_negative_int, default=0, help="random seed")


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser.

    Returns:
        The parser with all subcommands.
    """
    parser = argparse.ArgumentParser(
        prog="camera_trap_pipeline",
        description="Turn folders of camera-trap photos into a clean, queryable dataset.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument(
        "-v", "--verbose", action="count", default=0, help="-v for progress, -vv for debug detail"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    _add_run_parser(commands)
    _add_other_parsers(commands)
    return parser


def _settings_from(args: argparse.Namespace) -> PipelineSettings:
    base = PipelineSettings()
    return replace(
        base,
        valid_from=args.valid_from,
        workers=args.workers,
        report_title=args.title,
        quality=replace(base.quality, strip_top=args.strip_top, strip_bottom=args.strip_bottom),
        detection=replace(base.detection, count_conf=args.count_conf),
        classification=replace(base.classification, species_threshold=args.species_threshold),
        events=EventSettings(gap_s=args.event_gap),
        health=HealthSettings(gap_hours=args.gap_hours),
    )


def _model_factories(
    args: argparse.Namespace,
) -> tuple[DetectorFactory, ClassifierFactory | None]:
    detector: DetectorFactory
    classifier: ClassifierFactory | None
    if args.detector == "mock":
        detector = partial(MockDetector, load_ground_truth(args.input), args.seed)
    else:
        detector = partial(
            MegaDetector, args.detector_model, args.detect_threshold, force_cpu=args.device == "cpu"
        )
    if args.classifier == "mock":
        classifier = partial(MockClassifier, load_ground_truth(args.input), args.seed)
    elif args.classifier == "speciesnet":
        device = None if args.device == "auto" else args.device
        classifier = partial(SpeciesNetClassifier, args.classifier_model, device)
    else:
        classifier = None
    return detector, classifier


def _print_status(status: RunStatus) -> None:
    for row in status.rows():
        details = {k: v for k, v in row.info.items() if k not in ("done", "started_at")}
        finished = details.pop("finished_at", "")
        extras = ", ".join(f"{k}={v}" for k, v in details.items())
        print(f"{row.stage:<10} {row.state:<11} {finished or '':<20} {extras}".rstrip())


def _cmd_run(args: argparse.Namespace) -> int:
    detector, classifier = _model_factories(args)
    pipeline = Pipeline(
        args.input,
        args.out,
        _settings_from(args),
        detector,
        classifier,
        synthetic=args.detector == "mock",
    )
    status = pipeline.run(redo=args.redo)
    _print_status(status)
    print(f"report: {args.out / 'report.html'}")
    return EXIT_OK


def _cmd_status(args: argparse.Namespace) -> int:
    if not (args.out / "status.json").exists():
        print(f"error: no status.json in {args.out}", file=sys.stderr)
        return EXIT_INPUT_ERROR
    _print_status(RunStatus(args.out / "status.json", STAGES))
    return EXIT_OK


def _cmd_demo(args: argparse.Namespace) -> int:
    synthetic = SyntheticSettings(n_cameras=args.cameras, n_days=args.days)
    result = run_demo(args.out, synthetic, args.seed, args.simulate_crash, args.work)
    print(f"generated {result.n_images} synthetic photos")
    if result.crashed_after is not None:
        print(f"detection crashed after {result.crashed_after} photos and resumed from checkpoint")
    print(f"wrote {len(result.published)} files to {args.out.as_posix()}")
    print("self-check against the synthetic ground truth:")
    for check in result.checks:
        print(f"  {'PASS' if check.passed else 'FAIL'}  {check.name}: {check.detail}")
    return EXIT_OK if result.passed else EXIT_CHECK_FAILED


def _cmd_synth(args: argparse.Namespace) -> int:
    settings = SyntheticSettings(n_cameras=args.cameras, n_days=args.days)
    dataset = generate_dataset(args.out, settings, np.random.default_rng(args.seed))
    print(f"wrote {dataset.n_images} photos and ground_truth.json to {args.out}")
    return EXIT_OK


def _cmd_fetch(args: argparse.Namespace) -> int:
    settings = SampleSettings(n_cameras=args.cameras, photos_per_camera=args.per_camera)
    urls = {}
    if args.metadata_url:
        urls["metadata_url"] = args.metadata_url
    if args.image_base_url:
        urls["image_base_url"] = args.image_base_url
    fetch_public_sample(args.out, settings, np.random.default_rng(args.seed), **urls)
    print(f"sample written to {args.out} (see attribution.txt for the license)")
    return EXIT_OK


COMMANDS: dict[str, Callable[[argparse.Namespace], int]] = {
    "run": _cmd_run,
    "status": _cmd_status,
    "demo": _cmd_demo,
    "synth": _cmd_synth,
    "fetch_sample": _cmd_fetch,
}


def main(argv: Sequence[str] | None = None) -> int:
    """Parse arguments and run a subcommand.

    Args:
        argv: Arguments without the program name; defaults to ``sys.argv[1:]``.

    Returns:
        Process exit code: 0 success, 1 self-check failed, 2 usage or input error,
        3 missing optional model package.
    """
    args = build_parser().parse_args(argv)
    levels = {0: logging.WARNING, 1: logging.INFO}
    logging.basicConfig(
        level=levels.get(args.verbose, logging.DEBUG),
        format="%(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
        force=True,
    )
    try:
        return COMMANDS[args.command](args)
    except ModelDependencyError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_MISSING_DEPENDENCY
    except PipelineError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_INPUT_ERROR
