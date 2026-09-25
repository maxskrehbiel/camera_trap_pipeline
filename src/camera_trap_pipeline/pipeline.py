"""Stage orchestration: run each stage in order, skip finished ones, and honor ``--redo``."""

from __future__ import annotations

import logging
import shutil
from collections.abc import Callable, Iterable
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd

from . import __version__
from .atomic_io import atomic_write_json, write_parquet
from .classify import ClassifierFactory, run_classify
from .detect import DetectorFactory, run_detect
from .errors import InputError
from .events import build_events, build_photo_table, photo_species
from .ingest import find_new_images, ingest
from .plots import render_all
from .quality import run_quality
from .report import OutputFile, write_report
from .settings import PipelineSettings
from .status import RunStatus
from .summarize import build_summaries, overview

logger = logging.getLogger(__name__)

STAGES = ("ingest", "quality", "detect", "classify", "events", "summarize", "report")

# Files each stage owns. ``redo`` deletes a stage's own files so it recomputes from scratch;
# later stages only lose their done-flag and reuse checkpoint entries whose inputs match.
STAGE_FILES: dict[str, tuple[str, ...]] = {
    "ingest": ("media.parquet",),
    "quality": ("checkpoints/quality.jsonl", "quality.parquet"),
    "detect": ("checkpoints/detections.jsonl", "detections.parquet", "detection_runs.parquet"),
    "classify": ("checkpoints/classifications.jsonl", "classifications.parquet"),
    "events": ("photos.parquet", "events.parquet", "event_species.parquet"),
    "summarize": ("summary", "plots"),
    "report": ("report.html",),
}

OUTPUT_DESCRIPTIONS = {
    "media.parquet": ("photo", "Path, camera, timestamp and its source, EXIF make/model"),
    "quality.parquet": ("photo", "Sharpness, brightness, IR flag, dhash, near-duplicate group"),
    "detections.parquet": ("box", "Category, confidence and normalized box from the detector"),
    "classifications.parquet": ("animal box", "Species call after roll-up, top-5 labels"),
    "photos.parquet": ("photo", "Everything above joined, plus counts and event id"),
    "events.parquet": ("event", "Burst of photos from one camera: time span, counts, species"),
    "event_species.parquet": ("event and species", "Largest single-frame count per species"),
}


class Pipeline:
    """A resumable run over one input folder, writing everything to one output folder."""

    def __init__(
        self,
        input_dir: Path,
        out_dir: Path,
        settings: PipelineSettings,
        detector_factory: DetectorFactory | None,
        classifier_factory: ClassifierFactory | None,
        synthetic: bool = False,
    ) -> None:
        """Configure a run; nothing is read or loaded until ``run``.

        Args:
            input_dir: Folder of photos (optionally with ``metadata.csv``).
            out_dir: Folder for checkpoints, tables, plots and the report.
            settings: Stage settings.
            detector_factory: Builds the detector when detection has work to do.
            classifier_factory: Builds the classifier; None leaves animals unresolved.
            synthetic: Mark the report as generated from synthetic data.
        """
        self.input_dir = input_dir
        self.out_dir = out_dir
        self.settings = settings
        self.detector_factory = detector_factory
        self.classifier_factory = classifier_factory
        self.synthetic = synthetic
        self.status = RunStatus(out_dir / "status.json", STAGES)

    def run(self, redo: Iterable[str] = ()) -> RunStatus:
        """Run every stage that is not done yet.

        Args:
            redo: Stage names (or ``all``) to recompute even if they finished.

        Returns:
            The final run status.

        Raises:
            InputError: If the input folder does not exist, ``redo`` names an unknown
                stage, or detection has work to do but no detector was configured.
        """
        if not self.input_dir.is_dir():
            raise InputError(f"input folder not found: {self.input_dir}")
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.apply_redo(redo)
        self._warn_about_new_images()
        atomic_write_json(self.out_dir / "run_config.json", self._run_config())
        stages: dict[str, Callable[[], dict[str, Any]]] = {
            "ingest": self._ingest,
            "quality": self._quality,
            "detect": self._detect,
            "classify": self._classify,
            "events": self._events,
            "summarize": self._summarize,
            "report": self._report,
        }
        for stage in STAGES:
            if self.status.is_done(stage):
                logger.info("[%s] already done, skipping", stage)
                continue
            logger.info("[%s] running", stage)
            self.status.mark_started(stage)
            info = stages[stage]()
            self.status.mark_done(stage, **info)
            logger.info("[%s] done %s", stage, info)
        return self.status

    def apply_redo(self, redo: Iterable[str]) -> list[str]:
        """Delete the named stages' files and clear their flags and every later flag.

        Args:
            redo: Stage names, or ``all``.

        Returns:
            The stages whose flags were cleared.

        Raises:
            InputError: If a name is not a stage.
        """
        names = list(STAGES) if "all" in redo else list(redo)
        unknown = sorted(set(names) - set(STAGES))
        if unknown:
            raise InputError(f"unknown stage(s) {unknown}; choose from {', '.join(STAGES)} or all")
        if not names:
            return []
        for name in names:
            for relative in STAGE_FILES[name]:
                target = self.out_dir / relative
                if target.is_dir():
                    shutil.rmtree(target)
                elif target.exists():
                    target.unlink()
        first = min(STAGES.index(n) for n in names)
        cleared = self.status.invalidate_from(STAGES[first])
        logger.info("redo: cleared %s", ", ".join(cleared) or "nothing")
        return cleared

    def _path(self, relative: str) -> Path:
        return self.out_dir / relative

    def _read(self, name: str) -> pd.DataFrame:
        return pd.read_parquet(self._path(name))

    def _warn_about_new_images(self) -> None:
        media_path = self._path("media.parquet")
        if not (self.status.is_done("ingest") and media_path.exists()):
            return
        new = find_new_images(self.input_dir, pd.read_parquet(media_path, columns=["rel_path"]))
        if new:
            logger.warning(
                "%d image(s) were added since ingest; run with --redo ingest to include them "
                "(later stages only process the new photos)",
                len(new),
            )

    def _run_config(self) -> dict[str, Any]:
        return {
            "version": __version__,
            "input_folder_name": self.input_dir.name,
            "settings": asdict(self.settings),
        }

    def _ingest(self) -> dict[str, Any]:
        media = ingest(self.input_dir, self.settings.valid_from, self.settings.workers)
        write_parquet(media, self._path("media.parquet"))
        return {
            "photos": len(media),
            "cameras": int(media["camera_id"].nunique()),
            "unreadable": int((~media["readable"]).sum()),
            "invalid_timestamps": int((~media["timestamp_valid"]).sum()),
        }

    def _quality(self) -> dict[str, Any]:
        quality = run_quality(
            self._read("media.parquet"),
            self.input_dir,
            self._path("checkpoints/quality.jsonl"),
            self.settings.quality,
            self.settings.workers,
        )
        write_parquet(quality, self._path("quality.parquet"))
        return {
            "measured": int(quality["decode_ok"].sum()),
            "decode_errors": int((~quality["decode_ok"]).sum()),
            "near_duplicates": int(quality["is_near_duplicate"].sum()),
        }

    def _detect(self) -> dict[str, Any]:
        if self.detector_factory is None:
            raise InputError("no detector configured; pass a detector factory")
        boxes, runs = run_detect(
            self._read("media.parquet"),
            self._read("quality.parquet"),
            self.input_dir,
            self.detector_factory,
            self._path("checkpoints/detections.jsonl"),
            self.settings.workers,
        )
        write_parquet(boxes, self._path("detections.parquet"))
        write_parquet(runs, self._path("detection_runs.parquet"))
        return {
            "photos": len(runs),
            "boxes": len(boxes),
            "errors": int(runs["error"].notna().sum()),
            "models": sorted(runs["model"].dropna().unique().tolist()),
        }

    def _classify(self) -> dict[str, Any]:
        table = run_classify(
            self._read("detections.parquet"),
            self._read("media.parquet"),
            self.input_dir,
            self.classifier_factory,
            self._path("checkpoints/classifications.jsonl"),
            self.settings.detection,
            self.settings.classification,
            self.settings.workers,
        )
        write_parquet(table, self._path("classifications.parquet"))
        return {
            "boxes": len(table),
            "species_level": int((table["species_level"] == "species").sum()),
            "models": sorted(table["model"].dropna().unique().tolist()),
        }

    def _events(self) -> dict[str, Any]:
        boxes = self._read("detections.parquet")
        classifications = self._read("classifications.parquet")
        count_conf = self.settings.detection.count_conf
        photos = build_photo_table(
            self._read("media.parquet"),
            self._read("quality.parquet"),
            boxes,
            self._read("detection_runs.parquet"),
            classifications,
            count_conf,
        )
        per_photo = photo_species(boxes, classifications, count_conf)
        photos, events, event_species = build_events(photos, per_photo, self.settings.events.gap_s)
        write_parquet(photos, self._path("photos.parquet"))
        write_parquet(events, self._path("events.parquet"))
        write_parquet(event_species, self._path("event_species.parquet"))
        return {
            "events": len(events),
            "animal_events": int((events["observation_type"] == "animal").sum()),
            "event_species_rows": len(event_species),
        }

    def _load_results(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        return (
            self._read("photos.parquet"),
            self._read("events.parquet"),
            self._read("event_species.parquet"),
        )

    def _summarize(self) -> dict[str, Any]:
        photos, events, event_species = self._load_results()
        tables = build_summaries(photos, events, event_species, self.settings.health)
        summary_dir = self._path("summary")
        for name, table in tables.items():
            write_parquet(table, summary_dir / f"{name}.parquet")
            table.to_csv(summary_dir / f"{name}.csv", index=False, lineterminator="\n")
        atomic_write_json(summary_dir / "overview.json", overview(photos, events, event_species))
        plots = render_all(tables, self._path("plots"))
        return {"tables": sorted(tables), "plots": sorted(plots)}

    def _report(self) -> dict[str, Any]:
        photos, events, event_species = self._load_results()
        summary_dir = self._path("summary")
        tables = {p.stem: pd.read_parquet(p) for p in sorted(summary_dir.glob("*.parquet"))}
        outputs = [
            OutputFile(name, len(self._read(name)), grain, description)
            for name, (grain, description) in OUTPUT_DESCRIPTIONS.items()
        ]
        detect_info = self.status.info("detect")
        classify_info = self.status.info("classify")
        run_info = {
            "detector": ", ".join(detect_info.get("models", [])) or "none",
            "classifier": ", ".join(classify_info.get("models", [])) or "none",
            "event gap (s)": f"{self.settings.events.gap_s:g}",
            "counting confidence": f"{self.settings.detection.count_conf:g}",
            "species threshold": f"{self.settings.classification.species_threshold:g}",
            "long-silence threshold (h)": f"{self.settings.health.gap_hours:g}",
        }
        path = write_report(
            self._path("report.html"),
            title=self.settings.report_title,
            overview=overview(photos, events, event_species),
            tables=tables,
            events=events,
            plots={
                name: f"plots/{name}.png"
                for name in (
                    "species_events",
                    "activity_by_hour",
                    "species_by_camera",
                    "camera_uptime",
                )
            },
            outputs=outputs,
            run_info=run_info,
            synthetic=self.synthetic,
        )
        return {"report": path.name}
