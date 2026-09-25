"""Frozen configuration for every stage; the CLI builds these and the pipeline only reads them."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class QualitySettings:
    """Image-quality and near-duplicate settings.

    Attributes:
        max_side: Longest side, in pixels, that frames are shrunk to before measuring, so
            sharpness is comparable across cameras with different resolutions.
        strip_top: Fraction of the frame height to ignore at the top (burned-in info bar).
        strip_bottom: Fraction of the frame height to ignore at the bottom.
        dark_brightness: Mean gray level (0-255) below which a frame is flagged dark.
        overexposed_brightness: Mean gray level above which a frame is flagged overexposed.
        ir_max_channel_spread: Mean absolute difference between color channels below which
            a frame is treated as infrared (night) monochrome.
        blur_ratio: A frame is blurry when its sharpness is below this fraction of the
            median sharpness of the same camera in the same day/night mode.
        duplicate_max_bits: Largest difference-hash Hamming distance (out of 64 bits) that
            still counts as a near-duplicate.
        duplicate_window: How many earlier frames from the same camera each frame is
            compared against when looking for near-duplicates.
        duplicate_max_gap_s: Only frames taken within this many seconds of each other can
            be near-duplicates, so a static view photographed on different days is not.
    """

    max_side: int = 640
    strip_top: float = 0.0
    strip_bottom: float = 0.0
    dark_brightness: float = 30.0
    overexposed_brightness: float = 225.0
    ir_max_channel_spread: float = 4.0
    blur_ratio: float = 0.35
    duplicate_max_bits: int = 6
    duplicate_window: int = 8
    duplicate_max_gap_s: float = 300.0


@dataclass(frozen=True)
class DetectionSettings:
    """How detector output is used downstream.

    Attributes:
        count_conf: Boxes at or above this confidence are counted and classified. Lower
            boxes stay in ``detections.parquet`` for review but do not affect counts.
    """

    count_conf: float = 0.2


@dataclass(frozen=True)
class ClassificationSettings:
    """Species classification settings.

    Attributes:
        max_boxes_per_photo: Only the most confident animal boxes per photo are classified.
        species_threshold: Minimum score for a label; below it the score is summed up the
            taxonomy (genus, family, order, class) until some group reaches it.
    """

    max_boxes_per_photo: int = 10
    species_threshold: float = 0.65


@dataclass(frozen=True)
class EventSettings:
    """Event grouping.

    Attributes:
        gap_s: A photo more than this many seconds after the previous photo from the same
            camera starts a new event.
    """

    gap_s: float = 60.0


@dataclass(frozen=True)
class HealthSettings:
    """Camera-health reporting.

    Attributes:
        gap_hours: A silence longer than this between consecutive photos from one camera is
            reported as a possible outage.
    """

    gap_hours: float = 48.0


@dataclass(frozen=True)
class PipelineSettings:
    """Everything a pipeline run needs besides the models.

    Attributes:
        valid_from: Timestamps on or before this date are treated as an unset camera clock.
        workers: Threads used to read and decode images.
        report_title: Heading of the HTML report.
        quality: Image-quality settings.
        detection: Detection settings.
        classification: Classification settings.
        events: Event grouping settings.
        health: Camera-health settings.
    """

    valid_from: str = "2001-01-01"
    workers: int = 4
    report_title: str = "Camera-trap pipeline report"
    quality: QualitySettings = field(default_factory=QualitySettings)
    detection: DetectionSettings = field(default_factory=DetectionSettings)
    classification: ClassificationSettings = field(default_factory=ClassificationSettings)
    events: EventSettings = field(default_factory=EventSettings)
    health: HealthSettings = field(default_factory=HealthSettings)
