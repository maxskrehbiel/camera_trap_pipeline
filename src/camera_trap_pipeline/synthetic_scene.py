"""Drawing synthetic trail-camera frames: backgrounds, animals, people, vehicles, info bar.

Every frame also yields normalized ground-truth boxes for what was drawn.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from typing import TypedDict

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from ._types import Float32Array, FloatArray
from .synthetic_schedule import (
    SPECIES_BY_NAME,
    SUNRISE_HOUR,
    SUNSET_HOUR,
    SpeciesProfile,
    SyntheticSettings,
    Visit,
    is_night,
)

UINT8_MAX = 255
EDGE_MARGIN = 0.1
NIGHT_GAIN = 0.45
NIGHT_NOISE = 5.0
DAY_NOISE = 2.5
FOG_BLUR_RADIUS = 2.5
EXIF_FORMAT = "%Y:%m:%d %H:%M:%S"


class Subject(TypedDict):
    """Something drawn into a visit's frames, moving a little between frames."""

    kind: str
    species: str | None
    center: tuple[float, float]
    size: float
    facing: int
    speed: float


class TruthObject(TypedDict):
    """Ground truth for one drawn object."""

    category: str
    species: str | None
    lineage: str | None
    bbox: list[float]


@dataclass(frozen=True, eq=False)
class Scene:
    """Pre-rendered empty view of one camera.

    Attributes:
        day: Daylight background, ``H x W x 3`` floats.
        night: Infrared-style monochrome background, ``H x W x 3`` floats.
    """

    day: Float32Array
    night: Float32Array


def make_scene(settings: SyntheticSettings, rng: np.random.Generator) -> Scene:
    """Draw a camera's background once, in daylight and infrared versions.

    Args:
        settings: Dataset settings (frame size).
        rng: Random generator for the layout.

    Returns:
        The camera's scene.
    """
    day = _background(settings, rng).astype(np.float32)
    gray = day.mean(axis=2, keepdims=True) * NIGHT_GAIN
    return Scene(day=day, night=np.repeat(gray, 3, axis=2))


def _background(settings: SyntheticSettings, rng: np.random.Generator) -> FloatArray:
    height, width = settings.height, settings.width
    horizon = int(height * rng.uniform(0.28, 0.4))
    image = np.zeros((height, width, 3))
    sky_top, sky_low = np.array([150, 185, 215]), np.array([205, 220, 230])
    for row in range(horizon):
        image[row] = sky_top + (sky_low - sky_top) * row / max(horizon, 1)
    ground = np.array([95, 110, 60]) + rng.uniform(-15, 15, 3)
    image[horizon:] = ground
    image[horizon:] += rng.normal(0, 14, (height - horizon, width, 1))
    canvas = Image.fromarray(np.clip(image, 0, UINT8_MAX).astype(np.uint8))
    draw = ImageDraw.Draw(canvas)
    for _ in range(int(rng.integers(3, 7))):
        x = int(rng.uniform(0, width))
        trunk = int(rng.uniform(6, 16))
        draw.rectangle((x, 0, x + trunk, horizon + int(rng.uniform(10, 40))), fill=(70, 55, 40))
        crown = int(rng.uniform(30, 60))
        draw.ellipse((x - crown, -crown // 2, x + trunk + crown, crown), fill=(55, 85, 45))
    for _ in range(int(rng.integers(4, 9))):
        cx, cy = rng.uniform(0, width), rng.uniform(horizon, height * 0.9)
        r = rng.uniform(12, 35)
        draw.ellipse((cx - r, cy - r * 0.6, cx + r, cy + r * 0.6), fill=(60, 95, 50))
    return np.asarray(canvas, dtype=np.float64)


def _draw_animal(
    draw: ImageDraw.ImageDraw,
    profile: SpeciesProfile,
    center: tuple[float, float],
    width_px: float,
    facing: int,
    night: bool,
) -> tuple[float, float, float, float]:
    cx, cy = center
    body_w, body_h = width_px, width_px * 0.55
    color = (190, 190, 190) if night else profile.color
    left, top = cx - body_w / 2, cy - body_h / 2
    draw.ellipse((left, top, left + body_w, top + body_h), fill=color)
    head = body_h * 0.55
    hx = cx + facing * body_w * 0.5
    hy = top - head * 0.3
    draw.ellipse((hx - head / 2, hy - head / 2, hx + head / 2, hy + head / 2), fill=color)
    leg = body_h * 0.7
    for offset in (-0.3, -0.1, 0.15, 0.32):
        lx = cx + offset * body_w
        draw.line((lx, top + body_h * 0.8, lx, top + body_h * 0.8 + leg), fill=color, width=3)
    if night:
        eye = max(1.5, head * 0.12)
        draw.ellipse((hx - eye, hy - eye, hx + eye, hy + eye), fill=(255, 255, 255))
    x0 = min(left, hx - head / 2)
    x1 = max(left + body_w, hx + head / 2)
    return x0, hy - head / 2, x1, top + body_h * 0.8 + leg


def _draw_person(
    draw: ImageDraw.ImageDraw, center: tuple[float, float], height_px: float, night: bool
) -> tuple[float, float, float, float]:
    cx, cy = center
    width = height_px * 0.28
    color = (175, 175, 175) if night else (40, 55, 110)
    top = cy - height_px / 2
    draw.rounded_rectangle(
        (cx - width / 2, top + width * 0.8, cx + width / 2, cy + height_px / 2), 4, fill=color
    )
    draw.ellipse((cx - width * 0.4, top, cx + width * 0.4, top + width * 0.8), fill=color)
    return cx - width / 2, top, cx + width / 2, cy + height_px / 2


def _draw_vehicle(
    draw: ImageDraw.ImageDraw, center: tuple[float, float], width_px: float, night: bool
) -> tuple[float, float, float, float]:
    cx, cy = center
    height = width_px * 0.45
    color = (160, 160, 160) if night else (140, 30, 30)
    draw.rounded_rectangle(
        (cx - width_px / 2, cy - height / 2, cx + width_px / 2, cy + height / 2), 6, fill=color
    )
    for wheel in (-0.3, 0.3):
        wx = cx + wheel * width_px
        r = height * 0.28
        draw.ellipse((wx - r, cy + height / 2 - r, wx + r, cy + height / 2 + r), fill=(20, 20, 20))
    return cx - width_px / 2, cy - height / 2, cx + width_px / 2, cy + height / 2 + height * 0.28


def _to_fraction(
    box: tuple[float, float, float, float], width: int, height: int, usable_height: int
) -> list[float]:
    x0, y0 = max(box[0], 0.0), max(box[1], 0.0)
    x1, y1 = min(box[2], float(width)), min(box[3], float(usable_height))
    return [
        round(x0 / width, 4),
        round(y0 / height, 4),
        round(max(x1 - x0, 1.0) / width, 4),
        round(max(y1 - y0, 1.0) / height, 4),
    ]


def _draw_subject(
    draw: ImageDraw.ImageDraw, subject: Subject, night: bool
) -> tuple[tuple[float, float, float, float], str, str | None, str | None]:
    if subject["kind"] == "animal" and subject["species"] is not None:
        profile = SPECIES_BY_NAME[subject["species"]]
        box = _draw_animal(
            draw, profile, subject["center"], subject["size"], subject["facing"], night
        )
        return box, "animal", profile.name, profile.lineage
    if subject["kind"] == "person":
        return _draw_person(draw, subject["center"], subject["size"], night), "person", None, None
    return _draw_vehicle(draw, subject["center"], subject["size"], night), "vehicle", None, None


def render_frame(
    scene: Scene,
    settings: SyntheticSettings,
    moment: datetime,
    clock: datetime,
    camera_id: str,
    subjects: list[Subject],
    rng: np.random.Generator,
    blurred: bool,
) -> tuple[Image.Image, list[TruthObject]]:
    """Draw one frame.

    Args:
        scene: Camera background.
        settings: Dataset settings.
        moment: True capture time (sets day or night rendering).
        clock: Time shown by the camera clock (burned into the info bar).
        camera_id: Camera label for the info bar.
        subjects: Objects to draw.
        rng: Random generator for sensor noise.
        blurred: Apply lens blur (fog).

    Returns:
        The frame and a normalized ground-truth box for each subject.
    """
    night = is_night(moment)
    hour = moment.hour + moment.minute / 60
    noise = rng.standard_normal((settings.height, settings.width, 1), dtype=np.float32)
    if night:
        pixels = scene.night + noise * NIGHT_NOISE
    else:
        phase = math.pi * (hour - SUNRISE_HOUR) / (SUNSET_HOUR - SUNRISE_HOUR)
        pixels = scene.day * (0.75 + 0.25 * math.sin(phase)) + noise * DAY_NOISE
    image = Image.fromarray(np.clip(pixels, 0, UINT8_MAX).astype(np.uint8))
    draw = ImageDraw.Draw(image)
    usable = settings.height - round(settings.height * settings.info_strip)
    objects: list[TruthObject] = []
    for subject in subjects:
        box, category, species, lineage = _draw_subject(draw, subject, night)
        bbox = _to_fraction(box, settings.width, settings.height, usable)
        objects.append({"category": category, "species": species, "lineage": lineage, "bbox": bbox})
    if blurred:
        image = image.filter(ImageFilter.GaussianBlur(radius=FOG_BLUR_RADIUS))
        draw = ImageDraw.Draw(image)
    if settings.info_strip > 0:
        draw.rectangle((0, usable, settings.width, settings.height), fill=(0, 0, 0))
        label = f"{camera_id.upper()}  {clock:%Y-%m-%d %H:%M:%S}  {12 + moment.hour // 3}C"
        draw.text((6, usable + 3), label, fill=(255, 255, 255))
    return image, objects


def exif_for(clock: datetime, camera_id: str) -> Image.Exif:
    """Build the EXIF block a trail camera would write.

    Args:
        clock: Time shown by the camera clock.
        camera_id: Camera id, used for the serial number.

    Returns:
        EXIF with make, model, DateTime, DateTimeOriginal and BodySerialNumber.
    """
    exif = Image.Exif()
    stamp = clock.strftime(EXIF_FORMAT)
    exif[0x010F] = "camera_trap_pipeline"
    exif[0x0110] = "synthetic"
    exif[0x0132] = stamp
    # Assigning a dict to the EXIF IFD tag writes a sub-IFD on every supported Pillow version.
    exif[0x8769] = {0x9003: stamp, 0xA431: f"SYN-{camera_id}"}
    return exif


def subjects_for(
    visit: Visit, settings: SyntheticSettings, rng: np.random.Generator
) -> list[Subject]:
    """Place the animals, person or vehicle of a visit in the frame.

    Args:
        visit: The visit.
        settings: Dataset settings (frame size).
        rng: Random generator.

    Returns:
        One subject per animal, or a single person or vehicle; none for empty triggers.
    """
    width, height = settings.width, settings.height
    if visit.kind == "animal" and visit.species is not None:
        profile = SPECIES_BY_NAME[visit.species]
        subjects: list[Subject] = []
        for _ in range(visit.count):
            depth = float(rng.uniform(0.55, 0.8))
            subjects.append(
                {
                    "kind": "animal",
                    "species": profile.name,
                    "center": (float(rng.uniform(0.12, 0.88)) * width, depth * height),
                    "size": profile.size * width * depth * float(rng.uniform(1.0, 1.4)),
                    "facing": 1 if rng.random() < 0.5 else -1,
                    "speed": float(rng.uniform(0.01, 0.03)) * width,
                }
            )
        return subjects
    if visit.kind == "person":
        center, size, speed = (0.3 * width, 0.55 * height), 0.45 * height, 0.03 * width
    elif visit.kind == "vehicle":
        center, size, speed = (0.25 * width, 0.62 * height), 0.4 * width, 0.08 * width
    else:
        return []
    return [
        {
            "kind": visit.kind,
            "species": None,
            "center": center,
            "size": size,
            "facing": 1,
            "speed": speed,
        }
    ]


def step_subject(subject: Subject, width: int) -> None:
    """Move a subject one frame along, turning around before it leaves the frame.

    Args:
        subject: Subject to move in place.
        width: Frame width in pixels.
    """
    x = subject["center"][0] + subject["speed"] * subject["facing"]
    if not EDGE_MARGIN * width <= x <= (1 - EDGE_MARGIN) * width:
        subject["facing"] *= -1
        x = subject["center"][0] + subject["speed"] * subject["facing"]
    subject["center"] = (x, subject["center"][1])
