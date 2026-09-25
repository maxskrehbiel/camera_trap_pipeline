"""Image loading shared by the stages, and the exceptions that mean a file cannot be decoded."""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageOps

# Pillow signals a corrupt, truncated or oversized file with any of these.
IMAGE_DECODE_ERRORS: tuple[type[Exception], ...] = (
    OSError,
    ValueError,
    SyntaxError,
    Image.DecompressionBombError,
)


def load_rgb(path: Path) -> Image.Image:
    """Open an image, apply its EXIF orientation and decode it as RGB.

    Args:
        path: Image file.

    Returns:
        The fully decoded RGB image.
    """
    with Image.open(path) as image:
        return ImageOps.exif_transpose(image).convert("RGB")
