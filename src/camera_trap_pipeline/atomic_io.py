"""Crash-safe file writes: write to a temporary sibling, fsync, then rename over the target."""

from __future__ import annotations

import contextlib
import json
import os
import uuid
from pathlib import Path
from typing import Any

import pandas as pd


def _temp_sibling(path: Path) -> Path:
    return path.with_name(f".{path.name}.{uuid.uuid4().hex[:8]}.tmp")


def atomic_write_bytes(path: Path, data: bytes) -> None:
    """Write bytes so that readers see either the old file or the complete new one.

    Args:
        path: Destination file. Parent folders are created when missing.
        data: Full file contents.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = _temp_sibling(path)
    try:
        with tmp.open("wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        tmp.replace(path)
    except BaseException:
        with contextlib.suppress(OSError):
            tmp.unlink()
        raise


def atomic_write_text(path: Path, text: str) -> None:
    """Write UTF-8 text atomically.

    Args:
        path: Destination file.
        text: Full file contents.
    """
    atomic_write_bytes(path, text.encode("utf-8"))


def atomic_write_json(path: Path, obj: Any) -> None:
    """Write an object as indented JSON atomically.

    Args:
        path: Destination file.
        obj: JSON-serializable object; non-JSON values fall back to ``str``.
    """
    atomic_write_text(path, json.dumps(obj, indent=2, default=str) + "\n")


def write_parquet(frame: pd.DataFrame, path: Path) -> None:
    """Write a DataFrame to Parquet atomically, without the index.

    Args:
        frame: Table to write.
        path: Destination ``.parquet`` file.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = _temp_sibling(path)
    try:
        frame.to_parquet(tmp, index=False)
        tmp.replace(path)
    except BaseException:
        with contextlib.suppress(OSError):
            tmp.unlink()
        raise
