"""Exceptions raised on purpose by this package; the command line maps them to exit codes."""

from __future__ import annotations


class PipelineError(Exception):
    """Base class for errors this package raises deliberately."""


class InputError(PipelineError, ValueError):
    """Bad input data, configuration or arguments (exit code 2)."""


class ModelDependencyError(PipelineError, ImportError):
    """An optional model package is not installed (exit code 3)."""


class ImageFailedError(PipelineError):
    """A model reported that it could not process one image; the photo is recorded and skipped."""
