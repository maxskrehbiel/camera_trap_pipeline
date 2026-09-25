"""Shared fixtures: a small synthetic dataset and a guard that keeps unit tests offline."""

from __future__ import annotations

import socket
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from camera_trap_pipeline.mock_models import MockClassifier, MockDetector
from camera_trap_pipeline.pipeline import Pipeline
from camera_trap_pipeline.settings import PipelineSettings, QualitySettings
from camera_trap_pipeline.synthetic import (
    GroundTruth,
    SyntheticDataset,
    SyntheticSettings,
    generate_dataset,
    load_ground_truth,
)

TINY = SyntheticSettings(n_cameras=3, n_days=5, width=192, height=128, rate_scale=0.6)
TINY_SEED = 11
MODEL_SEED = 3


@pytest.fixture(autouse=True)
def no_network(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail any test that opens a network connection, unless it is marked integration."""
    if "integration" in request.keywords:
        return

    def refuse(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("network access is disabled in unit tests")

    monkeypatch.setattr(socket.socket, "connect", refuse)


@pytest.fixture(scope="session")
def tiny_dataset(tmp_path_factory: pytest.TempPathFactory) -> SyntheticDataset:
    """Three cameras over five days, with every injected field problem."""
    root = tmp_path_factory.mktemp("tiny") / "photos"
    return generate_dataset(root, TINY, np.random.default_rng(TINY_SEED))


@pytest.fixture(scope="session")
def tiny_truth(tiny_dataset: SyntheticDataset) -> GroundTruth:
    return load_ground_truth(tiny_dataset.root)


@pytest.fixture
def settings() -> PipelineSettings:
    """Pipeline settings matching the synthetic frames' info bar."""
    return PipelineSettings(workers=2, quality=QualitySettings(strip_bottom=TINY.info_strip))


@pytest.fixture(scope="session")
def reference_run(
    tiny_dataset: SyntheticDataset,
    tiny_truth: GroundTruth,
    tmp_path_factory: pytest.TempPathFactory,
) -> Path:
    """A complete run over the tiny dataset with the stand-in models."""
    out = tmp_path_factory.mktemp("reference") / "run"
    settings = PipelineSettings(workers=2, quality=QualitySettings(strip_bottom=TINY.info_strip))
    Pipeline(
        tiny_dataset.root,
        out,
        settings,
        lambda: MockDetector(tiny_truth, MODEL_SEED),
        lambda: MockClassifier(tiny_truth, MODEL_SEED),
        synthetic=True,
    ).run()
    return out
