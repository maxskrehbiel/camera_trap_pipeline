from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from camera_trap_pipeline.plots import render_all
from camera_trap_pipeline.report import OutputFile, html_table, write_report
from camera_trap_pipeline.settings import HealthSettings
from camera_trap_pipeline.summarize import build_summaries, overview


def sample_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    times = pd.to_datetime(
        ["2026-04-06 06:00", "2026-04-06 21:00", "2026-04-07 06:30", "2026-04-11 07:00"]
    )
    photos = pd.DataFrame(
        {
            "photo_id": ["a", "b", "c", "d"],
            "camera_id": ["cam_<b>", "cam_<b>", "cam_2", "cam_<b>"],
            "timestamp": times,
            "timestamp_valid": True,
            "decode_ok": True,
            "detected": True,
            "n_animals": [1, 2, 1, 0],
            "n_people": 0,
            "n_vehicles": 0,
            "is_ir": [False, True, False, False],
            "is_dark": False,
            "is_blurry": False,
            "is_near_duplicate": False,
        }
    )
    events = pd.DataFrame(
        {
            "event_id": ["e1", "e2", "e3", "e4"],
            "camera_id": ["cam_<b>", "cam_<b>", "cam_2", "cam_<b>"],
            "start": times,
            "duration_s": [2.0, 3.0, 1.0, 0.0],
            "n_photos": [3, 3, 3, 1],
            "observation_type": ["animal", "animal", "animal", "blank"],
            "species": ["coyote", "<script>alert(1)</script>", "coyote", None],
            "species_count": [1, 2, 1, 0],
        }
    )
    event_species = pd.DataFrame(
        {
            "event_id": ["e1", "e2", "e3"],
            "camera_id": ["cam_<b>", "cam_<b>", "cam_2"],
            "start": times[:3],
            "species": ["coyote", "<script>alert(1)</script>", "coyote"],
            "lineage": [
                "mammalia;carnivora;canidae;canis;latrans",
                "",
                "mammalia;carnivora;canidae;canis;latrans",
            ],
            "max_count": [1, 2, 1],
            "n_photos": 3,
            "max_score": 0.9,
        }
    )
    return photos, events, event_species


def test_html_table_escapes_and_formats() -> None:
    frame = pd.DataFrame(
        {
            "name": ["<b>x</b>", None],
            "n": [1234, 5],
            "share": [12.345, float("nan")],
            "when": [pd.Timestamp("2026-04-06 06:30"), pd.NaT],
            "flag": [True, False],
        }
    )
    rendered = html_table(
        frame, {"name": "Name", "n": "Count", "share": "%", "when": "When", "flag": "Flag"}
    )
    assert "&lt;b&gt;x&lt;/b&gt;" in rendered
    assert "<b>" not in rendered
    assert '<td class="num">1,234</td>' in rendered
    assert '<td class="num">12.3</td>' in rendered
    assert "<td>2026-04-06 06:30</td>" in rendered
    assert "<td>yes</td>" in rendered
    assert rendered.count("<td>-</td>") + rendered.count('<td class="num">-</td>') == 3


def test_charts_and_report_render(tmp_path: Path) -> None:
    photos, events, event_species = sample_tables()
    tables = build_summaries(photos, events, event_species, HealthSettings(gap_hours=48))
    plots = render_all(tables, tmp_path / "plots")
    for path in plots.values():
        with Image.open(path) as image:
            assert image.format == "PNG"
            assert image.width == 880
            assert np.asarray(image.convert("L")).std() > 0

    report = write_report(
        tmp_path / "report.html",
        title="Test <report>",
        overview=overview(photos, events, event_species),
        tables=tables,
        events=events,
        plots={name: f"plots/{path.name}" for name, path in plots.items()},
        outputs=[OutputFile("events.parquet", 4, "event", "Bursts")],
        run_info={"detector": "fake<1>"},
        synthetic=False,
    )
    html = report.read_text(encoding="utf-8")
    assert "<title>Test &lt;report&gt;</title>" in html
    assert "<script>" not in html
    assert "&lt;script&gt;" in html
    assert "cam_&lt;b&gt;" in html
    assert "Synthetic data" not in html
    assert 'src="plots/camera_uptime.png"' in html
    assert "fake&lt;1&gt;" in html
    assert str(tmp_path) not in html


def test_empty_timeline_still_renders(tmp_path: Path) -> None:
    photos, events, event_species = sample_tables()
    photos["timestamp_valid"] = False
    tables = build_summaries(photos, events, event_species.iloc[0:0], HealthSettings())
    plots = render_all(tables, tmp_path)
    assert all(path.exists() for path in plots.values())
