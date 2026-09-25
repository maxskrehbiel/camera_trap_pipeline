"""Self-contained HTML report: headline numbers, the four charts and their tables."""

from __future__ import annotations

import html
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from . import __version__
from .atomic_io import atomic_write_text

MAX_EVENT_ROWS = 12

_CSS = """
:root {
  color-scheme: light;
  --page: #f9f9f7; --surface: #fcfcfb; --ink: #0b0b0b; --ink-2: #52514e; --muted: #898781;
  --rule: #e1e0d9; --accent: #2a78d6; --note: #fff6e0; --note-rule: #eda100;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    color-scheme: dark;
    --page: #0d0d0d; --surface: #1a1a19; --ink: #ffffff; --ink-2: #c3c2b7; --muted: #898781;
    --rule: #2c2c2a; --accent: #3987e5; --note: #2a2415; --note-rule: #c98500;
  }
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--page); color: var(--ink);
  font: 15px/1.55 system-ui, -apple-system, "Segoe UI", sans-serif; }
main { max-width: 960px; margin: 0 auto; padding: 32px 16px 64px; }
h1 { font-size: 26px; margin: 0 0 4px; letter-spacing: -0.01em; }
h2 { font-size: 18px; margin: 40px 0 8px; padding-top: 16px; border-top: 1px solid var(--rule); }
p.lede { color: var(--ink-2); margin: 0 0 20px; }
.note { background: var(--note); border-left: 3px solid var(--note-rule); padding: 10px 14px;
  border-radius: 4px; color: var(--ink); margin: 16px 0 24px; }
.tiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 12px; }
.tile { background: var(--surface); border: 1px solid var(--rule); border-radius: 8px;
  padding: 12px; }
.tile .label { color: var(--ink-2); font-size: 13px; }
.tile .value { font-size: 24px; font-weight: 600; }
figure { margin: 12px 0; background: #fcfcfb; border: 1px solid var(--rule); border-radius: 8px;
  padding: 8px; }
figure img { width: 100%; height: auto; display: block; }
.scroll { overflow-x: auto; }
table { border-collapse: collapse; width: 100%; font-size: 13px; margin: 8px 0;
  font-variant-numeric: tabular-nums; }
th, td { text-align: left; padding: 6px 8px; border-bottom: 1px solid var(--rule);
  white-space: nowrap; }
th { color: var(--ink-2); font-weight: 600; }
td.num, th.num { text-align: right; }
details { margin: 8px 0; }
summary { cursor: pointer; color: var(--accent); }
code { font-size: 13px; }
footer { color: var(--muted); font-size: 13px; margin-top: 48px; }
"""


@dataclass(frozen=True)
class OutputFile:
    """One row of the report's data dictionary.

    Attributes:
        name: File name relative to the output folder.
        rows: Number of rows written.
        grain: What one row represents.
        description: What the file is for.
    """

    name: str
    rows: int
    grain: str
    description: str


def _format(value: Any) -> str:
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return "-"
    if isinstance(value, datetime):
        return value.strftime("%Y-%m-%d %H:%M")
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:,.1f}"
    if isinstance(value, int):
        return f"{value:,}"
    return str(value)


def _is_numeric(series: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series)


def html_table(frame: pd.DataFrame, columns: Mapping[str, str]) -> str:
    """Render selected columns as an escaped HTML table.

    Args:
        frame: Data to show.
        columns: Column name to header text, in display order.

    Returns:
        A ``<table>`` wrapped in a horizontally scrolling ``<div>``.
    """
    numeric = {c: _is_numeric(frame[c]) for c in columns}
    head = "".join(
        f'<th class="num">{html.escape(h)}</th>' if numeric[c] else f"<th>{html.escape(h)}</th>"
        for c, h in columns.items()
    )
    body = []
    for record in frame[list(columns)].to_dict("records"):
        cells = "".join(
            f'<td class="num">{html.escape(_format(record[c]))}</td>'
            if numeric[c]
            else f"<td>{html.escape(_format(record[c]))}</td>"
            for c in columns
        )
        body.append(f"<tr>{cells}</tr>")
    return (
        f'<div class="scroll"><table><thead><tr>{head}</tr></thead>'
        f"<tbody>{''.join(body)}</tbody></table></div>"
    )


def _tile(label: str, value: Any) -> str:
    return (
        f'<div class="tile"><div class="label">{html.escape(label)}</div>'
        f'<div class="value">{html.escape(_format(value))}</div></div>'
    )


def _figure(src: str, alt: str) -> str:
    return (
        f'<figure><img src="{html.escape(src)}" alt="{html.escape(alt)}" loading="lazy"></figure>'
    )


def _details(summary: str, content: str) -> str:
    return f"<details><summary>{html.escape(summary)}</summary>{content}</details>"


SPECIES_COLUMNS = {
    "species": "Species",
    "n_events": "Events",
    "n_cameras": "Cameras",
    "animals": "Animals (sum of event counts)",
    "first_seen": "First seen",
    "last_seen": "Last seen",
}
BY_CAMERA_COLUMNS = {
    "camera_id": "Camera",
    "species": "Species",
    "n_events": "Events",
    "animals": "Animals",
    "max_group": "Largest group",
    "events_per_100_days": "Events per 100 effort days",
}
HEALTH_COLUMNS = {
    "camera_id": "Camera",
    "n_photos": "Photos",
    "effort_days": "Effort days",
    "n_gaps": "Long silences",
    "longest_silence_hours": "Longest silence (h)",
    "n_invalid_timestamp": "Unset clock",
    "n_unreadable": "Unreadable",
    "pct_ir": "Night IR (%)",
    "pct_blurry": "Blurry (%)",
    "pct_near_duplicate": "Near-dup (%)",
    "pct_empty": "Empty (%)",
}
GAP_COLUMNS = {
    "camera_id": "Camera",
    "gap_start": "Last photo before",
    "gap_end": "First photo after",
    "gap_hours": "Hours",
}
SIGHTING_COLUMNS = {
    "event_id": "Event",
    "camera_id": "Camera",
    "start": "Start",
    "duration_s": "Seconds",
    "n_photos": "Photos",
    "species": "Species",
    "species_count": "Count",
}
OUTPUT_COLUMNS = {"name": "File", "rows": "Rows", "grain": "One row per", "description": "Contents"}


def _header(title: str, overview: Mapping[str, Any], synthetic: bool) -> list[str]:
    first, last = overview.get("first_photo"), overview.get("last_photo")
    span = (
        f"{_format(first)[:10]} to {_format(last)[:10]}" if first is not None else "no valid dates"
    )
    parts = [
        f"<h1>{html.escape(title)}</h1>",
        f'<p class="lede">{overview["n_photos"]:,} photos from {overview["n_cameras"]} cameras, '
        f"{html.escape(span)}.</p>",
    ]
    if synthetic:
        parts.append(
            '<p class="note">Synthetic data: the photos were generated by '
            "<code>camera_trap_pipeline.synthetic</code> and the detector and classifier outputs "
            "are simulated. The numbers show what the pipeline produces, not real wildlife.</p>"
        )
    tiles = [
        ("Photos", overview["n_photos"]),
        ("Cameras", overview["n_cameras"]),
        ("Animal events", overview["n_animal_events"]),
        ("Species", overview["n_species"]),
        ("Unreadable photos", overview["n_unreadable"]),
        ("Unset-clock photos", overview["n_invalid_timestamp"]),
        ("Near-duplicates (%)", overview["pct_near_duplicate"]),
    ]
    parts.append('<div class="tiles">' + "".join(_tile(k, v) for k, v in tiles) + "</div>")
    return parts


def _species_sections(tables: Mapping[str, pd.DataFrame], plots: Mapping[str, str]) -> list[str]:
    hourly = tables["activity_by_hour"].pivot_table(
        index="species", columns="hour", values="n_events", aggfunc="sum", fill_value=0
    )
    hourly.columns = [f"{int(h):02d}" for h in hourly.columns]
    hourly = hourly.reset_index()
    hour_columns = {"species": "Species", **{c: c for c in hourly.columns[1:]}}
    return [
        "<h2>Species</h2>",
        "<p>An event is one burst of photos from one camera; its count is the largest number of "
        "animals of that species in any single frame.</p>",
        _figure(plots["species_events"], "Bar chart of events per species"),
        html_table(tables["species_totals"], SPECIES_COLUMNS),
        "<h2>When animals are active</h2>",
        _figure(plots["activity_by_hour"], "Small multiples of events by hour for each species"),
        _details("Events by hour (table)", html_table(hourly, hour_columns)),
        "<h2>Where</h2>",
        _figure(plots["species_by_camera"], "Heatmap of events per species and camera"),
        _details(
            "Species by camera (table)",
            html_table(tables["species_by_camera"], BY_CAMERA_COLUMNS),
        ),
    ]


def _health_section(tables: Mapping[str, pd.DataFrame], plots: Mapping[str, str]) -> list[str]:
    parts = [
        "<h2>Camera health</h2>",
        "<p>Effort days are the days from first to last photo minus long silences. A trail camera "
        "only fires on motion, so a silence can be a quiet spell rather than an outage.</p>",
        _figure(plots["camera_uptime"], "Calendar strip of photos per camera per day"),
        html_table(tables["camera_health"], HEALTH_COLUMNS),
    ]
    if not tables["camera_gaps"].empty:
        parts.append(html_table(tables["camera_gaps"], GAP_COLUMNS))
    return parts


def _closing_sections(
    events: pd.DataFrame, outputs: Sequence[OutputFile], run_info: Mapping[str, str]
) -> list[str]:
    largest = (
        events[events["observation_type"] == "animal"]
        .sort_values(
            ["species_count", "n_photos", "start", "event_id"],
            ascending=[False, False, True, True],
        )
        .head(MAX_EVENT_ROWS)
    )
    dictionary = pd.DataFrame([asdict(o) for o in outputs])
    info = "".join(
        f"<li>{html.escape(k)}: <code>{html.escape(v)}</code></li>" for k, v in run_info.items()
    )
    return [
        "<h2>Largest sightings</h2>",
        html_table(largest, SIGHTING_COLUMNS),
        "<h2>Output tables</h2>",
        "<p>Every table is Parquet in the run folder and can be queried with pandas, DuckDB or "
        "any Arrow reader.</p>",
        html_table(dictionary, OUTPUT_COLUMNS),
        f"<footer><p>camera_trap_pipeline {html.escape(__version__)}</p><ul>{info}</ul></footer>",
    ]


def write_report(
    path: Path,
    *,
    title: str,
    overview: Mapping[str, Any],
    tables: Mapping[str, pd.DataFrame],
    events: pd.DataFrame,
    plots: Mapping[str, str],
    outputs: Sequence[OutputFile],
    run_info: Mapping[str, str],
    synthetic: bool,
) -> Path:
    """Write the HTML report.

    The page has no timestamps or generated ids, so the same run always gives the same bytes.

    Args:
        path: Destination ``report.html``.
        title: Page heading.
        overview: Output of ``summarize.overview``.
        tables: Output of ``summarize.build_summaries``.
        events: Event table.
        plots: Chart name to image path relative to the report.
        outputs: Data dictionary rows.
        run_info: Models and settings to list in the footer, as display strings.
        synthetic: Whether to show the synthetic-data notice.

    Returns:
        The written path.
    """
    parts = [
        *_header(title, overview, synthetic),
        *_species_sections(tables, plots),
        *_health_section(tables, plots),
        *_closing_sections(events, outputs, run_info),
    ]
    page = (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        f"<title>{html.escape(title)}</title><style>{_CSS}</style></head>"
        f"<body><main>{''.join(parts)}</main></body></html>\n"
    )
    atomic_write_text(path, page)
    return path
