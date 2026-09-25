"""Static PNG charts for the report, drawn with matplotlib's object API (no global pyplot state)."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Literal

import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Patch, Rectangle

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SERIES = "#2a78d6"
EMPTY_CELL = "#f0efec"
STATUS_SERIOUS = "#ec835a"
# One-hue blue ramp, light to dark (steps 100-700 of the reference sequential scale).
SEQUENTIAL = (
    "#cde2fb",
    "#b7d3f6",
    "#9ec5f4",
    "#86b6ef",
    "#6da7ec",
    "#5598e7",
    "#3987e5",
    "#2a78d6",
    "#256abf",
    "#1c5cab",
    "#184f95",
    "#104281",
    "#0d366b",
)
DPI = 110
WIDTH_IN = 8.0
MAX_BAR_ROWS = 15
MAX_PANELS = 6
PANEL_COLUMNS = 3
NIGHT_HOURS = ((-0.5, 5.5), (19.5, 23.5))
TITLE_IN = 0.85
_LUMINANCE_FOR_WHITE_TEXT = 0.45
SECONDS_PER_DAY = 86_400.0


def _new_figure(height_in: float) -> Figure:
    figure = Figure(figsize=(WIDTH_IN, height_in), dpi=DPI, facecolor=SURFACE)
    figure.set_layout_engine("none")
    return figure


def _titles(figure: Figure, title: str, subtitle: str) -> None:
    height = figure.get_figheight()
    figure.text(
        0.02, 1 - 0.12 / height, title, ha="left", va="top", fontsize=12, weight="bold", color=INK
    )
    figure.text(
        0.02, 1 - 0.38 / height, subtitle, ha="left", va="top", fontsize=9, color=INK_SECONDARY
    )


def _axes_box(
    figure: Figure, left: float, right: float, top_in: float, bottom_in: float
) -> tuple[float, float, float, float]:
    """Axes rectangle from horizontal fractions and vertical margins in inches."""
    height = figure.get_figheight()
    return (left, bottom_in / height, right - left, 1 - (top_in + bottom_in) / height)


def _style(axes: Axes, value_axis: Literal["x", "y"]) -> None:
    axes.set_facecolor(SURFACE)
    for side in ("top", "right", "left", "bottom"):
        axes.spines[side].set_visible(False)
    base = "left" if value_axis == "x" else "bottom"
    axes.spines[base].set_visible(True)
    axes.spines[base].set_color(AXIS)
    axes.tick_params(colors=INK_MUTED, labelsize=8.5, length=0)
    axes.grid(axis=value_axis, color=GRID, linewidth=1.0)
    axes.set_axisbelow(True)


def _save(figure: Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    # No "Software" entry, so the PNG bytes do not change with the matplotlib version.
    figure.savefig(path, facecolor=SURFACE, metadata={"Software": None})
    return path


def _ramp_color(value: float, maximum: float) -> str:
    if maximum <= 0:
        return SEQUENTIAL[0]
    position = min(max(value / maximum, 0.0), 1.0)
    return SEQUENTIAL[round(position * (len(SEQUENTIAL) - 1))]


def _text_on(fill: str) -> str:
    red, green, blue = (int(fill[i : i + 2], 16) / 255 for i in (1, 3, 5))
    luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
    return "#ffffff" if luminance < _LUMINANCE_FOR_WHITE_TEXT else INK


def species_events_chart(totals: pd.DataFrame, path: Path) -> Path:
    """Horizontal bars: number of events per label, most frequent first.

    Args:
        totals: Output of ``species_totals``.
        path: PNG destination.

    Returns:
        ``path``.
    """
    top = totals.head(MAX_BAR_ROWS)
    figure = _new_figure(TITLE_IN + 0.45 + 0.32 * max(len(top), 1))
    axes = figure.add_axes(_axes_box(figure, 0.26, 0.94, TITLE_IN, 0.45))
    _style(axes, "x")
    labels = list(top["species"])[::-1]
    values = list(top["n_events"])[::-1]
    axes.barh(labels, values, height=0.55, color=SERIES)
    peak = max(values, default=1)
    for row, value in enumerate(values):
        axes.text(
            value + peak * 0.01, row, f"{value:,}", va="center", fontsize=8.5, color=INK_SECONDARY
        )
    axes.set_xlim(0, peak * 1.12)
    axes.tick_params(axis="y", labelcolor=INK, labelsize=9)
    _titles(figure, "Events per species", "Independent sightings after grouping bursts")
    return _save(figure, path)


def activity_chart(activity: pd.DataFrame, totals: pd.DataFrame, path: Path) -> Path:
    """Small multiples: share of each label's events by hour of day.

    Args:
        activity: Output of ``activity_by_hour``.
        totals: Output of ``species_totals`` (sets panel order).
        path: PNG destination.

    Returns:
        ``path``.
    """
    order = list(totals["species"].head(MAX_PANELS))
    rows = max(1, math.ceil(len(order) / PANEL_COLUMNS))
    figure = _new_figure(TITLE_IN + 1.9 * rows)
    shares = {}
    for species in order:
        counts = activity[activity["species"] == species].sort_values("hour")["n_events"]
        total = counts.sum()
        shares[species] = [100.0 * c / total if total else 0.0 for c in counts]
    ceiling = max((max(v) for v in shares.values()), default=10.0) * 1.1 or 10.0
    top_margin = (TITLE_IN - 0.2) / figure.get_figheight()
    cell_height = (1.0 - top_margin - 0.05) / rows
    for index, species in enumerate(order):
        row, column = divmod(index, PANEL_COLUMNS)
        left = 0.06 + column * 0.315
        bottom = 1.0 - top_margin - (row + 1) * cell_height + 0.06 / rows
        axes = figure.add_axes((left, bottom, 0.27, cell_height * 0.68))
        _style(axes, "y")
        for start, end in NIGHT_HOURS:
            axes.axvspan(start, end, color=EMPTY_CELL, linewidth=0, zorder=0)
        axes.bar(range(24), shares[species], width=0.7, color=SERIES, zorder=2)
        axes.set_xlim(-0.5, 23.5)
        axes.set_ylim(0, ceiling)
        axes.set_xticks([0, 6, 12, 18, 23])
        axes.yaxis.set_major_formatter("{x:.0f}%")
        n_events = int(totals.loc[totals["species"] == species, "n_events"].iloc[0])
        axes.set_title(f"{species}  ({n_events} events)", loc="left", fontsize=9, color=INK, pad=4)
    _titles(
        figure,
        "Activity by hour of day",
        "Share of each species' events by camera-local hour; shaded 20:00-06:00",
    )
    return _save(figure, path)


def species_camera_heatmap(by_camera: pd.DataFrame, totals: pd.DataFrame, path: Path) -> Path:
    """Heatmap of events per camera (rows) and label (columns).

    Args:
        by_camera: Output of ``species_by_camera``.
        totals: Output of ``species_totals`` (sets column order).
        path: PNG destination.

    Returns:
        ``path``.
    """
    columns = list(totals["species"].head(MAX_BAR_ROWS))
    grid = by_camera.pivot_table(
        index="camera_id", columns="species", values="n_events", aggfunc="sum", fill_value=0
    ).reindex(columns=columns, fill_value=0)
    cameras = list(grid.index)
    figure = _new_figure(TITLE_IN + 1.2 + 0.42 * max(len(cameras), 1))
    axes = figure.add_axes(_axes_box(figure, 0.14, 0.98, TITLE_IN, 1.2))
    axes.set_facecolor(SURFACE)
    for side in axes.spines.values():
        side.set_visible(False)
    peak = float(grid.to_numpy().max()) if grid.size else 0.0
    for row, camera in enumerate(cameras):
        for column, species in enumerate(columns):
            value = float(grid.loc[camera, species])
            fill = _ramp_color(value, peak) if value > 0 else EMPTY_CELL
            axes.add_patch(
                Rectangle((column, row), 1, 1, facecolor=fill, edgecolor=SURFACE, linewidth=2)
            )
            text = f"{value:.0f}" if value > 0 else "-"
            color = _text_on(fill) if value > 0 else INK_MUTED
            axes.text(
                column + 0.5, row + 0.5, text, ha="center", va="center", fontsize=8.5, color=color
            )
    axes.set_xlim(0, max(len(columns), 1))
    axes.set_ylim(max(len(cameras), 1), 0)
    axes.set_xticks([c + 0.5 for c in range(len(columns))], columns, rotation=35, ha="right")
    axes.set_yticks([r + 0.5 for r in range(len(cameras))], cameras)
    axes.tick_params(colors=INK, labelsize=8.5, length=0)
    _titles(figure, "Species by camera", "Events per camera; darker cells mean more events")
    return _save(figure, path)


def _draw_day_cells(
    axes: Axes,
    daily: pd.DataFrame,
    cameras: list[str],
    first: pd.Timestamp,
    n_days: int,
    peak: float,
) -> None:
    counts = daily.set_index(["camera_id", "date"])["n_photos"]
    for row, camera in enumerate(cameras):
        for offset in range(n_days):
            value = float(counts.get((camera, first + pd.Timedelta(days=offset)), 0))
            fill = _ramp_color(value, peak) if value > 0 else EMPTY_CELL
            axes.add_patch(
                Rectangle(
                    (offset, row + 0.1), 1, 0.6, facecolor=fill, edgecolor=SURFACE, linewidth=2
                )
            )


def _draw_silences(axes: Axes, gaps: pd.DataFrame, cameras: list[str], first: pd.Timestamp) -> None:
    columns = ["camera_id", "gap_start", "gap_end", "gap_hours"]
    for camera, start, end, hours in gaps[columns].itertuples(index=False):
        row = cameras.index(camera) + 0.85
        x0 = (start - first).total_seconds() / SECONDS_PER_DAY
        x1 = (end - first).total_seconds() / SECONDS_PER_DAY
        axes.plot([x0, x1], [row, row], color=STATUS_SERIOUS, linewidth=2, solid_capstyle="round")
        label = f"gap {hours / 24:.1f} d"
        axes.text(x1 + 0.2, row, label, va="center", fontsize=8, color=INK_SECONDARY)


def _timeline_legend(axes: Axes, peak: float) -> None:
    legend = [
        Patch(facecolor=EMPTY_CELL, label="no photos"),
        Patch(facecolor=SEQUENTIAL[3], label="few photos"),
        Patch(facecolor=SEQUENTIAL[-2], label=f"many (max {peak:.0f}/day)"),
        Patch(facecolor=STATUS_SERIOUS, label="silence over threshold"),
    ]
    axes.legend(
        handles=legend,
        loc="upper left",
        bbox_to_anchor=(0.0, -0.12),
        ncol=4,
        frameon=False,
        fontsize=8,
        labelcolor=INK_SECONDARY,
        handlelength=1.2,
    )


def camera_timeline_chart(
    daily: pd.DataFrame, gaps: pd.DataFrame, health: pd.DataFrame, path: Path
) -> Path:
    """Calendar strip per camera: photos per day, with long silences marked.

    Args:
        daily: Output of ``daily_photos``.
        gaps: Output of ``camera_gaps``.
        health: Output of ``camera_health`` (camera order and invalid-clock counts).
        path: PNG destination.

    Returns:
        ``path``.
    """
    cameras = list(health["camera_id"])
    figure = _new_figure(TITLE_IN + 0.8 + 0.55 * max(len(cameras), 1))
    axes = figure.add_axes(_axes_box(figure, 0.12, 0.97, TITLE_IN, 0.8))
    _style(axes, "x")
    axes.grid(False)
    if daily.empty:
        return _save(figure, path)
    first = daily["date"].min()
    n_days = int((daily["date"].max() - first).days) + 1
    peak = float(daily["n_photos"].max())
    _draw_day_cells(axes, daily, cameras, first, n_days, peak)
    _draw_silences(axes, gaps, cameras, first)
    axes.set_xlim(0, n_days)
    axes.set_ylim(len(cameras), 0)
    ticks = list(range(0, n_days, max(1, n_days // 8)))
    axes.set_xticks(
        [t + 0.5 for t in ticks], [(first + pd.Timedelta(days=t)).strftime("%b %d") for t in ticks]
    )
    axes.set_yticks([r + 0.4 for r in range(len(cameras))], cameras)
    axes.tick_params(axis="y", labelcolor=INK, labelsize=9)
    _timeline_legend(axes, peak)
    invalid = int(health["n_invalid_timestamp"].sum())
    note = f"; {invalid} photo(s) with an unset clock not shown" if invalid else ""
    _titles(figure, "Camera uptime", f"Photos per camera per day{note}")
    return _save(figure, path)


def render_all(tables: dict[str, pd.DataFrame], plot_dir: Path) -> dict[str, Path]:
    """Draw every chart the report uses.

    Args:
        tables: Output of ``summarize.build_summaries``.
        plot_dir: Folder for the PNG files.

    Returns:
        Chart name to file path.
    """
    totals = tables["species_totals"]
    return {
        "species_events": species_events_chart(totals, plot_dir / "species_events.png"),
        "activity_by_hour": activity_chart(
            tables["activity_by_hour"], totals, plot_dir / "activity_by_hour.png"
        ),
        "species_by_camera": species_camera_heatmap(
            tables["species_by_camera"], totals, plot_dir / "species_by_camera.png"
        ),
        "camera_uptime": camera_timeline_chart(
            tables["daily_photos"],
            tables["camera_gaps"],
            tables["camera_health"],
            plot_dir / "camera_uptime.png",
        ),
    }
