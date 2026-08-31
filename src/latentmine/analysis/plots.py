"""Figures.

The three products of LLD section 7.2, plus the rollout panel of 7.1. All are
drawn in world coordinates via `mazes.render`'s helpers, so the row->x /
column->y convention is applied in exactly one place.

Colour follows the job the data does: single-hue sequential ramps for
magnitude (latent distance, geodesic distance), a validated categorical set
for regions, and a two-hue diverging ramp with a neutral midpoint for signed
differences. No rainbow anywhere - a rainbow ramp on a distance field invents
banding that reads as structure.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from ..mazes import geometry as geo  # noqa: E402
from ..mazes import render as mr  # noqa: E402
from ..mazes.layouts import MazeSpec  # noqa: E402

LATENT_CMAP = "Purples"  # sequential: latent distance
GEODESIC_CMAP = "Blues"  # sequential: true distance
DIVERGING_CMAP = "RdBu_r"  # diverging: signed difference, neutral midpoint
TEXT_PRIMARY = mr.TEXT_PRIMARY
TEXT_SECONDARY = mr.TEXT_SECONDARY
SURFACE = "#fcfcfb"


def _new_figure(ncols: int, width: float = 4.5, height: float = 4.6):
    fig, axes = plt.subplots(1, ncols, figsize=(width * ncols, height))
    fig.patch.set_facecolor(SURFACE)
    return fig, np.atleast_1d(axes)


def _save(fig, out: Path) -> Path:
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    return out


def latent_distance_field(
    spec: MazeSpec,
    d_latent_from_anchor: np.ndarray,
    anchor_cell: tuple[int, int],
    out: Path,
    title: str = "",
) -> Path:
    """The single most convincing one-panel result (LLD 7.2).

    Latent distance from one anchor beside the true geodesic distance from the
    same anchor, with the geodesic iso-contours overlaid on *both*. If the
    latent respects walls the two contour families agree; if it only knows
    position, the latent panel's contours are circles that ignore the wall.
    """
    anchor_index = list(spec.free_cells()).index(anchor_cell)
    geodesic = geo.geodesic_from(spec, anchor_cell)

    fig, axes = _new_figure(2)
    mr._draw_field(
        axes[0], spec, np.asarray(d_latent_from_anchor), LATENT_CMAP, "latent distance", contours=True
    )
    mr._draw_field(axes[1], spec, geodesic, GEODESIC_CMAP, "true geodesic distance", contours=True)
    for ax in axes:
        x, y = geo.cell_to_world(anchor_cell, spec.scaling)
        ax.plot(x, y, marker="*", ms=15, mfc="#ffffff", mec=TEXT_PRIMARY, mew=1.2, zorder=6)

    fig.suptitle(
        title or f"{spec.name}: latent vs geodesic from cell {anchor_cell}", fontsize=11, color=TEXT_PRIMARY
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    assert anchor_index >= 0
    return _save(fig, out)


def latent_map(
    spec: MazeSpec,
    coords: np.ndarray,
    cell_index: np.ndarray,
    out: Path,
    title: str = "",
    explained: float | None = None,
) -> Path:
    """The projected latent space, and the maze coloured by it.

    Left: each sample at its projected position, coloured by where it is in
    the maze. Right: the maze, each cell coloured by its projected coordinate.
    The right panel is the more legible direction - it shows the maze
    segmenting itself - and is the one most likely to reach the report.
    """
    cells = geo.free_cell_array(spec)
    world = geo.cells_to_world(cells, spec.scaling)[cell_index]
    rgb = _position_to_rgb(world, spec)

    fig, axes = _new_figure(2)
    axes[0].scatter(coords[:, 0], coords[:, 1], c=rgb, s=26, edgecolors="none")
    axes[0].set_title("latent projection, coloured by maze position", fontsize=9, color=TEXT_PRIMARY)
    axes[0].set_xlabel("component 1", fontsize=8, color=TEXT_SECONDARY)
    axes[0].set_ylabel("component 2", fontsize=8, color=TEXT_SECONDARY)
    axes[0].tick_params(labelsize=7, colors=TEXT_SECONDARY, length=2)
    for side in axes[0].spines.values():
        side.set_visible(False)
    axes[0].set_aspect("equal", adjustable="datalim")

    # Inverse map: colour each cell by its projected coordinate.
    projected_rgb = _coords_to_rgb(coords)
    grid = np.full((*spec.shape, 3), np.nan)
    for k, cell_k in enumerate(cell_index):
        grid[tuple(cells[cell_k])] = projected_rgb[k]
    grid = np.where(np.isnan(grid), matplotlib.colors.to_rgb(mr.WALL_COLOR), grid)
    mr._show(axes[1], spec, grid)
    mr._style_axes(axes[1], spec)
    axes[1].set_title("maze, coloured by latent coordinate", fontsize=9, color=TEXT_PRIMARY)

    suffix = f"   (2 components explain {explained:.0%} of variance)" if explained is not None else ""
    fig.suptitle((title or f"{spec.name}: latent map") + suffix, fontsize=11, color=TEXT_PRIMARY)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return _save(fig, out)


def _position_to_rgb(world: np.ndarray, spec: MazeSpec) -> np.ndarray:
    """2-D colourmap over maze position, so a projection can be read for
    continuity: neighbouring colours mean neighbouring cells."""
    x_min, x_max, y_min, y_max = geo.world_extent(spec)
    u = (world[:, 0] - x_min) / max(x_max - x_min, 1e-9)
    v = (world[:, 1] - y_min) / max(y_max - y_min, 1e-9)
    return np.stack([0.15 + 0.75 * u, 0.25 + 0.5 * v, 0.9 - 0.7 * u], axis=1).clip(0, 1)


def _coords_to_rgb(coords: np.ndarray) -> np.ndarray:
    span = coords.ptp(axis=0)
    span[span == 0] = 1.0
    normalised = (coords - coords.min(axis=0)) / span
    u, v = normalised[:, 0], normalised[:, 1]
    return np.stack([0.15 + 0.75 * u, 0.25 + 0.5 * v, 0.9 - 0.7 * u], axis=1).clip(0, 1)


def rollout_panel(
    spec: MazeSpec,
    positions: np.ndarray,
    latent_to_goal: np.ndarray,
    true_to_goal: np.ndarray,
    goal_xy: np.ndarray,
    out: Path,
    title: str = "",
) -> Path:
    """Trajectory, and the critic's own sense of progress along it (LLD 7.1).

    A correct picture has latent distance falling as the agent approaches. If
    it does not, the checkpoint is broken and nothing downstream is meaningful
    - this is the sanity check that runs before any of the fancier analysis.
    """
    fig, axes = _new_figure(2, width=5.0)

    mr._draw_layout(axes[0], spec, annotate=False)
    steps = np.arange(len(positions))
    axes[0].scatter(positions[:, 0], positions[:, 1], c=steps, cmap="cividis", s=9, zorder=4)
    axes[0].plot(*goal_xy, marker="*", ms=16, mfc="#ffffff", mec=TEXT_PRIMARY, mew=1.2, zorder=6)
    axes[0].set_title("trajectory, coloured by timestep", fontsize=9, color=TEXT_PRIMARY)

    ax = axes[1]
    ax.plot(steps, latent_to_goal, lw=2, color="#4a3aa7", label="latent distance to goal")
    twin = ax.twinx()
    twin.plot(steps, true_to_goal, lw=2, color="#9a9a94", ls="--", label="true distance to goal")
    ax.set_xlabel("timestep", fontsize=8, color=TEXT_SECONDARY)
    ax.set_ylabel("latent distance", fontsize=8, color="#4a3aa7")
    twin.set_ylabel("world distance", fontsize=8, color="#6b6b66")
    ax.tick_params(labelsize=7, colors=TEXT_SECONDARY, length=2)
    twin.tick_params(labelsize=7, colors=TEXT_SECONDARY, length=2)
    for side in list(ax.spines.values()) + list(twin.spines.values()):
        side.set_visible(False)
    lines = ax.get_lines() + twin.get_lines()
    ax.legend(lines, [line.get_label() for line in lines], fontsize=7, frameon=False, loc="upper right")
    ax.set_title("does the critic know it is getting closer?", fontsize=9, color=TEXT_PRIMARY)

    fig.suptitle(title or f"{spec.name}: rollout and its latent", fontsize=11, color=TEXT_PRIMARY)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return _save(fig, out)


def metric_summary(summaries: list[dict], out: Path, title: str = "") -> Path:
    """One bar per maze for the headline metrics, with seed spread as error bars.

    Two panels because the two numbers answer different questions: the partial
    correlation asks whether the latent knows anything beyond position, the
    wall-crossing ratio asks how hard it separates cells across a wall.
    """
    summaries = [s for s in summaries if s]
    if not summaries:
        raise ValueError("no summaries to plot")
    names = [s["maze"] for s in summaries]
    ticks = np.arange(len(names))

    fig, axes = _new_figure(2, width=5.2, height=4.0)
    for ax, key, label, colour in (
        (axes[0], "partial_rho_geodesic", "partial rho (geodesic | euclidean)", "#2a78d6"),
        (axes[1], "wall_crossing_ratio", "wall-crossing ratio", "#eb6834"),
    ):
        means = [s[key]["mean"] for s in summaries]
        errors = [s[key]["std"] for s in summaries]
        ax.bar(ticks, means, yerr=errors, capsize=3, color=colour, width=0.6)
        ax.set_xticks(ticks)
        ax.set_xticklabels(names, rotation=30, ha="right", fontsize=8, color=TEXT_SECONDARY)
        ax.set_title(label, fontsize=9, color=TEXT_PRIMARY)
        ax.tick_params(labelsize=7, colors=TEXT_SECONDARY, length=2)
        ax.grid(axis="y", lw=0.5, color="#e8e8e3")
        ax.set_axisbelow(True)
        for side in ax.spines.values():
            side.set_visible(False)
        for x, (mean, error) in enumerate(zip(means, errors, strict=True)):
            if np.isfinite(mean):
                ax.text(
                    x,
                    mean + (error if np.isfinite(error) else 0) + 0.02,
                    f"{mean:.2f}",
                    ha="center",
                    fontsize=7,
                    color=TEXT_PRIMARY,
                )
            else:
                # An absent bar must not read as a value of zero. open_room and
                # loop have no pairs facing each other through a wall, so the
                # ratio is undefined there rather than low.
                ax.text(
                    x,
                    0.02 * ax.get_ylim()[1],
                    "n/a",
                    ha="center",
                    fontsize=7,
                    style="italic",
                    color=TEXT_SECONDARY,
                )
    axes[1].axhline(1.0, color=TEXT_SECONDARY, lw=1, ls=":")
    axes[1].text(-0.45, 1.04, "1.0 = no wall awareness", fontsize=6.5, color=TEXT_SECONDARY, ha="left")

    fig.suptitle(title or "headline metrics, mean +/- sd over seeds", fontsize=11, color=TEXT_PRIMARY)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    return _save(fig, out)
