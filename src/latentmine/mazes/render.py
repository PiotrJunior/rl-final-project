"""Render the maze set to PNG - the first milestone of the build order.

Three panels per maze: the layout with its region overlay, the geodesic
distance field from the start cell, and betweenness centrality. The distance
field is a dry run of the figure that decides the whole research question
(LLD section 7.2), with true geodesics standing in for latent distance.

Everything is drawn in **world coordinates**, so grid row `i` runs along the
horizontal axis and grid column `j` runs up the vertical one, and the picture
is the ASCII art transposed. That is the convention every later figure uses
(see `geometry`), so eyeballing these plots is what catches a transposition
before it silently inverts the project.

    python -m latentmine.mazes.render --out figures/mazes
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from . import geometry as geo  # noqa: E402
from .layouts import DOORWAY, MAZES, MazeSpec, get, names  # noqa: E402

# Categorical slots 1-4 of the reference palette, validated for CVD separation.
# Regions are always direct-labelled with their letter, which is also what
# discharges the palette's contrast warning.
REGION_COLORS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")
DOORWAY_COLOR = "#9a9a94"  # neutral: a doorway is "excluded", not a category
WALL_COLOR = "#2b2b28"
FREE_COLOR = "#e8e8e3"
START_COLOR = "#0b0b0b"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"


def _extent(spec: MazeSpec) -> tuple[float, float, float, float]:
    """imshow extent in world units, padded by half a cell each side."""
    h = spec.scaling / 2.0
    return (-h, (spec.n_rows - 1) * spec.scaling + h, -h, (spec.n_cols - 1) * spec.scaling + h)


def _field(spec: MazeSpec, values: np.ndarray) -> np.ndarray:
    """Scatter per-free-cell values into an `(n_rows, n_cols)` grid, NaN at walls."""
    grid = np.full(spec.shape, np.nan)
    for k, (i, j) in enumerate(spec.free_cells()):
        grid[i, j] = values[k]
    return grid


def _show(ax, spec: MazeSpec, grid: np.ndarray, **kwargs):
    """imshow with the row->x / column->y convention applied exactly once.

    `grid` is indexed `[i][j]`; swapping those two axes puts `j` on imshow's
    vertical axis and `i` on its horizontal one, which is the convention.
    `swapaxes` rather than `.T` so an RGB `(i, j, 3)` array keeps its colour
    axis last.
    """
    return ax.imshow(
        np.swapaxes(grid, 0, 1), origin="lower", extent=_extent(spec), interpolation="nearest", **kwargs
    )


def _style_axes(ax, spec: MazeSpec) -> None:
    ax.set_aspect("equal")
    ax.set_xlabel("world x   (grid row i)", fontsize=8, color=TEXT_SECONDARY)
    ax.set_ylabel("world y   (grid col j)", fontsize=8, color=TEXT_SECONDARY)
    ax.tick_params(labelsize=7, colors=TEXT_SECONDARY, length=2)
    for side in ax.spines.values():
        side.set_visible(False)


def _draw_layout(ax, spec: MazeSpec, annotate: bool = True) -> None:
    """Walls, regions and start cells, with each region direct-labelled."""
    labels = spec.region_labels()
    colour_of = {lab: REGION_COLORS[k % len(REGION_COLORS)] for k, lab in enumerate(labels)}

    rgb = np.zeros((*spec.shape, 3))
    for i in range(spec.n_rows):
        for j in range(spec.n_cols):
            if spec.is_wall(i, j):
                hexc = WALL_COLOR
            elif spec.regions is None:
                hexc = FREE_COLOR
            elif spec.regions[i][j] == DOORWAY:
                hexc = DOORWAY_COLOR
            else:
                hexc = colour_of.get(spec.regions[i][j], FREE_COLOR)
            rgb[i, j] = matplotlib.colors.to_rgb(hexc)
    _show(ax, spec, rgb)

    # Direct labels at each region's centroid - identity is never colour-alone.
    for lab in labels:
        cells = np.array(spec.cells_in_region(lab), dtype=float)
        cx, cy = geo.cells_to_world(cells, spec.scaling).mean(axis=0)
        ax.text(cx, cy, lab, ha="center", va="center", fontsize=13, color="white", fontweight="bold")

    for cell in spec.start_cells():
        x, y = geo.cell_to_world(cell, spec.scaling)
        ax.plot(x, y, marker="o", ms=9, mfc=START_COLOR, mec="white", mew=1.4, zorder=5)
        if not annotate:
            continue
        ax.annotate(
            f"start  grid {cell} -> world ({x:.0f}, {y:.0f})",
            xy=(x, y),
            xytext=(6, 8),
            textcoords="offset points",
            fontsize=6.5,
            color=TEXT_PRIMARY,
        )
    _style_axes(ax, spec)
    ax.set_title("layout & regions", fontsize=9, color=TEXT_PRIMARY)


def _draw_field(ax, spec: MazeSpec, values: np.ndarray, cmap: str, title: str, contours: bool) -> None:
    grid = _field(spec, values)
    im = _show(ax, spec, grid, cmap=cmap)

    if contours:
        finite = np.isfinite(grid)
        if finite.any():
            xs = np.arange(spec.n_rows) * spec.scaling
            ys = np.arange(spec.n_cols) * spec.scaling
            # Masked, not filled: contour must not interpolate across a wall,
            # or a one-cell corridor grows contour lines that jump the blocks.
            masked = np.ma.masked_invalid(grid.T)
            ax.contour(xs, ys, masked, levels=8, colors="white", linewidths=0.7, alpha=0.75)

    # Walls painted last so contours are clipped to the free space rather than
    # trailing across the blocks.
    wall = np.where(geo.occupancy(spec), 1.0, np.nan)
    _show(ax, spec, wall, cmap=matplotlib.colors.ListedColormap([WALL_COLOR]), vmin=0, vmax=1, zorder=3)

    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.ax.tick_params(labelsize=6, colors=TEXT_SECONDARY, length=2)
    cb.outline.set_visible(False)
    _style_axes(ax, spec)
    ax.set_title(title, fontsize=9, color=TEXT_PRIMARY)


def render_maze(spec: MazeSpec, out_dir: Path) -> Path:
    """Write `<out_dir>/<name>.png` and return its path."""
    start = spec.start_cells()[0]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.6))
    fig.patch.set_facecolor("#fcfcfb")

    _draw_layout(axes[0], spec)
    _draw_field(
        axes[1],
        spec,
        geo.geodesic_from(spec, start),
        "Blues",
        f"geodesic distance from {start}",
        contours=True,
    )
    _draw_field(
        axes[2], spec, geo.betweenness_centrality(spec), "Oranges", "betweenness centrality", contours=False
    )

    free = len(spec.free_cells())
    artic = len(geo.articulation_points(spec))
    fig.suptitle(
        f"{spec.name}   -   {spec.n_rows}x{spec.n_cols} grid, {free} free cells, "
        f"{artic} articulation points, scaling {spec.scaling:g}",
        fontsize=11,
        color=TEXT_PRIMARY,
    )
    fig.tight_layout(rect=(0, 0.02, 1, 0.94))
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{spec.name}.png"
    fig.savefig(path, dpi=140, facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


def render_overview(out_dir: Path) -> Path:
    """One row of every maze's layout, for a whole-set glance."""
    specs = [get(n) for n in names()]
    fig, axes = plt.subplots(1, len(specs), figsize=(3.1 * len(specs), 3.6))
    fig.patch.set_facecolor("#fcfcfb")
    for ax, spec in zip(axes, specs, strict=True):
        _draw_layout(ax, spec, annotate=False)
        ax.set_title(spec.name, fontsize=9, color=TEXT_PRIMARY)
        ax.set_xlabel("")
        ax.set_ylabel("")
    fig.suptitle(
        "maze set - world coordinates: grid row i -> x (horizontal), grid col j -> y (vertical)",
        fontsize=10,
        color=TEXT_PRIMARY,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "overview.png"
    fig.savefig(path, dpi=140, facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("figures/mazes"))
    parser.add_argument("--maze", action="append", help="render only this maze (repeatable)")
    args = parser.parse_args(argv)

    chosen = args.maze or list(names())
    for name in chosen:
        print(f"wrote {render_maze(MAZES[name] if name in MAZES else get(name), args.out)}")
    if not args.maze:
        print(f"wrote {render_overview(args.out)}")


if __name__ == "__main__":
    main()
