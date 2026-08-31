"""The metrics that make a negative result reportable.

The proposal explicitly permits a negative answer to "is the maze visible in
the latent space?". That is only worth anything if it is quantitative, so
every claim in the write-up should cite one of these (LLD section 6).

Naming throughout: `d_geo` is the geodesic distance through the maze, `d_euc`
the straight-line world distance ignoring walls, and `d_lat` the distance
between goal latents under the critic's own geometry.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from ..mazes import geometry as geo
from ..mazes.layouts import DOORWAY, MazeSpec


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    from scipy.stats import spearmanr

    return float(spearmanr(np.asarray(a).ravel(), np.asarray(b).ravel()).statistic)


def _upper_triangle(matrix: np.ndarray) -> np.ndarray:
    iu = np.triu_indices_from(matrix, k=1)
    return matrix[iu]


# ---------------------------------------------------------------------------
# A. geodesic vs Euclidean (the headline)
# ---------------------------------------------------------------------------


@dataclass
class GeometryResult:
    """How well latent distance tracks the maze rather than the plane."""

    rho_geodesic: float
    rho_euclidean: float
    gap: float
    partial_rho_geodesic: float
    n_pairs: int

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def geometry_correlation(d_lat: np.ndarray, d_geo: np.ndarray, d_euc: np.ndarray) -> GeometryResult:
    """Spearman correlation of latent distance with geodesic and Euclidean.

    `gap` is the headline number, but on its own it is weak: geodesic and
    Euclidean distance are themselves strongly correlated, so a latent that
    only knows position still scores well against `d_geo`.
    `partial_rho_geodesic` is the statistic that actually isolates "knows
    about walls" - the correlation with `d_geo` after linearly removing
    `d_euc` from both, computed on ranks.
    """
    lat, gd, eu = (_upper_triangle(np.asarray(m, dtype=float)) for m in (d_lat, d_geo, d_euc))
    finite = np.isfinite(lat) & np.isfinite(gd) & np.isfinite(eu)
    lat, gd, eu = lat[finite], gd[finite], eu[finite]

    rho_geo, rho_euc = _spearman(lat, gd), _spearman(lat, eu)
    return GeometryResult(
        rho_geodesic=rho_geo,
        rho_euclidean=rho_euc,
        gap=rho_geo - rho_euc,
        partial_rho_geodesic=_partial_spearman(lat, gd, eu),
        n_pairs=int(lat.size),
    )


def _partial_spearman(x: np.ndarray, y: np.ndarray, control: np.ndarray) -> float:
    """Spearman correlation of `x` and `y` with `control` partialled out."""
    from scipy.stats import rankdata

    rx, ry, rc = (rankdata(v).astype(float) for v in (x, y, control))

    def residual(v):
        design = np.stack([rc, np.ones_like(rc)], axis=1)
        coefficients, *_ = np.linalg.lstsq(design, v, rcond=None)
        return v - design @ coefficients

    ex, ey = residual(rx), residual(ry)
    denominator = np.linalg.norm(ex) * np.linalg.norm(ey)
    return float(ex @ ey / denominator) if denominator > 0 else float("nan")


# ---------------------------------------------------------------------------
# B. wall crossing (the sharp version)
# ---------------------------------------------------------------------------


@dataclass
class WallCrossingResult:
    ratio: float
    median_across_wall: float
    median_same_side: float
    n_across: int
    n_same: int

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def wall_crossing_ratio(
    spec: MazeSpec,
    d_lat: np.ndarray,
    detour_threshold: float = 3.0,
    euclid_cells: float = 2.5,
) -> WallCrossingResult:
    """Latent separation of cells that are near in space but far through the maze.

    Selects pairs within `euclid_cells` of each other whose geodesic detour
    exceeds `detour_threshold` - cells facing each other through a wall - and
    compares their median latent distance with same-side pairs at a comparable
    Euclidean distance. A ratio near 1 means the encoder ignores walls; large
    means it respects them. One number per maze, and the cleanest headline
    result in the project.
    """
    d_euc = geo.euclidean_matrix(spec)
    detour = geo.detour_ratio(spec)
    near = d_euc <= euclid_cells * spec.scaling
    off_diagonal = ~np.eye(len(d_euc), dtype=bool)

    across = near & (detour >= detour_threshold) & off_diagonal
    same = near & (detour < 1.5) & off_diagonal

    lat = np.asarray(d_lat, dtype=float)
    across_values, same_values = lat[across], lat[same]
    if across_values.size == 0:
        # No wall-facing pairs: open_room, by construction. Reported as NaN
        # rather than as a ratio of 1, which would read as a real measurement.
        return WallCrossingResult(
            float("nan"),
            float("nan"),
            float(np.median(same_values)) if same_values.size else float("nan"),
            0,
            int(same_values.size),
        )
    median_across = float(np.median(across_values))
    median_same = float(np.median(same_values)) if same_values.size else float("nan")
    ratio = median_across / median_same if median_same and np.isfinite(median_same) else float("nan")
    return WallCrossingResult(
        ratio=ratio,
        median_across_wall=median_across,
        median_same_side=median_same,
        n_across=int(across_values.size),
        n_same=int(same_values.size),
    )


# ---------------------------------------------------------------------------
# C. room purity
# ---------------------------------------------------------------------------


@dataclass
class PurityResult:
    latent_purity: float
    position_purity: float
    lift: float
    k: int
    n_cells: int

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def room_purity(spec: MazeSpec, d_lat: np.ndarray, k: int = 10) -> PurityResult:
    """Fraction of each cell's `k` nearest latent neighbours in the same room.

    Compared against the same statistic on raw `(x, y)`, which is the "no
    information beyond position" null: in `open_room` the two must agree,
    because there is no wall for the latent to know about. `lift` is the
    difference, and it is the number worth reporting.

    Doorway cells are excluded - their room membership is genuinely ambiguous,
    and counting them as failures of whichever room they are not assigned to
    would understate the metric.
    """
    if spec.regions is None:
        raise ValueError(f"{spec.name}: no regions overlay, room purity is undefined")

    cells = list(spec.free_cells())
    keep = np.array([spec.regions[i][j] != DOORWAY for (i, j) in cells])
    labels = np.array([spec.regions[i][j] for (i, j) in cells])[keep]

    lat = np.asarray(d_lat, dtype=float)[np.ix_(keep, keep)]
    pos = geo.euclidean_matrix(spec)[np.ix_(keep, keep)]

    return PurityResult(
        latent_purity=_knn_purity(lat, labels, k),
        position_purity=_knn_purity(pos, labels, k),
        lift=_knn_purity(lat, labels, k) - _knn_purity(pos, labels, k),
        k=k,
        n_cells=int(keep.sum()),
    )


def _knn_purity(distance: np.ndarray, labels: np.ndarray, k: int) -> float:
    n = len(labels)
    k = min(k, n - 1)
    if k < 1:
        return float("nan")
    d = distance.copy()
    np.fill_diagonal(d, np.inf)
    neighbours = np.argsort(d, axis=1)[:, :k]
    return float((labels[neighbours] == labels[:, None]).mean())


# ---------------------------------------------------------------------------
# D. projection faithfulness
# ---------------------------------------------------------------------------


def trustworthiness(d_high: np.ndarray, d_low: np.ndarray, k: int = 10) -> float:
    """Penalty for points the projection pulled together that were far apart.

    Without this, a bad 2-D picture cannot be distinguished from a latent
    space with no structure in it - and those are very different findings.
    """
    return _rank_penalty(d_high, d_low, k)


def continuity(d_high: np.ndarray, d_low: np.ndarray, k: int = 10) -> float:
    """The dual: penalty for true neighbours the projection pushed apart."""
    return _rank_penalty(d_low, d_high, k)


def _rank_penalty(d_a: np.ndarray, d_b: np.ndarray, k: int) -> float:
    n = len(d_a)
    k = min(k, n - 2)
    if k < 1:
        return float("nan")
    a, b = d_a.copy(), d_b.copy()
    np.fill_diagonal(a, np.inf)
    np.fill_diagonal(b, np.inf)

    rank_in_a = np.argsort(np.argsort(a, axis=1), axis=1)
    neighbours_b = np.argsort(b, axis=1)[:, :k]

    total = 0.0
    for i in range(n):
        ranks = rank_in_a[i, neighbours_b[i]]
        intruders = ranks[ranks >= k]
        total += np.sum(intruders - k + 1)
    norm = 2.0 / (n * k * (2 * n - 3 * k - 1))
    return float(1.0 - norm * total)


# ---------------------------------------------------------------------------
# assembling a report
# ---------------------------------------------------------------------------


@dataclass
class MazeReport:
    """Every metric for one (maze, checkpoint) pair."""

    maze: str
    run_id: str
    step: int
    geometry: GeometryResult
    wall_crossing: WallCrossingResult
    purity: PurityResult | None
    projection: dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "maze": self.maze,
            "run_id": self.run_id,
            "step": self.step,
            "geometry": self.geometry.as_dict(),
            "wall_crossing": self.wall_crossing.as_dict(),
            "purity": self.purity.as_dict() if self.purity else None,
            "projection": self.projection,
        }


def evaluate(
    spec: MazeSpec,
    d_lat: np.ndarray,
    run_id: str = "",
    step: int = 0,
    k: int = 10,
) -> MazeReport:
    """Metrics A-C for one latent distance matrix over the free cells."""
    d_geo = geo.geodesic_matrix(spec)
    d_euc = geo.euclidean_matrix(spec)
    return MazeReport(
        maze=spec.name,
        run_id=run_id,
        step=step,
        geometry=geometry_correlation(d_lat, d_geo, d_euc),
        wall_crossing=wall_crossing_ratio(spec, d_lat),
        purity=room_purity(spec, d_lat, k=k) if spec.regions is not None else None,
    )


def aggregate(reports: list[MazeReport]) -> dict[str, Any]:
    """Mean and standard deviation across seeds.

    Everything in the write-up is reported over at least three seeds, and a
    finding that does not survive them is reported as not surviving.
    """
    if not reports:
        return {}

    def stat(path: tuple[str, str]) -> dict[str, float]:
        values = []
        for report in reports:
            section = getattr(report, path[0])
            if section is not None:
                values.append(getattr(section, path[1]))
        values = [v for v in values if v is not None and np.isfinite(v)]
        if not values:
            return {"mean": float("nan"), "std": float("nan"), "n": 0}
        return {"mean": float(np.mean(values)), "std": float(np.std(values)), "n": len(values)}

    return {
        "maze": reports[0].maze,
        "n_seeds": len(reports),
        "rho_geodesic": stat(("geometry", "rho_geodesic")),
        "rho_euclidean": stat(("geometry", "rho_euclidean")),
        "gap": stat(("geometry", "gap")),
        "partial_rho_geodesic": stat(("geometry", "partial_rho_geodesic")),
        "wall_crossing_ratio": stat(("wall_crossing", "ratio")),
        "purity_lift": stat(("purity", "lift")),
    }
