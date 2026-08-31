"""Reconstruct a maze's walls from its latent space alone.

The advanced extension of LLD section 9b. For every pair of spatially adjacent
cells, decide whether the passage between them is open by asking whether their
latents are close. The interesting question is not whether this works on the
maze it was calibrated on - it is whether a threshold calibrated on one maze
transfers to another without recalibration, which is what `calibrate` and
`reconstruct` are separated for.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..mazes import geometry as geo
from ..mazes.layouts import MazeSpec


@dataclass
class Reconstruction:
    """A predicted occupancy grid and how it compares with the truth."""

    open_edges: dict[tuple[tuple[int, int], tuple[int, int]], bool]
    threshold: float
    edge_f1: float = float("nan")
    edge_precision: float = float("nan")
    edge_recall: float = float("nan")
    occupancy_iou: float = float("nan")
    predicted_occupancy: np.ndarray | None = None


def _candidate_edges(spec: MazeSpec) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    """Every orthogonally adjacent pair of *free* cells, plus pairs separated
    by exactly one wall cell.

    Both are needed: the first are passages that exist, the second are the
    passages a wall blocks. Scoring on only the first would make "everything is
    open" a perfect answer.
    """
    edges = []
    for i, j in spec.free_cells():
        for di, dj in ((0, 1), (1, 0)):
            near = (i + di, j + dj)
            far = (i + 2 * di, j + 2 * dj)
            if _inside(spec, near) and spec.is_free(*near):
                edges.append(((i, j), near))
            elif _inside(spec, near) and _inside(spec, far) and spec.is_free(*far):
                edges.append(((i, j), far))
    return edges


def _inside(spec: MazeSpec, cell) -> bool:
    return 0 <= cell[0] < spec.n_rows and 0 <= cell[1] < spec.n_cols


def latent_scale(d_lat: np.ndarray) -> float:
    """Median pairwise latent distance - the maze's own unit of length.

    Uses no knowledge of the walls, only the distance matrix, so it is
    available for a maze we are trying to reconstruct.
    """
    d = np.asarray(d_lat, dtype=float)
    off_diagonal = d[~np.eye(len(d), dtype=bool)]
    return float(np.median(off_diagonal))


def calibrate(spec: MazeSpec, d_lat: np.ndarray, quantile: float = 0.5) -> float:
    """Calibrate a **scale-free** threshold from pairs known to be open.

    Returns the threshold as a multiple of `latent_scale`, not as an absolute
    distance. That matters for transfer, and the reason is measured rather
    than assumed: latent distance has no common unit across mazes. On a
    geodesic-embedded latent, adjacent cells sit 5.41 apart in `two_rooms` and
    11.31 apart in `loop`, so `two_rooms`' absolute threshold declares every
    passage in `loop` closed and reconstruction collapses to edge-F1 0.00.
    Dividing by each maze's own median pairwise distance removes that, and
    still uses no ground truth about the target's walls.

    Calibration itself sees only 4-adjacent free-cell pairs - passages we know
    exist in the *source* maze - never the walls being looked for.
    """
    index = {cell: k for k, cell in enumerate(spec.free_cells())}
    known_open = [
        d_lat[index[cell], index[nb]]
        for cell in spec.free_cells()
        for nb, weight in geo.neighbours(spec, cell, connectivity=4)
        if weight == 1.0
    ]
    if not known_open:
        raise ValueError(f"{spec.name}: no adjacent free pairs to calibrate on")
    return float(np.quantile(known_open, quantile) * 2.0 / latent_scale(d_lat))


def reconstruct(spec: MazeSpec, d_lat: np.ndarray, threshold_ratio: float) -> Reconstruction:
    """Declare each candidate passage open iff the latents are close enough.

    `threshold_ratio` is in units of this maze's own `latent_scale`, so a
    ratio calibrated elsewhere carries over.
    """
    index = {cell: k for k, cell in enumerate(spec.free_cells())}
    threshold = threshold_ratio * latent_scale(d_lat)
    predicted = {}
    for a, b in _candidate_edges(spec):
        if a not in index or b not in index:
            continue
        predicted[(a, b)] = bool(d_lat[index[a], index[b]] <= threshold)
    return _score(spec, predicted, threshold)


def _score(spec: MazeSpec, predicted: dict, threshold: float) -> Reconstruction:
    """Compare against the true passages."""
    truth = {}
    for a, b in predicted:
        step = (abs(a[0] - b[0]), abs(a[1] - b[1]))
        if step in ((0, 1), (1, 0)):
            truth[(a, b)] = True  # directly adjacent free cells
        else:
            between = ((a[0] + b[0]) // 2, (a[1] + b[1]) // 2)
            truth[(a, b)] = spec.is_free(*between)

    keys = list(predicted)
    p = np.array([predicted[k] for k in keys])
    t = np.array([truth[k] for k in keys])
    true_positive = int((p & t).sum())
    precision = true_positive / max(int(p.sum()), 1)
    recall = true_positive / max(int(t.sum()), 1)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0

    occupancy = _occupancy_from_edges(spec, predicted)
    truth_occupancy = geo.occupancy(spec)
    intersection = np.logical_and(occupancy, truth_occupancy).sum()
    union = np.logical_or(occupancy, truth_occupancy).sum()

    return Reconstruction(
        open_edges=predicted,
        threshold=threshold,
        edge_f1=float(f1),
        edge_precision=float(precision),
        edge_recall=float(recall),
        occupancy_iou=float(intersection / union) if union else float("nan"),
        predicted_occupancy=occupancy,
    )


def _occupancy_from_edges(spec: MazeSpec, predicted: dict) -> np.ndarray:
    """Wall grid implied by the closed passages.

    A cell between two free cells is called a wall when the passage through it
    is predicted closed. Cells the latent never saw stay walls, which matches
    the ground truth: they are walls.
    """
    occupancy = np.ones(spec.shape, dtype=bool)
    for cell in spec.free_cells():
        occupancy[cell] = False
    for (a, b), is_open in predicted.items():
        step = (abs(a[0] - b[0]), abs(a[1] - b[1]))
        if step in ((0, 2), (2, 0)):
            between = ((a[0] + b[0]) // 2, (a[1] + b[1]) // 2)
            occupancy[between] = not is_open
    return occupancy


def transfer(
    source: MazeSpec,
    source_d_lat: np.ndarray,
    target: MazeSpec,
    target_d_lat: np.ndarray,
) -> Reconstruction:
    """Calibrate on one maze, reconstruct another.

    The question worth asking. Reconstructing the maze a threshold was tuned
    on is close to circular; carrying the threshold across is the test of
    whether latent distance means the same thing in two different mazes.
    """
    return reconstruct(target, target_d_lat, calibrate(source, source_d_lat))
