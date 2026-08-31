"""Latent interpolation, decoded into maze waypoints.

Deliverable 3.2. Four paths are produced for every start/goal pair, because
"the decoded path bends around the wall" only means something in contrast
(LLD section 8.3):

1. `straight` - a line in raw `(x, y)`, which cuts through walls by construction;
2. `geodesic` - the BFS oracle;
3. `latent_linear` - the thing under test;
4. `latent_graph` - shortest path through a kNN graph over the embedded goals.

(4) exists because of a specific predicted failure: even a latent space that
encodes the maze perfectly may not be *convex*, so the straight line between
two latents can leave the manifold of realisable goals and decode to nonsense.
If (3) fails and (4) succeeds, the finding is "the structure is there but the
space is not convex", which is far more interesting than "interpolation does
not work" - and it is only available if (4) is built.

Interpolation follows the energy function: linear for `norm`/`l2`, slerp for
`dot`/`cosine`. Using linear interpolation under a cosine energy is a real bug
being pre-empted, not a hypothetical.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..embed import is_euclidean, pairwise_latent_distance
from ..mazes import geometry as geo
from ..mazes.layouts import MazeSpec


@dataclass
class Path:
    """A candidate route through the maze."""

    name: str
    points: np.ndarray  # (n, 2) world xy
    valid_fraction: float = float("nan")
    length: float = float("nan")
    length_ratio: float = float("nan")
    monotonicity: float = float("nan")


def interpolate_latents(z0: np.ndarray, z1: np.ndarray, n: int, energy_fn: str) -> np.ndarray:
    """`n` points from `z0` to `z1`, in the geometry the critic was trained in."""
    t = np.linspace(0.0, 1.0, n)[:, None]
    z0 = np.asarray(z0, dtype=np.float64).ravel()
    z1 = np.asarray(z1, dtype=np.float64).ravel()
    if is_euclidean(energy_fn):
        return (1 - t) * z0 + t * z1
    return _slerp(z0, z1, t)


def _slerp(z0: np.ndarray, z1: np.ndarray, t: np.ndarray) -> np.ndarray:
    n0 = z0 / max(np.linalg.norm(z0), 1e-12)
    n1 = z1 / max(np.linalg.norm(z1), 1e-12)
    omega = np.arccos(np.clip(n0 @ n1, -1.0, 1.0))
    if omega < 1e-8:
        return np.repeat(z0[None, :], len(t), axis=0)
    sin_omega = np.sin(omega)
    return (np.sin((1 - t) * omega) / sin_omega) * n0 + (np.sin(t * omega) / sin_omega) * n1


def latent_graph_path(
    latents: np.ndarray,
    start_index: int,
    goal_index: int,
    energy_fn: str,
    k: int = 6,
) -> np.ndarray:
    """Indices along the shortest path through a kNN graph over `latents`.

    Stays on the manifold of embedded goals by construction, which is what
    makes it the useful fallback when the straight line does not.
    """
    import heapq

    d = pairwise_latent_distance(latents, latents, energy_fn)
    n = len(d)
    k = min(k, n - 1)
    neighbours = np.argsort(d, axis=1)[:, 1 : k + 1]

    dist = np.full(n, np.inf)
    previous = np.full(n, -1, dtype=int)
    dist[start_index] = 0.0
    heap = [(0.0, start_index)]
    seen = set()
    while heap:
        cost, node = heapq.heappop(heap)
        if node in seen:
            continue
        seen.add(node)
        if node == goal_index:
            break
        for other in neighbours[node]:
            step = cost + d[node, other]
            if step < dist[other]:
                dist[other] = step
                previous[other] = node
                heapq.heappush(heap, (step, int(other)))

    if not np.isfinite(dist[goal_index]):
        return np.array([start_index, goal_index])
    path = [goal_index]
    while path[-1] != start_index:
        path.append(int(previous[path[-1]]))
        if previous[path[-1]] == -1 and path[-1] != start_index:
            return np.array([start_index, goal_index])
    return np.array(path[::-1])


def score_path(spec: MazeSpec, points: np.ndarray, start_cell, goal_cell, name: str) -> Path:
    """Metric F (LLD section 6): is this actually a usable route?

    `valid_fraction` is the headline - the share of waypoints in free space.
    `length_ratio` compares the walked length with the geodesic, and
    `monotonicity` is the fraction of steps that make progress towards the
    goal in geodesic terms, which catches a path that wanders while technically
    staying legal.
    """
    points = np.asarray(points, dtype=float).reshape(-1, 2)
    cells = [geo.world_to_cell(tuple(p), spec.scaling) for p in points]

    def inside(cell):
        return 0 <= cell[0] < spec.n_rows and 0 <= cell[1] < spec.n_cols and spec.is_free(*cell)

    valid = np.array([inside(c) for c in cells])
    length = float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())
    geodesic_length = float(geo.geodesic_from(spec, start_cell)[list(spec.free_cells()).index(goal_cell)])

    to_goal = geo.geodesic_from(spec, goal_cell)
    index = {cell: k for k, cell in enumerate(spec.free_cells())}
    progress = [to_goal[index[c]] for c in cells if c in index]
    if len(progress) > 1:
        deltas = np.diff(progress)
        monotonicity = float((deltas <= 1e-9).mean())
    else:
        monotonicity = float("nan")

    return Path(
        name=name,
        points=points,
        valid_fraction=float(valid.mean()),
        length=length,
        length_ratio=length / geodesic_length if geodesic_length > 0 else float("nan"),
        monotonicity=monotonicity,
    )


def straight_line(spec: MazeSpec, start_cell, goal_cell, n: int) -> np.ndarray:
    a = np.array(geo.cell_to_world(start_cell, spec.scaling))
    b = np.array(geo.cell_to_world(goal_cell, spec.scaling))
    t = np.linspace(0, 1, n)[:, None]
    return (1 - t) * a + t * b


def geodesic_waypoints(spec: MazeSpec, start_cell, goal_cell) -> np.ndarray:
    """The oracle route: cells along a true shortest path."""
    to_goal = geo.geodesic_from(spec, goal_cell, in_world_units=False)
    index = {cell: k for k, cell in enumerate(spec.free_cells())}
    current = tuple(start_cell)
    route = [current]
    for _ in range(4 * len(index)):
        if current == tuple(goal_cell):
            break
        options = [nb for nb, _ in geo.neighbours(spec, current)]
        current = min(options, key=lambda c: to_goal[index[c]])
        route.append(current)
    return geo.cells_to_world(np.array(route), spec.scaling)
