"""Grid/world geometry and graph distances for maze specs.

**This module owns the coordinate convention and nothing else may reimplement
it.** Upstream `make_maze` places every wall geom, start and goal at world
position `(i * scaling, j * scaling)` for grid cell `(i, j)`. So:

    grid row    i  ->  world x
    grid column j  ->  world y

A plot with x horizontal and y vertical therefore shows the ASCII art
transposed. That is correct and intended; getting it backwards silently
inverts every figure in the project, which is why `cell_to_world` is the only
place the mapping is written down and why a test asserts it directly.

Cell centres sit exactly on `(i * scaling, j * scaling)` - there is no
half-cell offset, because upstream does not apply one.

Like `layouts`, this module imports nothing from JAX, brax or mujoco.
"""

from __future__ import annotations

import heapq
import math
from collections import deque

import numpy as np

from .layouts import MazeSpec

# 4-connected steps, then the four diagonals.
_ORTHOGONAL = ((-1, 0), (1, 0), (0, -1), (0, 1))
_DIAGONAL = ((-1, -1), (-1, 1), (1, -1), (1, 1))


# ---------------------------------------------------------------------------
# grid <-> world
# ---------------------------------------------------------------------------


def cell_to_world(cell: tuple[int, int], scaling: float) -> tuple[float, float]:
    """Grid cell -> world `(x, y)`. Row indexes x, column indexes y."""
    i, j = cell
    return (i * scaling, j * scaling)


def cells_to_world(cells, scaling: float) -> np.ndarray:
    """Vectorised `cell_to_world`. Returns `(N, 2)` of world `(x, y)`."""
    arr = np.asarray(cells, dtype=float).reshape(-1, 2)
    return arr * scaling


def world_to_cell(xy: tuple[float, float], scaling: float) -> tuple[int, int]:
    """World `(x, y)` -> nearest grid cell. Inverse of `cell_to_world`."""
    x, y = xy
    return (int(round(x / scaling)), int(round(y / scaling)))


def worlds_to_cells(xy: np.ndarray, scaling: float) -> np.ndarray:
    """Vectorised `world_to_cell`. Takes `(N, 2)`, returns integer `(N, 2)`."""
    return np.rint(np.asarray(xy, dtype=float).reshape(-1, 2) / scaling).astype(int)


def world_extent(spec: MazeSpec) -> tuple[float, float, float, float]:
    """`(x_min, x_max, y_min, y_max)` spanning all cell centres."""
    return (
        0.0,
        (spec.n_rows - 1) * spec.scaling,
        0.0,
        (spec.n_cols - 1) * spec.scaling,
    )


# ---------------------------------------------------------------------------
# occupancy and indexing
# ---------------------------------------------------------------------------


def occupancy(spec: MazeSpec) -> np.ndarray:
    """Boolean `(n_rows, n_cols)` array, True where the cell is a wall."""
    return np.array(
        [[spec.is_wall(i, j) for j in range(spec.n_cols)] for i in range(spec.n_rows)],
        dtype=bool,
    )


def free_cell_array(spec: MazeSpec) -> np.ndarray:
    """Free cells as an integer `(M, 2)` array in `spec.free_cells()` order.

    That order is canonical: every distance matrix, embedding array and metric
    in the project is indexed by it.
    """
    return np.array(spec.free_cells(), dtype=int).reshape(-1, 2)


def free_cell_index(spec: MazeSpec) -> dict[tuple[int, int], int]:
    """Map each free cell to its row in `free_cell_array`."""
    return {cell: k for k, cell in enumerate(spec.free_cells())}


# ---------------------------------------------------------------------------
# graph structure
# ---------------------------------------------------------------------------


def neighbours(spec: MazeSpec, cell: tuple[int, int], connectivity: int = 8):
    """Traversable neighbours of a free cell.

    With `connectivity=8` a diagonal step is allowed only when both of the
    orthogonal cells it passes between are free, so the agent cannot squeeze
    through the corner where two walls meet. Without that check an 8-connected
    graph would tunnel diagonally through the wall junctions in `two_rooms`
    and `four_rooms`, quietly deflating exactly the geodesic gap we are trying
    to measure.
    """
    if connectivity not in (4, 8):
        raise ValueError(f"connectivity must be 4 or 8, got {connectivity}")
    i, j = cell
    out = []
    for di, dj in _ORTHOGONAL:
        if spec.is_free(i + di, j + dj):
            out.append(((i + di, j + dj), 1.0))
    if connectivity == 8:
        for di, dj in _DIAGONAL:
            if not spec.is_free(i + di, j + dj):
                continue
            if not (spec.is_free(i + di, j) and spec.is_free(i, j + dj)):
                continue  # corner cutting
            out.append(((i + di, j + dj), math.sqrt(2.0)))
    return out


def is_connected(spec: MazeSpec, connectivity: int = 4) -> bool:
    """Whether every free cell is reachable from every other.

    Defaults to 4-connectivity: a maze that is connected only via diagonal
    steps is not one a physical agent can actually traverse.
    """
    cells = spec.free_cells()
    if not cells:
        return False
    seen = {cells[0]}
    queue = deque(seen)
    while queue:
        for nb, _ in neighbours(spec, queue.popleft(), connectivity):
            if nb not in seen:
                seen.add(nb)
                queue.append(nb)
    return len(seen) == len(cells)


def articulation_points(spec: MazeSpec, connectivity: int = 4) -> tuple[tuple[int, int], ...]:
    """Free cells whose removal disconnects the maze - the ground truth for
    bottleneck detection (LLD section 9).

    Iterative Hopcroft-Tarjan; recursion would be fine at these sizes but an
    explicit stack keeps it safe if the maze set ever grows.
    """
    cells = spec.free_cells()
    if not cells:
        return ()
    order: dict[tuple[int, int], int] = {}
    low: dict[tuple[int, int], int] = {}
    parent: dict[tuple[int, int], tuple[int, int] | None] = {}
    found: set[tuple[int, int]] = set()
    counter = 0

    for root in cells:
        if root in order:
            continue
        parent[root] = None
        stack = [(root, iter(neighbours(spec, root, connectivity)))]
        order[root] = low[root] = counter
        counter += 1
        root_children = 0
        while stack:
            node, it = stack[-1]
            advanced = False
            for nb, _ in it:
                if nb not in order:
                    parent[nb] = node
                    order[nb] = low[nb] = counter
                    counter += 1
                    if node == root:
                        root_children += 1
                    stack.append((nb, iter(neighbours(spec, nb, connectivity))))
                    advanced = True
                    break
                if nb != parent[node]:
                    low[node] = min(low[node], order[nb])
            if not advanced:
                stack.pop()
                if stack:
                    up = stack[-1][0]
                    low[up] = min(low[up], low[node])
                    if up != root and low[node] >= order[up]:
                        found.add(up)
        if root_children > 1:
            found.add(root)
    return tuple(sorted(found))


# ---------------------------------------------------------------------------
# distances
# ---------------------------------------------------------------------------


def geodesic_from(
    spec: MazeSpec,
    source: tuple[int, int],
    connectivity: int = 8,
    in_world_units: bool = True,
) -> np.ndarray:
    """Shortest-path distance from `source` to every free cell.

    Returns a `(M,)` array indexed by `free_cell_array` order; unreachable
    cells are `inf`. Dijkstra rather than BFS because 8-connectivity gives
    diagonal steps a cost of sqrt(2).

    On `in_world_units`: distances are multiplied by `spec.scaling` so they
    are directly comparable with Euclidean world distance and with latent
    distance. Rank-based metrics do not care, but ratios and plots do.

    On `connectivity`: 8 is the default because a 4-connected graph measures
    Manhattan rather than Euclidean path length, which would make geodesic and
    Euclidean distance disagree by up to sqrt(2) even in `open_room` - and
    `open_room` is precisely the control where they are supposed to agree. The
    octile metric still carries a residual anisotropy of about 8%; it is the
    same for every maze, so it cancels in the comparisons that matter, but it
    is a floor on how well any encoder can appear to match `d_geo`.
    """
    if not spec.is_free(*source):
        raise ValueError(f"{spec.name}: source {source} is not a free cell")
    index = free_cell_index(spec)
    dist = np.full(len(index), np.inf)
    dist[index[source]] = 0.0
    heap = [(0.0, source)]
    settled = set()
    while heap:
        d, cell = heapq.heappop(heap)
        if cell in settled:
            continue
        settled.add(cell)
        for nb, w in neighbours(spec, cell, connectivity):
            nd = d + w
            k = index[nb]
            if nd < dist[k]:
                dist[k] = nd
                heapq.heappush(heap, (nd, nb))
    return dist * spec.scaling if in_world_units else dist


def geodesic_matrix(spec: MazeSpec, connectivity: int = 8, in_world_units: bool = True) -> np.ndarray:
    """All-pairs geodesic distance, `(M, M)` in `free_cell_array` order."""
    cells = spec.free_cells()
    return np.stack([geodesic_from(spec, c, connectivity, in_world_units) for c in cells], axis=0)


def euclidean_matrix(spec: MazeSpec) -> np.ndarray:
    """All-pairs straight-line world distance, ignoring walls. `(M, M)`."""
    world = cells_to_world(free_cell_array(spec), spec.scaling)
    diff = world[:, None, :] - world[None, :, :]
    return np.sqrt((diff**2).sum(-1))


def detour_ratio(spec: MazeSpec, connectivity: int = 8) -> np.ndarray:
    """`d_geo / d_euc` per pair, with the diagonal set to 1.

    How much longer the walk is than the straight line. Large entries are the
    pairs that separate a wall-aware encoder from a position-only one, and are
    what the wall-crossing metric (LLD section 6B) selects on.
    """
    geo = geodesic_matrix(spec, connectivity)
    euc = euclidean_matrix(spec)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(euc > 0, geo / euc, 1.0)
    return ratio


def betweenness_centrality(spec: MazeSpec, connectivity: int = 8, normalised: bool = True) -> np.ndarray:
    """Fraction of all-pairs shortest paths passing through each free cell.

    Brandes' algorithm on the weighted maze graph. Returns `(M,)` in
    `free_cell_array` order.

    This, not `articulation_points`, is the ground truth for bottleneck
    detection. Articulation points look like the obvious choice and are exact,
    but they are the wrong measure for most of this maze set:

    - `four_rooms` has **no** articulation points at all. Four doorways mean
      there is always a second route, so removing any single cell leaves the
      maze connected - yet the four doorways are plainly bottlenecks.
    - `spiral` has 47 of them out of 49 free cells, since every interior cell
      of a corridor is a cut vertex. That labels almost the whole maze a
      bottleneck, which is no more useful than labelling none of it.

    Betweenness degrades gracefully in both cases: it peaks sharply at the
    doorways of `two_rooms` and `four_rooms`, is smooth along `spiral`, and is
    flat in `open_room` - which is what makes `open_room` a usable
    false-positive test.
    """
    cells = spec.free_cells()
    index = free_cell_index(spec)
    score = np.zeros(len(cells))

    for source in cells:
        # Single-source shortest paths, accumulating predecessors and counts.
        sigma = np.zeros(len(cells))
        sigma[index[source]] = 1.0
        dist = np.full(len(cells), np.inf)
        dist[index[source]] = 0.0
        preds: list[list[int]] = [[] for _ in cells]
        order: list[int] = []
        settled = set()
        heap = [(0.0, source)]
        while heap:
            d, cell = heapq.heappop(heap)
            if cell in settled:
                continue
            settled.add(cell)
            order.append(index[cell])
            for nb, w in neighbours(spec, cell, connectivity):
                k, nd = index[nb], d + w
                if nd < dist[k] - 1e-12:
                    dist[k] = nd
                    sigma[k] = sigma[index[cell]]
                    preds[k] = [index[cell]]
                    heapq.heappush(heap, (nd, nb))
                elif abs(nd - dist[k]) <= 1e-12 and nb not in settled:
                    sigma[k] += sigma[index[cell]]
                    preds[k].append(index[cell])

        # Accumulate dependencies back-to-front.
        delta = np.zeros(len(cells))
        for k in reversed(order):
            for p in preds[k]:
                if sigma[k] > 0:
                    delta[p] += (sigma[p] / sigma[k]) * (1.0 + delta[k])
            if k != index[source]:
                score[k] += delta[k]

    if normalised:
        n = len(cells)
        if n > 2:
            score /= (n - 1) * (n - 2)
    return score


def top_bottleneck_cells(spec: MazeSpec, k: int, connectivity: int = 8) -> tuple[tuple[int, int], ...]:
    """The `k` free cells with highest betweenness - ground truth for scoring
    the latent-space bottleneck detectors of LLD section 9."""
    score = betweenness_centrality(spec, connectivity)
    cells = spec.free_cells()
    return tuple(cells[i] for i in np.argsort(score)[::-1][:k])
