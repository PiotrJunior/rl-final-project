"""Find a maze's bottlenecks from its latent space alone.

Two detectors that fail independently (LLD section 9), plus a null, scored
against betweenness centrality on the true maze graph.

The design called for three detectors. Two survived contact with the data:
`spectral` is excellent where a single cut is the right model (F1 1.00 on
`two_rooms` and `spiral`) and useless where it is not (0.00 on `four_rooms`),
while `betweenness` on the latent kNN graph is the best all-rounder (0.86 on
`four_rooms`). The third idea failed in two different forms - see
`latent_centrality`, which is retained as the null that shows the other two
find constrictions rather than merely the middle of the maze.

**Ground truth is betweenness, not articulation points.** Articulation points
look like the obvious choice and are exact, but `four_rooms` has none at all -
four doorways mean there is always a second route - and `spiral` has 47 of its
49 cells, since every interior cell of a corridor is a cut vertex. Betweenness
degrades gracefully in both.

Detection is scored with a **one-cell tolerance**: in `four_rooms` all four
doorways score an identical 0.246 but their flanking cells score 0.251 and
outrank them, because a flanking cell carries the doorway's traffic plus
intra-room traffic. Pointing at a cell adjacent to a doorway is not a failure.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..mazes import geometry as geo
from ..mazes.layouts import MazeSpec


@dataclass
class Detection:
    """What one detector found, and how well."""

    method: str
    scores: np.ndarray  # per free cell, higher = more bottleneck-like
    predicted: tuple[tuple[int, int], ...]
    precision: float = float("nan")
    recall: float = float("nan")
    f1: float = float("nan")


def _knn_graph(d_lat: np.ndarray, k: int) -> np.ndarray:
    """Symmetric kNN adjacency from a latent distance matrix."""
    n = len(d_lat)
    k = min(k, n - 1)
    adjacency = np.zeros((n, n))
    order = np.argsort(d_lat, axis=1)
    for i in range(n):
        for j in order[i, 1 : k + 1]:
            adjacency[i, j] = adjacency[j, i] = 1.0
    return adjacency


def spectral(spec: MazeSpec, d_lat: np.ndarray, k: int = 6, top: int = 3) -> Detection:
    """Fiedler vector of the latent kNN graph; cells adjacent to a sign change.

    The natural fit for `two_rooms`, which really is one cut. It is the wrong
    model for `four_rooms`, where a single bisection cannot describe four
    doorways - which is why it is one of three detectors rather than the only
    one.
    """
    adjacency = _knn_graph(d_lat, k)
    degree = adjacency.sum(1)
    with np.errstate(divide="ignore"):
        inverse_sqrt = np.where(degree > 0, 1.0 / np.sqrt(degree), 0.0)
    laplacian = np.eye(len(adjacency)) - inverse_sqrt[:, None] * adjacency * inverse_sqrt[None, :]
    values, vectors = np.linalg.eigh(laplacian)
    fiedler = vectors[:, np.argsort(values)[1]]

    # A cell scores highly when its latent neighbours disagree about the side
    # of the cut it is on.
    scores = np.array(
        [
            float(np.abs(np.sign(fiedler[adjacency[i] > 0]) - np.sign(fiedler[i])).mean())
            if (adjacency[i] > 0).any()
            else 0.0
            for i in range(len(adjacency))
        ]
    )
    return Detection("spectral", scores, _top_cells(spec, scores, top))


def betweenness(spec: MazeSpec, d_lat: np.ndarray, k: int = 6, top: int = 3) -> Detection:
    """Betweenness centrality on the latent kNN graph.

    Handles several doorways at once, where a single spectral cut cannot.
    """
    adjacency = _knn_graph(d_lat, k)
    scores = _graph_betweenness(adjacency)
    return Detection("betweenness", scores, _top_cells(spec, scores, top))


def latent_centrality(spec: MazeSpec, d_lat: np.ndarray, top: int = 3, tol: float = 0.05) -> Detection:
    """Fraction of cell pairs whose latent geodesic passes through each cell.

    **This is the null detector, not a third bottleneck detector.** A cell `c`
    is counted when `d(u, c) + d(c, v) ~= d(u, v)`, using only the distance
    matrix - no graph, no `k`. That sounds like betweenness, but on a metric
    embedding it is dominated by simple centrality: a cell in the middle of a
    large open room lies between most pairs, and a one-cell doorway does not.
    Measured F1 on a geodesic-embedded latent bears that out - 1.00 on
    `open_room`, where the "bottleneck" really is just the centre, and 0.00 on
    `two_rooms`, where it is a doorway.

    It is kept for exactly that reason. Reported beside `spectral` and
    `betweenness`, it shows that those two are picking out constrictions
    rather than merely finding the middle of the maze, which is the obvious
    alternative explanation for a detector that looks like it works.

    An earlier third detector - "latent stretch", the finite-difference
    magnitude of `d psi / d(x, y)`, on the theory that a doorway compresses
    space - was tried and dropped. It does not survive contact with the thing
    it is meant to detect: in a latent space that is an isometric embedding of
    the geodesic metric, adjacent cells are uniformly spaced everywhere, so
    the gradient is flat and it scored 0.00 on every maze in the set.
    """
    d = np.asarray(d_lat, dtype=float)
    n = len(d)
    scores = np.zeros(n)
    for c in range(n):
        through = d[:, c][:, None] + d[c, :][None, :]
        on_path = through <= d * (1.0 + tol)
        np.fill_diagonal(on_path, False)
        on_path[c, :] = False
        on_path[:, c] = False
        scores[c] = float(on_path.mean())
    return Detection("latent_centrality", scores, _top_cells(spec, scores, top))


def _top_cells(spec: MazeSpec, scores: np.ndarray, top: int) -> tuple[tuple[int, int], ...]:
    cells = list(spec.free_cells())
    return tuple(cells[i] for i in np.argsort(scores)[::-1][:top])


def _graph_betweenness(adjacency: np.ndarray) -> np.ndarray:
    """Brandes on an unweighted graph."""
    from collections import deque

    n = len(adjacency)
    score = np.zeros(n)
    neighbours = [np.flatnonzero(adjacency[i]) for i in range(n)]
    for source in range(n):
        stack, predecessors = [], [[] for _ in range(n)]
        sigma = np.zeros(n)
        sigma[source] = 1
        distance = np.full(n, -1)
        distance[source] = 0
        queue = deque([source])
        while queue:
            node = queue.popleft()
            stack.append(node)
            for other in neighbours[node]:
                if distance[other] < 0:
                    distance[other] = distance[node] + 1
                    queue.append(other)
                if distance[other] == distance[node] + 1:
                    sigma[other] += sigma[node]
                    predecessors[other].append(node)
        delta = np.zeros(n)
        while stack:
            node = stack.pop()
            for p in predecessors[node]:
                delta[p] += (sigma[p] / sigma[node]) * (1 + delta[node])
            if node != source:
                score[node] += delta[node]
    return score


def ground_truth(spec: MazeSpec, top: int = 3) -> tuple[tuple[int, int], ...]:
    """The cells a detector should find: highest true betweenness."""
    return geo.top_bottleneck_cells(spec, top)


def score_detection(spec: MazeSpec, detection: Detection, top: int = 3, tolerance: int = 1) -> Detection:
    """Precision, recall and F1 against the true bottlenecks.

    `tolerance` counts a prediction as a hit if it lies within that many cells
    of a true bottleneck. One is the right default and not a fudge: a doorway
    and its flanking cells are not distinguishable by betweenness.
    """
    truth = ground_truth(spec, top)
    if not truth:
        return detection

    def near(cell, target):
        return abs(cell[0] - target[0]) + abs(cell[1] - target[1]) <= tolerance

    predicted = detection.predicted[:top]
    hits = sum(any(near(p, t) for t in truth) for p in predicted)
    found = sum(any(near(p, t) for p in predicted) for t in truth)
    precision = hits / len(predicted) if predicted else float("nan")
    recall = found / len(truth)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    return Detection(detection.method, detection.scores, detection.predicted, precision, recall, f1)


def contrast(scores: np.ndarray) -> float:
    """Peak-to-mean of a score field.

    The false-positive test: a detector that "finds" bottlenecks in
    `open_room` is measuring its own hyperparameters. Contrast on the control
    is what the walled mazes are read against.
    """
    mean = float(np.mean(scores))
    return float(np.max(scores) / mean) if mean > 0 else float("nan")
