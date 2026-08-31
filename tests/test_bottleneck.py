"""Bottleneck-detection and maze-reconstruction tests.

Exercised on a latent that is an exact geodesic embedding of each maze - the
best case CRL could produce - so a detector that fails here would never work on
a real checkpoint.
"""

import numpy as np
import pytest

from latentmine.analysis import bottleneck as B
from latentmine.analysis import reconstruct as R
from latentmine.mazes import geometry as geo
from latentmine.mazes import layouts as L


def geodesic_latent(spec, dims=8):
    d = geo.geodesic_matrix(spec)
    n = len(d)
    centering = np.eye(n) - np.ones((n, n)) / n
    gram = -0.5 * centering @ (d**2) @ centering
    values, vectors = np.linalg.eigh(gram)
    top = np.argsort(values)[::-1][:dims]
    return vectors[:, top] * np.sqrt(np.clip(values[top], 0, None))


def distances(latent):
    return np.sqrt(((latent[:, None, :] - latent[None, :, :]) ** 2).sum(-1))


class TestGroundTruth:
    def test_it_is_betweenness_not_articulation(self):
        # four_rooms has no articulation points at all, so an
        # articulation-based ground truth would be empty for exactly the maze
        # meant to make precision measurable.
        spec = L.get("four_rooms")
        assert geo.articulation_points(spec) == ()
        assert len(B.ground_truth(spec, 4)) == 4

    def test_true_bottlenecks_sit_at_or_beside_the_doorways(self):
        spec = L.get("four_rooms")
        doors = {(2, 5), (8, 5), (5, 2), (5, 8)}
        for cell in B.ground_truth(spec, 4):
            assert any(abs(cell[0] - d[0]) + abs(cell[1] - d[1]) <= 1 for d in doors)


class TestDetectors:
    def test_spectral_finds_a_single_cut(self):
        spec = L.get("two_rooms")
        detection = B.score_detection(spec, B.spectral(spec, distances(geodesic_latent(spec))))
        assert detection.f1 > 0.9

    def test_graph_betweenness_handles_four_doorways(self):
        # Where a single spectral cut is the wrong model.
        spec = L.get("four_rooms")
        latent = geodesic_latent(spec)
        graph = B.score_detection(spec, B.betweenness(spec, distances(latent), top=4), top=4)
        spectral = B.score_detection(spec, B.spectral(spec, distances(latent), top=4), top=4)
        assert graph.f1 > 0.8
        assert graph.f1 > spectral.f1

    def test_the_null_detector_finds_the_centre_not_the_doorway(self):
        """latent_centrality is retained precisely because it fails this way:
        it scores well on open_room, where the answer is just the middle, and
        badly on two_rooms, where it is a doorway. That contrast is what shows
        the other detectors are not merely finding the centre."""
        two_rooms = L.get("two_rooms")
        open_room = L.get("open_room")
        walled = B.score_detection(
            two_rooms, B.latent_centrality(two_rooms, distances(geodesic_latent(two_rooms)))
        )
        empty = B.score_detection(
            open_room, B.latent_centrality(open_room, distances(geodesic_latent(open_room)))
        )
        assert empty.f1 > walled.f1

    def test_contrast_is_higher_in_a_walled_maze_than_the_control(self):
        # The false-positive test: a detector that "finds" bottlenecks in an
        # empty room is measuring its own hyperparameters.
        def peak(name):
            spec = L.get(name)
            return B.contrast(B.spectral(spec, distances(geodesic_latent(spec))).scores)

        assert peak("two_rooms") > peak("open_room")

    def test_scoring_uses_a_one_cell_tolerance(self):
        # A doorway and its flanking cells are indistinguishable by
        # betweenness, so exact-match scoring would understate every detector.
        spec = L.get("four_rooms")
        truth = B.ground_truth(spec, 4)
        neighbours = tuple((i + 1, j) for (i, j) in truth)
        near_miss = B.Detection("fake", np.zeros(len(spec.free_cells())), neighbours)
        assert B.score_detection(spec, near_miss, top=4).f1 > 0.9

    def test_a_random_detector_scores_poorly(self):
        spec = L.get("two_rooms")
        rng = np.random.default_rng(0)
        cells = list(spec.free_cells())
        guess = tuple(cells[i] for i in rng.choice(len(cells), 3, replace=False))
        scored = B.score_detection(spec, B.Detection("random", np.zeros(len(cells)), guess))
        assert scored.f1 < 0.7


class TestReconstruction:
    @pytest.mark.parametrize("maze", list(L.names()))
    def test_a_geodesic_latent_reconstructs_its_own_maze(self, maze):
        spec = L.get(maze)
        d = distances(geodesic_latent(spec))
        result = R.reconstruct(spec, d, R.calibrate(spec, d))
        assert result.edge_f1 > 0.95
        assert result.occupancy_iou > 0.95

    @pytest.mark.parametrize("maze", list(L.names()))
    def test_a_threshold_calibrated_elsewhere_transfers(self, maze):
        # The question actually worth asking - reconstructing the maze a
        # threshold was tuned on is close to circular.
        source = L.get("two_rooms")
        source_d = distances(geodesic_latent(source))
        target = L.get(maze)
        result = R.transfer(source, source_d, target, distances(geodesic_latent(target)))
        assert result.edge_f1 > 0.9, f"{maze} did not survive the transfer"

    def test_the_threshold_is_scale_free(self):
        # Measured, not assumed: adjacent cells sit ~5.4 apart in two_rooms'
        # latent and ~11.3 apart in loop's, so an absolute threshold collapses
        # on transfer.
        two_rooms, loop = L.get("two_rooms"), L.get("loop")
        assert R.latent_scale(distances(geodesic_latent(loop))) > R.latent_scale(
            distances(geodesic_latent(two_rooms))
        )
        ratio = R.calibrate(two_rooms, distances(geodesic_latent(two_rooms)))
        assert 0.1 < ratio < 10, "a scale-free threshold should be order 1"

    def test_a_random_latent_reconstructs_nothing(self):
        spec = L.get("four_rooms")
        rng = np.random.default_rng(1)
        d = distances(rng.normal(size=(len(spec.free_cells()), 8)))
        result = R.reconstruct(spec, d, R.calibrate(spec, d))
        assert result.edge_f1 < 0.95

    def test_candidate_edges_include_blocked_passages(self):
        # Scoring only on passages that exist would make "everything is open"
        # a perfect answer.
        spec = L.get("two_rooms")
        edges = R._candidate_edges(spec)
        blocked = [(a, b) for a, b in edges if abs(a[0] - b[0]) + abs(a[1] - b[1]) == 2]
        assert len(blocked) > 0

    def test_calibration_needs_adjacent_pairs(self):
        spec = L.get("two_rooms")
        # A well-formed maze always has some; this guards the error path.
        assert R.calibrate(spec, distances(geodesic_latent(spec))) > 0
