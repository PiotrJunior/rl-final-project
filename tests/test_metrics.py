"""Metric tests.

A metric that cannot separate a known-good case from a known-bad one is not
evidence. Each metric here is exercised against three synthetic latents whose
answers we know in advance:

* **geodesic** - a classical-MDS embedding of the maze's own geodesic matrix,
  i.e. a latent that encodes exactly what we hope CRL learns;
* **position** - the Euclidean distance matrix, a latent that knows where it
  is but nothing about walls;
* **random** - noise.

The position-only case is the important one. It scores highly against the
geodesic distance simply because geodesic and Euclidean distance are strongly
correlated, so any metric that cannot tell it from the geodesic case is not
measuring what the project claims to measure.
"""

import numpy as np
import pytest

from latentmine.analysis import metrics as M
from latentmine.mazes import geometry as geo
from latentmine.mazes import layouts as L


def geodesic_latent(spec, dims: int = 8) -> np.ndarray:
    """Classical MDS of the maze's geodesic matrix."""
    d = geo.geodesic_matrix(spec)
    n = len(d)
    centering = np.eye(n) - np.ones((n, n)) / n
    gram = -0.5 * centering @ (d**2) @ centering
    values, vectors = np.linalg.eigh(gram)
    top = np.argsort(values)[::-1][:dims]
    return vectors[:, top] * np.sqrt(np.clip(values[top], 0, None))


def latent_distances(latent: np.ndarray) -> np.ndarray:
    diff = latent[:, None, :] - latent[None, :, :]
    return np.sqrt((diff**2).sum(-1))


@pytest.fixture(params=["two_rooms", "four_rooms", "spiral"])
def spec(request):
    return L.get(request.param)


@pytest.fixture
def d_geodesic(spec):
    return latent_distances(geodesic_latent(spec))


@pytest.fixture
def d_position(spec):
    return geo.euclidean_matrix(spec)


@pytest.fixture
def d_random(spec):
    rng = np.random.default_rng(0)
    return latent_distances(rng.normal(size=(len(spec.free_cells()), 8)))


class TestGeometryCorrelation:
    def test_a_geodesic_latent_scores_near_one(self, spec, d_geodesic):
        result = M.geometry_correlation(d_geodesic, geo.geodesic_matrix(spec), geo.euclidean_matrix(spec))
        assert result.rho_geodesic > 0.95

    def test_a_random_latent_scores_near_zero(self, spec, d_random):
        result = M.geometry_correlation(d_random, geo.geodesic_matrix(spec), geo.euclidean_matrix(spec))
        assert abs(result.rho_geodesic) < 0.3

    def test_partial_correlation_is_zero_for_a_position_only_latent(self, spec, d_position):
        """The invariant that holds in every maze: strip the Euclidean
        component and a latent which knows only position has nothing left."""
        result = M.geometry_correlation(d_position, geo.geodesic_matrix(spec), geo.euclidean_matrix(spec))
        assert abs(result.partial_rho_geodesic) < 0.05

    @pytest.mark.parametrize("maze", ["two_rooms", "four_rooms", "open_room"])
    def test_the_naive_statistic_is_fooled_where_the_detour_is_small(self, maze):
        """Why the partial statistic is needed at all. In room-shaped mazes
        geodesic and Euclidean distance are strongly correlated, so a
        position-only latent scores well on the naive number and only the
        partial one exposes it."""
        spec = L.get(maze)
        euclid = geo.euclidean_matrix(spec)
        result = M.geometry_correlation(euclid, geo.geodesic_matrix(spec), euclid)
        assert result.rho_geodesic > 0.6, "the naive statistic is fooled..."
        assert abs(result.partial_rho_geodesic) < 0.05, "...the partial one is not"

    def test_spiral_defeats_even_the_naive_statistic(self):
        """The spiral exists to decouple the two distances, and does so well
        enough that a position-only latent scores poorly outright - the
        property that makes it the most diagnostic maze in the set."""
        spec = L.get("spiral")
        euclid = geo.euclidean_matrix(spec)
        result = M.geometry_correlation(euclid, geo.geodesic_matrix(spec), euclid)
        assert result.rho_geodesic < 0.4

    def test_partial_correlation_keeps_a_geodesic_latent(self, spec, d_geodesic):
        result = M.geometry_correlation(d_geodesic, geo.geodesic_matrix(spec), geo.euclidean_matrix(spec))
        assert result.partial_rho_geodesic > 0.5

    def test_the_gap_orders_the_three_cases(self, spec, d_geodesic, d_position):
        args = (geo.geodesic_matrix(spec), geo.euclidean_matrix(spec))
        assert M.geometry_correlation(d_geodesic, *args).gap > M.geometry_correlation(d_position, *args).gap

    def test_pair_count_is_the_upper_triangle(self, spec, d_geodesic):
        n = len(spec.free_cells())
        result = M.geometry_correlation(d_geodesic, geo.geodesic_matrix(spec), geo.euclidean_matrix(spec))
        assert result.n_pairs == n * (n - 1) // 2


class TestWallCrossing:
    def test_a_geodesic_latent_separates_across_the_wall(self, spec, d_geodesic):
        assert M.wall_crossing_ratio(spec, d_geodesic).ratio > 2.0

    def test_a_position_only_latent_does_not(self, spec, d_position):
        assert M.wall_crossing_ratio(spec, d_position).ratio < 1.7

    def test_open_room_has_no_wall_facing_pairs(self):
        # Reported as NaN rather than a ratio of 1, which would read as a
        # measurement rather than an absence of one.
        spec = L.get("open_room")
        result = M.wall_crossing_ratio(spec, latent_distances(geodesic_latent(spec)))
        assert result.n_across == 0
        assert np.isnan(result.ratio)

    def test_the_window_admits_pairs_across_a_one_cell_wall(self):
        # Walls are one cell thick, so the nearest cross-wall pair is two
        # cells apart; too tight a window silently selects nothing.
        spec = L.get("two_rooms")
        assert M.wall_crossing_ratio(spec, latent_distances(geodesic_latent(spec))).n_across > 0


class TestRoomPurity:
    def test_a_geodesic_latent_beats_the_position_null(self):
        spec = L.get("two_rooms")
        result = M.room_purity(spec, latent_distances(geodesic_latent(spec)))
        assert result.lift > 0

    def test_a_random_latent_is_far_worse_than_the_null(self):
        spec = L.get("four_rooms")
        rng = np.random.default_rng(1)
        d = latent_distances(rng.normal(size=(len(spec.free_cells()), 8)))
        assert M.room_purity(spec, d).lift < -0.1

    def test_the_position_latent_lifts_by_exactly_zero(self):
        spec = L.get("two_rooms")
        result = M.room_purity(spec, geo.euclidean_matrix(spec))
        assert result.lift == pytest.approx(0.0, abs=1e-12)

    def test_doorway_cells_are_excluded(self):
        spec = L.get("two_rooms")
        result = M.room_purity(spec, geo.euclidean_matrix(spec))
        assert result.n_cells == len(spec.scorable_cells()) == 56

    def test_mazes_without_rooms_are_refused(self):
        spec = L.get("spiral")
        with pytest.raises(ValueError, match="room purity is undefined"):
            M.room_purity(spec, latent_distances(geodesic_latent(spec)))


class TestProjectionFaithfulness:
    def test_an_identical_embedding_is_perfectly_trustworthy(self):
        spec = L.get("two_rooms")
        d = geo.geodesic_matrix(spec)
        assert M.trustworthiness(d, d) == pytest.approx(1.0)
        assert M.continuity(d, d) == pytest.approx(1.0)

    def test_a_scrambled_embedding_is_not(self):
        spec = L.get("two_rooms")
        d = geo.geodesic_matrix(spec)
        rng = np.random.default_rng(2)
        order = rng.permutation(len(d))
        scrambled = d[np.ix_(order, order)]
        assert M.trustworthiness(d, scrambled) < 0.8


class TestReports:
    def test_evaluate_assembles_every_applicable_metric(self):
        spec = L.get("two_rooms")
        report = M.evaluate(spec, latent_distances(geodesic_latent(spec)), run_id="r", step=5)
        assert report.maze == "two_rooms" and report.step == 5
        assert report.purity is not None
        assert report.as_dict()["geometry"]["rho_geodesic"] > 0.9

    def test_purity_is_omitted_where_it_is_undefined(self):
        spec = L.get("spiral")
        assert M.evaluate(spec, latent_distances(geodesic_latent(spec))).purity is None

    def test_aggregate_reports_mean_and_spread_over_seeds(self):
        spec = L.get("two_rooms")
        reports = []
        for seed in range(3):
            rng = np.random.default_rng(seed)
            noisy = geodesic_latent(spec) + 0.01 * rng.normal(size=geodesic_latent(spec).shape)
            reports.append(M.evaluate(spec, latent_distances(noisy)))
        summary = M.aggregate(reports)
        assert summary["n_seeds"] == 3
        assert summary["rho_geodesic"]["mean"] > 0.9
        assert summary["rho_geodesic"]["std"] < 0.1

    def test_aggregate_of_nothing_is_empty(self):
        assert M.aggregate([]) == {}
