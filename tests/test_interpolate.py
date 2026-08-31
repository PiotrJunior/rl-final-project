"""Interpolation and path-scoring tests.

The scoring functions are checked against paths whose answers are obvious - a
straight line through a spiral is mostly inside walls, the BFS oracle is not -
so that a real decoded path can be read against a calibrated scale.
"""

import numpy as np
import pytest

from latentmine.decoder import interpolate as I
from latentmine.mazes import layouts as L


class TestLatentInterpolation:
    def test_linear_under_a_euclidean_energy(self):
        z0, z1 = np.zeros(4), np.ones(4)
        points = I.interpolate_latents(z0, z1, 5, "norm")
        assert points.shape == (5, 4)
        np.testing.assert_allclose(points[0], z0)
        np.testing.assert_allclose(points[-1], z1)
        np.testing.assert_allclose(points[2], 0.5 * np.ones(4))

    def test_slerp_under_a_spherical_energy(self):
        # The bug being pre-empted: linear interpolation under a cosine energy
        # cuts through the inside of the sphere.
        z0 = np.array([1.0, 0.0])
        z1 = np.array([0.0, 1.0])
        points = I.interpolate_latents(z0, z1, 5, "cosine")
        norms = np.linalg.norm(points, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-9)
        np.testing.assert_allclose(points[2], [np.sqrt(0.5), np.sqrt(0.5)], atol=1e-9)

    def test_linear_would_not_stay_on_the_sphere(self):
        z0, z1 = np.array([1.0, 0.0]), np.array([0.0, 1.0])
        linear = I.interpolate_latents(z0, z1, 5, "norm")
        assert np.linalg.norm(linear[2]) < 0.95

    def test_identical_endpoints_are_handled(self):
        z = np.array([1.0, 2.0, 3.0])
        points = I.interpolate_latents(z, z, 4, "cosine")
        assert np.isfinite(points).all()


class TestLatentGraphPath:
    def test_it_follows_the_manifold_rather_than_the_chord(self):
        # Points on a semicircle: the graph path visits intermediate points,
        # while a straight line between the endpoints would leave the curve.
        angles = np.linspace(0, np.pi, 12)
        latents = np.stack([np.cos(angles), np.sin(angles)], axis=1)
        path = I.latent_graph_path(latents, 0, 11, "norm", k=2)
        assert path[0] == 0 and path[-1] == 11
        assert len(path) > 2, "a chord would be two nodes"
        assert list(path) == sorted(path), "should walk along the curve"

    def test_a_disconnected_graph_degrades_gracefully(self):
        latents = np.array([[0.0, 0.0], [0.1, 0.0], [100.0, 0.0], [100.1, 0.0]])
        path = I.latent_graph_path(latents, 0, 3, "norm", k=1)
        assert path[0] == 0 and path[-1] == 3


class TestPathScoring:
    def test_a_straight_line_through_a_spiral_is_mostly_illegal(self):
        spec = L.get("spiral")
        start, goal = (1, 1), (5, 5)
        scored = I.score_path(spec, I.straight_line(spec, start, goal, 40), start, goal, "straight")
        assert scored.valid_fraction < 0.7
        # And far shorter than the real route - it ignores the corridor.
        assert scored.length_ratio < 0.3

    def test_the_oracle_route_is_legal_and_monotone(self):
        for maze, start, goal in (("spiral", (1, 1), (5, 5)), ("two_rooms", (1, 1), (7, 9))):
            spec = L.get(maze)
            scored = I.score_path(spec, I.geodesic_waypoints(spec, start, goal), start, goal, "geodesic")
            assert scored.valid_fraction == 1.0
            assert scored.monotonicity == pytest.approx(1.0)
            assert scored.length_ratio == pytest.approx(1.0, abs=0.15)

    def test_a_wall_crossing_line_scores_below_one(self):
        spec = L.get("two_rooms")
        start, goal = (1, 1), (1, 9)  # straight across the dividing wall
        scored = I.score_path(spec, I.straight_line(spec, start, goal, 25), start, goal, "straight")
        assert scored.valid_fraction < 1.0

    def test_scoring_survives_points_outside_the_grid(self):
        spec = L.get("two_rooms")
        points = np.array([[4.0, 4.0], [-500.0, -500.0], [28.0, 36.0]])
        scored = I.score_path(spec, points, (1, 1), (7, 9), "junk")
        assert 0.0 <= scored.valid_fraction < 1.0
        assert np.isfinite(scored.length)

    def test_geodesic_waypoints_start_and_end_where_asked(self):
        spec = L.get("four_rooms")
        start, goal = (1, 1), (9, 9)
        points = I.geodesic_waypoints(spec, start, goal)
        from latentmine.mazes import geometry as geo

        np.testing.assert_allclose(points[0], geo.cell_to_world(start, spec.scaling))
        np.testing.assert_allclose(points[-1], geo.cell_to_world(goal, spec.scaling))
