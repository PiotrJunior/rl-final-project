"""Sampling tests: the grid that the dense latent map is built on, and the
teleport that keeps `phi`'s inputs physically plausible."""

import numpy as np
import pytest

from latentmine import sampling
from latentmine.mazes import geometry as geo
from latentmine.mazes import layouts as L


class TestGoalGrid:
    def test_one_sample_per_free_cell_by_default(self):
        spec = L.get("two_rooms")
        xy, index = sampling.goal_grid(spec)
        assert len(xy) == len(spec.free_cells())
        np.testing.assert_array_equal(index, np.arange(len(spec.free_cells())))

    def test_samples_land_on_cell_centres(self):
        spec = L.get("two_rooms")
        xy, _ = sampling.goal_grid(spec)
        np.testing.assert_allclose(xy, geo.cells_to_world(geo.free_cell_array(spec), spec.scaling))

    def test_subdivision_multiplies_the_sample_count(self):
        spec = L.get("loop")
        xy, index = sampling.goal_grid(spec, subdivisions=3)
        assert len(xy) == 9 * len(spec.free_cells())
        assert index.max() == len(spec.free_cells()) - 1

    def test_subdivided_samples_stay_inside_their_cell(self):
        spec = L.get("loop")
        xy, index = sampling.goal_grid(spec, subdivisions=4)
        cells = geo.free_cell_array(spec)
        offset = xy / spec.scaling - cells[index]
        assert np.abs(offset).max() < 0.5, "a sample escaped its own cell"

    def test_every_sample_maps_back_to_a_free_cell(self):
        spec = L.get("four_rooms")
        xy, index = sampling.goal_grid(spec, subdivisions=3)
        cells = geo.free_cell_array(spec)
        for point, k in zip(xy, index, strict=True):
            assert spec.is_free(*cells[k])
            assert geo.world_to_cell(tuple(point), spec.scaling) == tuple(cells[k])

    def test_jitter_is_reproducible_and_bounded(self):
        spec = L.get("spiral")
        a, _ = sampling.goal_grid(spec, subdivisions=2, jitter=0.1, seed=7)
        b, _ = sampling.goal_grid(spec, subdivisions=2, jitter=0.1, seed=7)
        c, _ = sampling.goal_grid(spec, subdivisions=2, jitter=0.1, seed=8)
        np.testing.assert_allclose(a, b)
        assert not np.allclose(a, c)

    def test_subdivisions_must_be_positive(self):
        with pytest.raises(ValueError, match="subdivisions"):
            sampling.goal_grid(L.get("loop"), subdivisions=0)


class TestTeleport:
    def test_only_the_position_dimensions_change(self):
        # Indices 0 and 1 are x, y; everything after is pose and velocity, and
        # keeping it is what holds an Ant state on the physics manifold.
        base = np.arange(31, dtype=np.float32)
        moved = sampling.teleport_states(base, np.array([[10.0, 20.0], [30.0, 40.0]]))
        assert moved.shape == (2, 31)
        np.testing.assert_allclose(moved[:, 0], [10.0, 30.0])
        np.testing.assert_allclose(moved[:, 1], [20.0, 40.0])
        np.testing.assert_allclose(moved[0, 2:], base[2:])
        np.testing.assert_allclose(moved[1, 2:], base[2:])

    def test_the_goal_can_be_set_independently(self):
        base = np.arange(6, dtype=np.float32)
        moved = sampling.teleport_states(base, np.array([[1.0, 2.0]]), goal_xy=np.array([[7.0, 8.0]]))
        np.testing.assert_allclose(moved[0, -2:], [7.0, 8.0])

    def test_per_target_poses_are_supported(self):
        base = np.tile(np.arange(6, dtype=np.float32), (3, 1))
        base[:, 3] = [1.0, 2.0, 3.0]
        moved = sampling.teleport_states(base, np.zeros((3, 2)))
        np.testing.assert_allclose(moved[:, 3], [1.0, 2.0, 3.0])

    def test_mismatched_lengths_are_rejected(self):
        with pytest.raises(ValueError, match="rows"):
            sampling.teleport_states(np.zeros((3, 6)), np.zeros((5, 2)))

    def test_the_input_is_not_mutated(self):
        base = np.arange(6, dtype=np.float32)
        original = base.copy()
        sampling.teleport_states(base, np.array([[9.0, 9.0]]))
        np.testing.assert_allclose(base, original)


class TestHelpers:
    def test_split_state_and_goal(self):
        obs = np.arange(31, dtype=np.float32)[None, :]
        state, goal = sampling.split_state_and_goal(obs, state_dim=29)
        assert state.shape == (1, 29) and goal.shape == (1, 2)
        np.testing.assert_allclose(goal[0], [29.0, 30.0])

    def test_pose_bank_samples_without_replacement(self):
        obs = np.arange(40, dtype=np.float32).reshape(10, 4)
        bank = sampling.pose_bank(obs, n=5, seed=0)
        assert bank.shape == (5, 4)
        assert len({tuple(row) for row in bank}) == 5

    def test_pose_bank_caps_at_the_available_rows(self):
        obs = np.zeros((3, 4))
        assert len(sampling.pose_bank(obs, n=100)) == 3

    def test_cell_centres_match_the_canonical_order(self):
        spec = L.get("two_rooms")
        np.testing.assert_allclose(
            sampling.cell_centres(spec),
            geo.cells_to_world(geo.free_cell_array(spec), spec.scaling),
        )
