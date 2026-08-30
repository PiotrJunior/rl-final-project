"""Geometry tests.

The first class of test here exists for one reason: upstream maps grid row `i`
to world x and column `j` to world y, and getting that backwards silently
transposes every figure in the project without raising anything. These assert
the convention directly rather than through a round trip, which would pass
just as happily under a transposition.
"""

import math

import numpy as np
import pytest

from latentmine.mazes import geometry as geo
from latentmine.mazes import layouts as L
from latentmine.mazes.layouts import MazeSpec

# Hand-checkable maze. Free cells: the ring around the single interior wall
# at (2, 2).
#     #####
#     #S..#
#     #.#.#
#     #...#
#     #####
SMALL = MazeSpec(
    name="small",
    grid=("#####", "#S..#", "#.#.#", "#...#", "#####"),
    scaling=2.0,
)


class TestConvention:
    def test_row_indexes_x_and_column_indexes_y(self):
        assert geo.cell_to_world((1, 0), 4.0) == (4.0, 0.0)
        assert geo.cell_to_world((0, 1), 4.0) == (0.0, 4.0)

    def test_moving_a_row_changes_only_x(self):
        x0, y0 = geo.cell_to_world((1, 3), 4.0)
        x1, y1 = geo.cell_to_world((2, 3), 4.0)
        assert x1 - x0 == 4.0
        assert y1 == y0

    def test_extent_follows_grid_shape_not_its_transpose(self):
        # two_rooms is 9 rows x 11 columns, so world x spans 8*scaling and
        # world y spans 10*scaling. A transposition swaps these.
        spec = L.get("two_rooms")
        assert spec.shape == (9, 11)
        x_min, x_max, y_min, y_max = geo.world_extent(spec)
        assert (x_min, x_max) == (0.0, 8 * spec.scaling)
        assert (y_min, y_max) == (0.0, 10 * spec.scaling)

    def test_world_to_cell_inverts_cell_to_world(self):
        for cell in L.get("four_rooms").free_cells():
            assert geo.world_to_cell(geo.cell_to_world(cell, 4.0), 4.0) == cell

    def test_vectorised_matches_scalar(self):
        spec = L.get("spiral")
        cells = geo.free_cell_array(spec)
        world = geo.cells_to_world(cells, spec.scaling)
        expected = np.array([geo.cell_to_world(tuple(c), spec.scaling) for c in cells])
        np.testing.assert_allclose(world, expected)
        np.testing.assert_array_equal(geo.worlds_to_cells(world, spec.scaling), cells)


class TestOccupancyAndIndexing:
    def test_occupancy_matches_spec(self):
        occ = geo.occupancy(SMALL)
        assert occ.shape == (5, 5)
        assert occ[2, 2]  # the interior block
        assert not occ[1, 1]
        assert occ[0].all() and occ[-1].all()

    def test_free_cell_order_is_row_major_and_canonical(self):
        cells = geo.free_cell_array(SMALL)
        assert [tuple(c) for c in cells] == list(SMALL.free_cells())
        index = geo.free_cell_index(SMALL)
        assert index[(1, 1)] == 0


class TestDistances:
    def test_four_connected_distances_are_hand_checked(self):
        d = geo.geodesic_from(SMALL, (1, 1), connectivity=4, in_world_units=False)
        index = geo.free_cell_index(SMALL)
        expected = {
            (1, 1): 0,
            (1, 2): 1,
            (1, 3): 2,
            (2, 1): 1,
            (2, 3): 3,
            (3, 1): 2,
            (3, 2): 3,
            (3, 3): 4,
        }
        for cell, want in expected.items():
            assert d[index[cell]] == pytest.approx(want), cell

    def test_diagonals_may_not_cut_a_wall_corner(self):
        # (1, 2) -> (2, 3) is a diagonal step past the corner of the block at
        # (2, 2). Forbidding it makes 8-connectivity agree with 4 here; allowing
        # it would give 1 + sqrt(2) ~ 2.41 instead of 3.
        d8 = geo.geodesic_from(SMALL, (1, 1), connectivity=8, in_world_units=False)
        index = geo.free_cell_index(SMALL)
        assert d8[index[(2, 3)]] == pytest.approx(3.0)
        assert d8[index[(3, 2)]] == pytest.approx(3.0)

    def test_diagonals_are_used_when_the_corner_is_clear(self):
        d = geo.geodesic_from(L.get("open_room"), (1, 1), connectivity=8, in_world_units=False)
        index = geo.free_cell_index(L.get("open_room"))
        assert d[index[(3, 3)]] == pytest.approx(2 * math.sqrt(2))

    def test_world_units_scale_the_result(self):
        cells = geo.geodesic_from(SMALL, (1, 1), in_world_units=False)
        world = geo.geodesic_from(SMALL, (1, 1), in_world_units=True)
        np.testing.assert_allclose(world, cells * SMALL.scaling)

    def test_matrix_is_symmetric_finite_and_zero_on_the_diagonal(self):
        m = geo.geodesic_matrix(L.get("two_rooms"))
        np.testing.assert_allclose(m, m.T, atol=1e-9)
        np.testing.assert_allclose(np.diag(m), 0.0)
        assert np.isfinite(m).all()

    def test_open_room_geodesic_tracks_euclidean(self):
        # The control: with no interior wall the two must agree up to the
        # octile anisotropy, which is bounded by ~8%.
        ratio = geo.detour_ratio(L.get("open_room"))
        off = ~np.eye(ratio.shape[0], dtype=bool)
        assert ratio[off].max() < 1.09

    def test_walls_force_a_large_detour(self):
        assert (
            geo.detour_ratio(L.get("two_rooms"))[
                ~np.eye(len(L.get("two_rooms").free_cells()), dtype=bool)
            ].max()
            > 3.0
        )
        assert geo.detour_ratio(L.get("spiral")).max() > 10.0

    def test_source_must_be_free(self):
        with pytest.raises(ValueError, match="not a free cell"):
            geo.geodesic_from(SMALL, (2, 2))


class TestGraphStructure:
    def test_every_maze_is_connected(self):
        for name in L.names():
            assert geo.is_connected(L.get(name)), name

    def test_two_rooms_cut_vertices_are_the_doorway_passage(self):
        # The wall sits in column 5 only, so crossing it means traversing
        # (4, 4) -> (4, 5) -> (4, 6); all three are cut vertices.
        assert geo.articulation_points(L.get("two_rooms")) == ((4, 4), (4, 5), (4, 6))

    def test_four_rooms_has_no_cut_vertices(self):
        # Four doorways mean there is always a second route. This is why
        # betweenness, not articulation, is the bottleneck ground truth.
        assert geo.articulation_points(L.get("four_rooms")) == ()

    def test_open_room_has_no_cut_vertices(self):
        assert geo.articulation_points(L.get("open_room")) == ()


class TestBetweenness:
    def test_loop_is_perfectly_uniform(self):
        # Every cell of a ring carries the same share of shortest paths.
        b = geo.betweenness_centrality(L.get("loop"))
        assert b.std() == pytest.approx(0.0, abs=1e-9)

    def test_two_rooms_peaks_at_the_doorway_passage(self):
        spec = L.get("two_rooms")
        top = set(geo.top_bottleneck_cells(spec, 3))
        assert top == {(4, 4), (4, 5), (4, 6)}

    def test_bottleneck_contrast_separates_walled_from_open(self):
        # The signal is the contrast against the open-room null, not the
        # absolute value.
        def peak_over_mean(name):
            b = geo.betweenness_centrality(L.get(name))
            return b.max() / b.mean()

        assert peak_over_mean("two_rooms") > 2 * peak_over_mean("open_room")

    def test_four_rooms_doorways_score_far_above_the_mean(self):
        # They rank 8th-11th rather than 1st-4th because each doorway's two
        # flanking cells carry its traffic plus intra-room traffic, so
        # detection must be scored with a one-cell tolerance.
        spec = L.get("four_rooms")
        b = geo.betweenness_centrality(spec)
        cells = list(spec.free_cells())
        for door in ((2, 5), (8, 5), (5, 2), (5, 8)):
            assert b[cells.index(door)] > 3 * b.mean()
